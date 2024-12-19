from accelerate import accelerator
import torch
from datasets import tqdm, load_dataset
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.trainer_callback import TrainerControl, TrainerState
from dataclasses import dataclass, field
from typing import Optional
from peft import LoraConfig
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import dill
from evaluate import load as load_metric
import torch
import os
import sys
import csv
import time
import psutil
import math
import traceback
from typing import Dict, Union, Any


def infer_txt_generator(prompt, modelI, tokenizerI):
    """
    Parameters
    prompt : str
        The input prompt for generating text.

    Returns
    generated_text : str
        The generated text based on the input prompt.

    Description This function performs text generation using a pre-trained language model. It takes an input prompt
    and generates text based on the prompt using the following steps:

    Encodes the input prompt using the tokenizer, returning a tensor representation of the input.
    Creates an attention mask tensor of ones with the same shape as the input tensor.
    If a CUDA-enabled GPU is available,
    moves the input tensor and attention mask tensor to the GPU and sets the model to use the GPU.
    Generates text
    using the model's generate method, passing the input tensor, attention mask, and the ID for the end-of-sentence
    token as the padding token ID.
    Decodes the generated output tensor into human-readable text, skipping any special tokens.
    Returns the generated text.
    """
    inputs = tokenizerI.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        attention_mask = attention_mask.to("cuda")
        modelI.to("cuda")
    outputs = modelI.generate(inputs,
                              attention_mask=attention_mask,
                              pad_token_id=tokenizerI.eos_token_id,
                              max_length=150)
    generated_text = tokenizerI.decode(outputs[0], skip_special_tokens=True)
    return generated_text


import math
from accelerate import Accelerator

def train_model_txt_generator(model_name, trainer_data, model_pass, accelerator: Accelerator, optimizer, scheduler_pass,
                              epoch_pass, custom_path=''):
    model_title = model_name.replace('/', '-').replace(':', '_')
    file_path = f'{custom_path}/{model_title}/{model_title}_training_results.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    model_pass.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")
    criterion = CrossEntropyLoss()

    with open(file_path, mode='w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Epoch', 'Batch', 'Training Loss', 'Time', 'Throughput (Seq/sec)', 'Disk Read IOPS', 'Disk Write IOPS'])

    try:
        for epochR in range(epoch_pass):
            training_losses = []
            disk_read_count = 0
            disk_write_count = 0
            start_batch = time.time()
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True)):
                    if batch is None:
                        continue
                    inputs, targets = batch
                    
                    # Check for NaN values in inputs and targets
                    if torch.isnan(inputs).any() or torch.isnan(targets).any():
                        print(f"NaN detected in inputs or targets at epoch {epochR}, batch {batch_num}")
                        continue
                    
                    batch_size = len(inputs)
                    optimizer.zero_grad()
                    outputs = model_pass(inputs)

                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
                    
                    # Check for NaN values in loss
                    if math.isnan(loss.item()):
                        print(f"NaN detected in loss at epoch {epochR}, batch {batch_num}")
                        continue

                    accelerator.backward(loss)
                    num_iterations += 1

                    training_losses.append(loss.item())  # Store the training loss

                    # Gradient Accumulation
                    if num_iterations == accumulation_steps:
                        # Gradient clipping
                        clip_grad_norm_(model_pass.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler_pass.step(loss)
                        num_iterations = 0

                    # Update disk IOPS counters
                    disk_io_counters = psutil.disk_io_counters()
                    disk_read_count += disk_io_counters.read_count
                    disk_write_count += disk_io_counters.write_count

                    # Calculate Batch time and throughput
                    end_batch = time.time()
                    batch_time = end_batch - start_batch

                    # Calculate throughput of the system for each batch
                    throughput = batch_size * (batch_num + 1) / batch_time

                    # Check for NaN values in throughput
                    if math.isnan(throughput):
                        print(f"NaN detected in throughput at epoch {epochR}, batch {batch_num}")
                        continue  # Skip this batch

                    # Calculate disk read and write IOPS for each batch
                    disk_read_iops = disk_read_count / batch_time
                    disk_write_iops = disk_write_count / batch_time

                    # Write the results for each batch
                    writer.writerow(
                        [epochR, batch_num, training_losses[-1], batch_time, throughput, disk_read_iops, disk_write_iops]
                    )

            accelerator.free_memory()
    except Exception as e:
        # Handle exception and log error message
        print(traceback.format_exc())
        # Close the CSV file properly
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)

def pretrain_model_txt_generator(model_name, training_data_loader, modelPM, accelerator: accelerator, optimizer, schedulerPM,
                                 total_epochs, custom_path=''):
    """
    Pretrain Model

    This function performs the pretraining process for a given model using the specified
    training data. It follows the following steps:

    1. Sets the model to training mode.
    2. Initializes variables for tracking training loss, number of iterations, and batch size.
    3. Sets the default tensor type to "torch.FloatTensor".
    4. Iterates through each epoch.
    5. Within each epoch, iterates through each batch of the training data.
    6. If the batch is None (empty), skips to the next iteration which accounts for the last batch being empty.
    7. Retrieves inputs and targets from the batch.
    8. Zeroes the gradients of the optimizer.
    9. Passes the inputs through the model to get the outputs.
    10. Calculates the loss using the specified criterion.
    11. Performs backward propagation of gradients using the accelerator (if available).
    12. Updates the model parameters based on the gradients and optimizer's update rule.
    13. Adjusts the learning rate using the scheduler.
    14. Performs gradient accumulation if the number of iterations reaches the specified accumulation steps.
    15. Frees memory using the accelerator.
    16. Calculates the epoch time and throughput.
    17. Saves the model's state dictionary to a file.
    18. Writes the training loss, epoch time, and throughput to a CSV file.

    Parameters
    ----------
    training_data_loader : DataLoader
        DataLoader object containing the training data for the model.
    modelPM : transformer.GPT2LMHeadModel
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    schedulerPM : torch.optim.lr_scheduler._LRScheduler
        The scheduler used for adjusting the learning rate during training.
    total_epochs : int
        The total number of epochs to pretrain the model.

    Returns
    -------
    None

    Note
    ----
    The validation aspect of the original function has been removed in this version.

    """
    model_title = model_name.replace('/', '-').replace(':', '_')
    file_path = f'{custom_path}/{model_title}/{model_title}_training_results.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Initialize Model mode and High level Values
    modelPM.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")
    criterion = CrossEntropyLoss()

    # Write the header to the CSV file
    with open(file_path, mode='w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Epoch',
                         'Batch',
                         'Training Loss',
                         'Time',
                         'Throughput (Seq/sec)',
                         'Disk Read IOPS',
                         'Disk Write IOPS'])

    try:
        for epochR in range(total_epochs):
            # Training loop
            start = time.time()
            training_losses = []  # Initialize list to store training losses
            disk_read_count = 0  # Initialize disk read counter
            disk_write_count = 0  # Initialize disk write counter
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(
                        tqdm(training_data_loader, desc=f'Epoch {epochR}, Batch', leave=True)):
                    if batch is None:
                        # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                        continue
                    inputs, targets = batch
                    batch_size = len(inputs)
                    optimizer.zero_grad()
                    outputs = modelPM(inputs)
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
                    accelerator.backward(loss)
                    num_iterations += 1

                    training_losses.append(loss.item())  # Store the training loss

                    # Gradient Accumulation
                    if num_iterations == accumulation_steps:
                        # Gradient clipping
                        clip_grad_norm_(modelPM.parameters(), max_grad_norm)
                        optimizer.step()
                        schedulerPM.step(loss)
                        num_iterations = 0

                    # Update disk IOPS counters
                    disk_io_counters = psutil.disk_io_counters()
                    disk_read_count = disk_io_counters.read_count
                    disk_write_count = disk_io_counters.write_count

                    # Calculate Batch time
                    end = time.time()
                    batch_time = end - start

                    # Calculate throughput of the system for each batch
                    throughput = batch_size * (batch_num + 1) / batch_time

                    # Calculate disk read and write IOPS for each batch
                    disk_read_iops = disk_read_count / batch_time
                    disk_write_iops = disk_write_count / batch_time

                    # Write the results for each batch
                    writer.writerow(
                        [epochR, batch_num, training_losses[-1], batch_time, throughput, disk_read_iops,
                         disk_write_iops])

            # Save the model's state dictionary after each epoch
            checkpoint_path = f'{model_title}/Checkpoints/{model_title}_epoch_{epochR}.pth'
            if not os.path.exists(os.path.dirname(f'{model_title}/Checkpoints/')):
                os.makedirs(os.path.dirname(f'{model_title}/Checkpoints/'))
            torch.save(modelPM.state_dict(), checkpoint_path)
            # Free memory
            accelerator.free_memory()
    except Exception as e:
        # Handle exception and log error message
        print(traceback.format_exc())
        # Close the CSV file properly
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)


def train_model_BERT(model_name, trainer_data, model_pass, accelerator: accelerator, optimizer, scheduler_pass, epoch_pass, custom_path=''):
    """
    Parameters
    ----------
    trainer_data : DataLoader
        DataLoader object containing the training data for the model.
    model_pass : transformer.GPT2LMHeadModel
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    scheduler_pass : torch.optim.lr_scheduler._LRScheduler
        The scheduler used for adjusting the learning rate during training.
    epoch_pass : int
        The total number of epochs to train the model.

    Returns
    -------
    None

    Description
    -----------
    This function performs the training process for a given model using the specified
    training data. It follows the following steps:

    1. Sets the model to training mode.
    2. Initializes variables for tracking training loss, number of iterations, and batch size.
    3. Sets the default tensor type to "torch.FloatTensor".
    4. Iterates through each epoch.
    5. Within each epoch, iterates through each batch of the training data.
    6. If the batch is None (empty), skips to the next iteration which accounts for the last batch being empty.
    7. Retrieves inputs and targets from the batch.
    8. Zeroes the gradients of the optimizer.
    9. Passes the inputs through the model to get the outputs.
    10. Calculates the loss using the specified criterion.
    11. Performs backward propagation of gradients using the accelerator (if available).
    12. Updates the model parameters based on the gradients and optimizer's update rule.
    13. Adjusts the learning rate using the scheduler.
    14. Performs gradient accumulation if the number of iterations reaches the specified accumulation steps.
    15. Frees memory using the accelerator.
    16. Calculates the epoch time and throughput.
    17. Saves the model's state dictionary to a file.
    18. Writes the training loss, epoch time, and throughput to a CSV file.

    Note
    ----
    The validation aspect of the original function has been removed in this version.

    """
    model_title = model_name.replace('/', '-').replace(':', '_')
    file_path = f'{custom_path}/{model_title}/{model_title}_training_results.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Initialize Model mode and High level Values
    model_pass.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")
    criterion = CrossEntropyLoss()

    # Write the header to the CSV file
    with open(file_path, mode='w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Epoch',
                         'Batch',
                         'Training Loss',
                         'Time',
                         'Throughput (Seq/sec)',
                         'Disk Read IOPS',
                         'Disk Write IOPS'])
    try:
        for epochR in range(epoch_pass):
            # Training loop
            training_losses = []  # Initialize list to store training losses
            start = time.time()
            disk_read_count = 0  # Initialize disk read counter
            disk_write_count = 0  # Initialize disk write counter
            torch.set_default_tensor_type("torch.FloatTensor")
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(
                        tqdm(trainer_data, desc=f'Training Epoch {epochR}, Batch', leave=True)):
                   
                    if batch is None:
                        # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                        continue
                    inputs, attention_masks, targets = batch
                    batch_size = len(inputs)
                    optimizer.zero_grad()
                    outputs = model_pass(inputs, attention_mask=attention_masks, labels=targets)

                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))

                    accelerator.backward(loss)
                    num_iterations += 1

                    training_losses.append(loss.item())  # Store the training loss

                    # Gradient Accumulation
                    if num_iterations == accumulation_steps:
                        # Gradient clipping
                        clip_grad_norm_(model_pass.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler_pass.step(loss)
                        num_iterations = 0

                    # Update disk IOPS counters
                    disk_io_counters = psutil.disk_io_counters()
                    disk_read_count += disk_io_counters.read_count
                    disk_write_count += disk_io_counters.write_count

                    # Calculate Batch time
                    end = time.time()
                    batch_time = end - start

                    # Calculate throughput of the system for each batch
                    throughput = batch_size * (batch_num + 1) / batch_time

                    # Calculate disk read and write IOPS for each batch
                    disk_read_iops = disk_read_count / batch_time
                    disk_write_iops = disk_write_count / batch_time

                    # Write the results for each batch
                    writer.writerow(
                        [epochR, batch_num, training_losses[-1], batch_time, throughput, disk_read_iops,
                         disk_write_iops])

            accelerator.free_memory()
    except Exception as e:
        # Handle exception and log error message
        print(traceback.format_exc())
        # Close the CSV file properly
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)


def pretrain_model_BERT(model_name, training_data_loader, modelPM, optimizer, schedulerPM, total_epochs, custom_path=''):
    """
    Pretrain Model

    This function performs the pretraining process for a given model using the specified
    training data. It follows the following steps:

    1. Sets the model to training mode.
    2. Initializes variables for tracking training loss, number of iterations, and batch size.
    3. Sets the default tensor type to "torch.FloatTensor".
    4. Iterates through each epoch.
    5. Within each epoch, iterates through each batch of the training data.
    6. If the batch is None (empty), skips to the next iteration which accounts for the last batch being empty.
    7. Retrieves inputs and targets from the batch.
    8. Zeroes the gradients of the optimizer.
    9. Passes the inputs through the model to get the outputs.
    10. Calculates the loss using the specified criterion.
    11. Performs backward propagation of gradients using the accelerator (if available).
    12. Updates the model parameters based on the gradients and optimizer's update rule.
    13. Adjusts the learning rate using the scheduler.
    14. Performs gradient accumulation if the number of iterations reaches the specified accumulation steps.
    15. Frees memory using the accelerator.
    16. Calculates the epoch time and throughput.
    17. Saves the model's state dictionary to a file.
    18. Writes the training loss, epoch time, and throughput to a CSV file.

    Parameters
    ----------
    training_data_loader : DataLoader
        DataLoader object containing the training data for the model.
    modelPM : transformer.GPT2LMHeadModel
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    schedulerPM : torch.optim.lr_scheduler._LRScheduler
        The scheduler used for adjusting the learning rate during training.
    total_epochs : int
        The total number of epochs to pretrain the model.

    Returns
    -------
    None

    Note
    ----
    The validation aspect of the original function has been removed in this version.

    """
    model_title = model_name.replace('/', '-').replace(':', '_')
    file_path = f'{custom_path}/{model_title}/{model_title}_training_results.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Initialize Model mode and High level Values
    modelPM.train()
    max_grad_norm = 1
    num_iterations = 0
    accumulation_steps = 1
    torch.set_default_tensor_type("torch.FloatTensor")
    criterion = CrossEntropyLoss()

    # Write the header to the CSV file
    with open(file_path, mode='w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(['Epoch',
                         'Batch',
                         'Training Loss',
                         'Time',
                         'Throughput (Seq/sec)',
                         'Disk Read IOPS',
                         'Disk Write IOPS'])

    try:
        for epochR in range(total_epochs):
            # Training loop
            start = time.time()
            training_losses = []  # Initialize list to store training losses
            disk_read_count = 0  # Initialize disk read counter
            disk_write_count = 0  # Initialize disk write counter
            with open(file_path, mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                for batch_num, batch in enumerate(
                        tqdm(training_data_loader, desc=f'Epoch {epochR}, Batch', leave=True)):
                    if batch is None:
                        # Skip the last empty batch (As the multicore Encoder returns NoneType for last index)
                        continue
                    inputs, targets = batch
                    batch_size = len(inputs)
                    optimizer.zero_grad()
                    outputs = modelPM(inputs)
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
                    accelerator.backward(loss)
                    num_iterations += 1

                    training_losses.append(loss.item())  # Store the training loss

                    # Gradient Accumulation
                    if num_iterations == accumulation_steps:
                        # Gradient clipping
                        clip_grad_norm_(modelPM.parameters(), max_grad_norm)
                        optimizer.step()
                        schedulerPM.step(loss)
                        num_iterations = 0

                    # Update disk IOPS counters
                    disk_io_counters = psutil.disk_io_counters()
                    disk_read_count += disk_io_counters.read_count
                    disk_write_count += disk_io_counters.write_count

                    # Calculate Batch time
                    end = time.time()
                    batch_time = end - start

                    # Calculate throughput of the system for each batch
                    throughput = batch_size * (batch_num + 1) / batch_time

                    # Calculate disk read and write IOPS for each batch
                    disk_read_iops = disk_read_count / batch_time
                    disk_write_iops = disk_write_count / batch_time

                    # Write the results for each batch
                    writer.writerow(
                        [epochR, batch_num, training_losses[-1], batch_time, throughput, disk_read_iops,
                         disk_write_iops])

            accelerator.free_memory()
    except Exception as e:
        # Handle exception and log error message
        print(traceback.format_exc())
        # Close the CSV file properly
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properly
        sys.exit(1)


def Grab_Tokenizer_Custom(tokenizer_type, pad_token='<pad>', eos_token='<eos>', bos_token='<bos>'):
    """
    Parameters
    tokenizer_type : str
        The type of tokenizer to retrieve (e.g., 'gpt2-large').
    pad_token : str
        The token used for padding.
    eos_token : str
        The token used to signify the end of a sequence.
    bos_token : str
        The token used to signify the beginning of a sequence.

    Returns
    tokenizerGrab : Tokenizer
        Tokenizer object for tokenizing text using the specified model.

    Description: This function retrieves a tokenizer from the specified pretrained model and configures it with custom
    tokens for padding, beginning of sequence, and end of sequence. It performs the following steps: Retrieves the
    tokenizer using AutoTokenizer.from_pretrained() with the specified model type. Sets the padding token to the
    specified pad_token. Sets the end-of-sequence token to the specified eos_token. Sets the beginning-of-sequence
    token to the specified bos_token. Returns the configured Tokenizer object.
    """

    tokenizerGrab = AutoTokenizer.from_pretrained(tokenizer_type, use_fast=True)
    tokenizerGrab.pad_token = pad_token
    tokenizerGrab.eos_token = eos_token
    tokenizerGrab.bos_token = bos_token
    return tokenizerGrab

def Grab_Tokenizer(tokenizer_type: str):
    """
    Parameters
    tokenizer_type : str
        The type of tokenizer to retrieve (e.g., 'gpt2-large').

    Returns
    tokenizerGrab : Tokenizer
        Tokenizer object for tokenizing text using the specified model.

    Description This function retrieves a tokenizer from the specified pretrained model.
    """
    tokenizerGrab = AutoTokenizer.from_pretrained(tokenizer_type)
    return tokenizerGrab


def collate_fn_txt_generator(batch):
    """
    Parameters
    batch : List
        A list of dictionaries, where each dictionary represents
        a batch item with 'input_ids' and 'attention_mask' keys.

    Returns
    input_ids : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the input IDs for each batch item.
    attention_masks : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the attention masks for each batch item.

    Description: This function is a collate function used in data loading for creating batches of data. It takes a
    batch of samples and performs the following steps:
        Extracts the 'input_ids' from each dictionary item in the batch.
        Extracts the 'attention_mask' from each dictionary item in the batch.
        Pads the 'input_ids' sequences to have the same length within the batch using the pad_sequence function.
        Pads the 'attention_masks' sequences to have the same length within the batch using the pad_sequence function.
        Returns the padded 'input_ids' and 'attention_masks' as tensors.
    """
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_masks = [item['attention_mask'].squeeze() for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)

    return input_ids, attention_masks


def collate_fn_BERT(batch):
    """
    Parameters
    batch : List
        A list of dictionaries, where each dictionary represents
        a batch item with 'input_ids', 'attention_mask', and 'token_type_ids' keys.

    Returns
    input_ids : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the input IDs for each batch item.
    attention_masks : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the attention masks for each batch item.
    token_type_ids : Tensor
        Padded tensor of shape (batch_size, max_sequence_length) containing the token type IDs for each batch item.

    Description: This function is a collate function used in data loading for creating batches of data. It takes a
    batch of samples and performs the following steps:
        Extracts the 'input_ids' from each dictionary item in the batch.
        Extracts the 'attention_mask' from each dictionary item in the batch.
        Extracts the 'token_type_ids' from each dictionary item in the batch.
        Pads the 'input_ids' sequences to have the same length within the batch using the pad_sequence function.
        Pads the 'attention_masks' sequences to have the same length within the batch using the pad_sequence function.
        Pads the 'token_type_ids' sequences to have the same length within the batch using the pad_sequence function.
        Returns the padded 'input_ids', 'attention_masks', and 'token_type_ids' as tensors.
    """
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_masks = [item['attention_mask'].squeeze() for item in batch]
    token_type_ids = [item['token_type_ids'].squeeze() for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)

    return input_ids, attention_masks, token_type_ids


def infer_BERT(prompt, tokenizer=None, model=None):
    """
    Parameters
    ----------
    prompt : str
        The input prompt for generating predictions.

    Description
    -----------
    This function performs text inference using a pre-trained language model. It takes an
    input prompt and generates predictions for masked tokens in the prompt using the following steps:

    1. Encode the input prompt using the tokenizers `encode_plus` method, returning a dictionary of input tensors.
    2. Locate the masked token(s) in the input tensor.
    3. If a CUDA-enabled GPU is available, move the input tensors and the model to the GPU.
    4. Disable gradient calculations by wrapping the following code block with `torch.no_grad()`.
    5. Generate output from the model by passing the input tensors as keyword arguments.
    6. Retrieve the logits from the output.
    7. Get the logits for the masked word(s) by indexing the logits tensor with the mask indices.
    8. Find the top 5 predicted tokens and their indices based on the highest logits.
    9. Calculate the probabilities of each token prediction by applying softmax to the mask word logits.
    10. Convert the top 5 token indices and their probabilities to lists.
    11. Write the predicted words and their probabilities to a CSV file.

    Note: The `tokenizer` and `model` variables used in this function need to be defined and available
    in the current scope.

    """
    # Encode the input prompt, looking for masked tokens
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    # Locate the masked token(s)
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    if torch.cuda.is_available():
        # Move everything to the GPU if available
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")
    with torch.no_grad():
        # Generate output from the model
        outputs = model(**inputs)

    # Retrieve the logits from the output
    logits = outputs.logits
    # Get the logits for the masked word(s)
    mask_word_logits = logits[0, mask_index, :]
    # Find the top 5 predicted tokens and their indices
    top_5_tokens = torch.topk(mask_word_logits, 5, dim=1).indices[0].tolist()
    # Calculate the probabilities of each token prediction
    probabilities = torch.nn.functional.softmax(mask_word_logits, dim=1)[0]
    top_5_token_probs = probabilities[top_5_tokens].tolist()

    # Prepare data for CSV
    csv_data = []
    for i, token in enumerate(top_5_tokens):
        word = tokenizer.decode([token])
        probability = top_5_token_probs[i]
        csv_data.append([word, probability])

    # Write data to CSV
    with open('Model_Sample_Inferences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Word", "Probability"])
        writer.writerows(csv_data)


@dataclass
class Finetune_GPTQ_Args:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="ybelkada/llama-7b-GPTQ-test",
        metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."}
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={"help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."},
    )
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

def get_project_directory():
    # Get the absolute path of the script being executed
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Navigate up two levels to get outside of 'hpc-profiler'
    project_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return project_directory

def create_and_prepare_GPTQ_model(model_name: str, dataset_name: str, lora_alpha: int, lora_dropout: float, lora_r: int, device_map: Optional[dict] = None, output_dir: str = get_project_directory()):
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
        print("=" * 80)

    # Count Number of GPUs Available
    if device_map is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device_map = {f"cuda:{i}": i for i in range(torch.cuda.device_count())}
            else:
                device_map = {"": 0}
        else:
            device_map = None
    
    os.makedirs(output_dir, exist_ok=True)
    cache_file_path = os.path.join(output_dir, 'quantized_model_peft_tokenizer.pkl')  # Define a file path within the directory
    # Check if the cache file already exists
    if os.path.exists(cache_file_path):
        try:
            print(f"Loading cached data from {cache_file_path}")
            with open(cache_file_path, 'rb') as cache_file:
                return dill.load(cache_file)
        except (EOFError, dill.UnpicklingError) as e:
            print(f"Failed to load cache file: {e}. Regenerating the model...")
    
    
    quantize_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantize_config,
        trust_remote_code=True
    )
    
    # Refer to this to explain why this is necessary: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 

    # Create Lora Config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Cache quantized model to disk
    # with open(cache_file_path, 'wb') as cache_file:
    #     dill.dump((model, peft_config, tokenizer), cache_file)
    #     print(f"Processed samples cached to {cache_file_path}")
    
    
    return model, peft_config, tokenizer


def finetune_gptq_model(args: Finetune_GPTQ_Args):
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    model, peft_config, tokenizer = create_and_prepare_GPTQ_model(args)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    dataset = load_dataset(args.dataset_name, split="train")

    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
    )

    trainer.train()

    if args.merge_and_push:
        output_dir = os.path.join(args.output_dir, "final_checkpoints")
        trainer.model.save_pretrained(output_dir)

        del model
        torch.cuda.empty_cache()

    return trainer.model
class SharedState:
    """
    Manages shared state information for logging and benchmarking during training.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.batch_num = 0
        self.epoch_start_time = 0
        self.disk_read_count = 0
        self.disk_write_count = 0

class LoggingCallback(TrainerCallback):
    """
    Custom TrainerCallback for logging training results.
    """
    def __init__(self, shared_state: SharedState):
        super().__init__()
        self.shared_state = shared_state
        
    def on_train_begin(self, args: TrainingArguments, trainer_state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.shared_state.log_file_path, 'w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['Epoch', 'Batch', 'Training Loss', 'Time', 'Throughput (Seq/sec)', 'Disk Read IOPS', 'Disk Write IOPS'])

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.shared_state.epoch_start_time = time.time()
        self.shared_state.disk_read_count = 0
        self.shared_state.disk_write_count = 0

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model_save_path = os.path.dirname(self.shared_state.log_file_path)
        model_filename = f"model_epoch_{state.epoch}.bin"
        model_path = os.path.join(model_save_path, model_filename)
        torch.save(state.model.state_dict(), model_path)
        print(f"Model saved to {model_path} after epoch {state.epoch}")

class Benchmark_Language_Model_Trainer(Trainer):
    """
    Custom Trainer class for language models.
    """
    def __init__(self, *args, shared_state: SharedState, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_state = shared_state
        self.batch_num = 0
        
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        loss = super().training_step(model, inputs)
        
        # Benchmark calculations
        disk_io_counters = psutil.disk_io_counters()
        disk_read_count = disk_io_counters.read_count
        disk_write_count = disk_io_counters.write_count
        end_time = time.time()
        batch_time = end_time - self.shared_state.epoch_start_time
        batch_size = inputs['input_ids'].shape[0]
        throughput = batch_size * (self.batch_num + 1) / batch_time
        disk_read_iops = disk_read_count / batch_time
        disk_write_iops = disk_write_count / batch_time
        epoch_num = math.floor(self.state.epoch)
        
        with open(self.shared_state.log_file_path, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch_num, self.batch_num, loss.item(), batch_time, throughput, disk_read_iops, disk_write_iops])
        
        self.batch_num += 1
        return loss

def compute_metrics(eval_pred):
    metric = load_metric("perplexity")
    logits, labels = eval_pred
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_language_model(model_name, train_data, epochs=3, lr=0.00006, batch_size=1, custom_path=''):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_title = model_name.replace('/', '-').replace(':', '_')
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_dir = os.path.join(parent_dir, custom_path, model_title)
    print("output_dir", output_dir)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_total_limit=3,
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    file_path = f"{output_dir}/{model_title}_training_results.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    shared_state = SharedState(file_path)
    
    trainer = Benchmark_Language_Model_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback(shared_state)],
        shared_state=shared_state
    )
    
    try:
        trainer.train()
    except Exception as e:
        print(traceback.format_exc())
        return None
    except KeyboardInterrupt:
        print("Training Loop Aborted")
        return None
    
    return model