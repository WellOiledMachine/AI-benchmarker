# Authors: Tyson Limato
# project: HPC Profiler
# Date: 2024-05-09
import os
import csv
import time
import torch
import psutil
import traceback
import os
import sys
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, Seq2SeqTrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback 
from transformers.trainer_seq2seq import Seq2SeqTrainer
import math
import time
from typing import Dict, Union, Any, Literal
from accelerate import Accelerator
from datasetLoaders.TTSData import TTSDataCollatorWithPadding, print_dict_structure
from torch.nn.utils.rnn import pad_sequence
from functools import partial
# Training Loop for TTS model with fSorward pass and backward pass distributed via Huggingface Accelerate.
# We can not use the Trainer Class provideed by huggingface as we need to be able to log training performance and system analytics.
# A manual Training loop also allows us to more transparently communicate to the user what performance enhancing methods (gradient clipping, quantization, data parallel, etc) are being used.

class SharedState: # Written by Cody Sloan, Refactored for TTS.
    """
    Parameters
    ----------
    log_file_path : str
        The path to the csv log file where the training results will be written.
        
    Description
    -----------
    Provides a way to communicate information between the LoggingCallback and the Benchmark_Image_Segmentation_Trainer
    classes.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.batch_num = 0
        self.epoch_start_time = 0
        self.disk_read_count = 0
        self.disk_write_count = 0

class LoggingCallback(TrainerCallback):  # Written by Cody Sloan, Refactored by Tyson Limato for TTS.
    """
    Parameters
    ----------
    shared_state : SharedState
        The shared state object that will be used to communicate information between this class and the
        Benchmark_Image_Segmentation_Trainer class.
            
    Description
    -----------
    This is a custom TrainerCallback class used to initialize the log file for the training results.
    It resets the start time and disk IOPS counters at the beginning of each epoch to be used in the training loop.
    It stores information needed in the training loop in the state object, which is part of the Trainer class. This
    allows us to reset epoch specific information at the beginning of each epoch, and then use that information in
    the training loop.
    
    Link: https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.TrainerCallback
    
    """
    def __init__(self, shared_state: SharedState):
        super().__init__()
        self.shared_state = shared_state
        
    def on_train_begin(self, args: Seq2SeqTrainingArguments, trainer_state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.shared_state.log_file_path, 'w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['Epoch',
                         'Batch',
                         'Training Loss',
                         'Time',
                         'Throughput (Seq/sec)',
                         'Disk Read IOPS',
                         'Disk Write IOPS'])

        
    def on_epoch_begin(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.shared_state.epoch_start_time = time.time()
        self.shared_state.disk_read_count = 0
        self.shared_state.disk_write_count = 0

class Benchmark_TTS_Trainer(Seq2SeqTrainer): # Written by Cody Sloan, Refactored by Tyson Limato for TTS.
    """
    A custom Seq2SeqTrainer for benchmarking Text-to-Speech (TTS) model training.

    This class extends the Hugging Face Seq2SeqTrainer to include additional benchmarking
    capabilities such as tracking disk I/O operations per second (IOPS) and calculating
    throughput. It logs these metrics along with the training loss for each batch.

    Attributes:
        shared_state (SharedState): An object to share state information between the trainer and callbacks.
        batch_num (int): The current batch number.
        initial_disk_read_count (int): Initial disk read count at the start of each epoch.
        initial_disk_write_count (int): Initial disk write count at the start of each epoch.

    Methods:
        on_epoch_begin(*args, **kwargs): Initializes disk I/O counters and epoch start time at the beginning of each epoch.
        training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]): Performs a training step and logs benchmarking metrics.
    """

    def __init__(self, *args, shared_state: SharedState, **kwargs):
        """
        Initializes the Benchmark_TTS_Trainer.

        Parameters:
            shared_state (SharedState): An object to share state information between the trainer and callbacks.
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)
        self.shared_state = shared_state
        self.batch_num = 0
        self.initial_disk_read_count = 0
        self.initial_disk_write_count = 0

    def on_epoch_begin(self, *args, **kwargs):
        """
        Initializes disk I/O counters and epoch start time at the beginning of each epoch.

        Parameters:
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments for the base class.
        """
        super().on_epoch_begin(*args, **kwargs)
        # Initialize disk I/O counters at the start of each epoch
        disk_io_counters = psutil.disk_io_counters()
        self.initial_disk_read_count = disk_io_counters.read_count
        self.initial_disk_write_count = disk_io_counters.write_count
        self.shared_state.epoch_start_time = time.time()

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        # Debugging: Print the inputs to check for None values
        #for key, value in inputs.items():
            #if value is None:
            #    print(f"Warning: {key} is None")
            #else:
            #    print(f"{key} shape: {value.shape}")

        # Perform the normal training step defined by the Trainer class

        #TODO: THIS IS THE PROBLEM AREA
        
        loss = super().training_step(model, inputs)
        
        # Calculate results for the benchmark
        disk_io_counters = psutil.disk_io_counters()
        disk_read_count = disk_io_counters.read_count - self.initial_disk_read_count
        disk_write_count = disk_io_counters.write_count - self.initial_disk_write_count
        
        # Calculate batch time
        end_time = time.time()
        batch_time = end_time - self.shared_state.epoch_start_time
        
        # Calculate throughput of the system for this batch only
        batch_size = inputs['input_ids'].shape[0]  # Adjusted to input_ids
        throughput = batch_size / batch_time
        
        # Calculate disk read and write IOPS for this batch
        disk_read_iops = disk_read_count / batch_time
        disk_write_iops = disk_write_count / batch_time
        
        # Get integer epoch number
        epoch_num = math.floor(self.state.epoch)
        
        # Write the results for each batch
        with open(self.shared_state.log_file_path, 'a', newline='') as log_file:
            writer = csv.writer(log_file)  
            writer.writerow([epoch_num,
                            self.batch_num,
                            loss.item(),
                            batch_time,
                            throughput,
                            disk_read_iops,
                            disk_write_iops])
        
        # Update the batch number
        self.batch_num += 1
        
        return loss

def train_tts_model(model: any, model_name: Union[str, any], processor: Union[None, Any], train_data, optimizer=None, scheduler=None, epochs=3, batch_size=1, precision: Literal['fp16','fp32', 'bf16'] = 'fp16', custom_path=''):
    """
    Trains a Text-to-Speech (TTS) model using the specified parameters and dataset.

    This function sets up the training environment, initializes the model and processor,
    defines training arguments, and starts the training loop. It also handles logging and
    benchmarking of training performance.

    Parameters:
        model_name (Union[str, any]): The name of the model to load from Hugging Face or a pre-initialized model object.
        processor (Union[None, Any]): The processor to use for data preprocessing, either as a string name or a pre-initialized processor object.
        train_data (Dataset): The dataset to use for training.
        optimizer (Optional): The optimizer to use for training. If None, the default optimizer will be used.
        scheduler (Optional): The learning rate scheduler to use for training. If None, the default scheduler will be used.
        epochs (int, optional): The number of training epochs. Default is 3.
        batch_size (int, optional): The batch size to use for training. Default is 1.
        precision (Literal['fp16', 'fp32', 'bf16'], optional): The precision to use for training. Default is 'fp16'.
        custom_path (str, optional): Custom path for saving the model and training results. Default is an empty string.

    Returns:
        model: The trained TTS model.

    Raises:
        Exception: If an error occurs during training, the exception is caught and the traceback is printed.
        KeyboardInterrupt: If the training loop is interrupted by the user, the interruption is handled gracefully.

    Notes:
        - The function uses `Seq2SeqTrainer` for training, which is suitable for sequence-to-sequence tasks like TTS.
        - The function logs training performance and system analytics, including disk IOPS and throughput.
        - The function supports mixed precision training with options for 'fp16', 'fp32', and 'bf16'.
        - The function handles caching of training results to avoid reprocessing data unnecessarily.

    Example:
        >>> model = train_tts_model(
                model_name="facebook/wav2vec2-large-xlsr-53",
                processor="facebook/wav2vec2-large-xlsr-53",
                train_data=train_dataset,
                epochs=5,
                batch_size=4,
                precision='fp16',
                custom_path='my_custom_path'
            )
    """
    if isinstance(model_name, str) and model is not None:
        model = load_tts_model(model_name, model_class="SpeechT5ForTextToSpeech")
    else:
        model = model
    if isinstance(processor, str):
        processor = load_processor(processor, processor_class="SpeechT5Processor")
    else:
        processor = processor
    # Define the title of the model for file saving
    model_title = model_name.replace('/', '-').replace(':', '_')
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_dir = os.path.join(parent_dir, custom_path, model_title)
    print("output_dir", output_dir)

    match precision:
        case 'bf16':
            bf16=True
            fp16=False
            fp32=False
        case 'fp16':
            bf16=False
            fp16=True
            fp32=False
        case 'fp32':
            bf16=False
            fp16=False
            fp32=True
    # Define the training arguments
    training_args = create_training_args(
        output_dir=output_dir,
        max_steps=4000,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=fp16,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=1,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=['labels'],
        push_to_hub=False
        )
    # Reference this to see why evaluate doesn't support TTS evaluation.
    # https://huggingface.co/learn/audio-course/en/chapter6/evaluation
    """If we are willing to go outside of the huggingface ecosystem we can use the following package to provide some evaluation metrics for TTS.
    https://pypi.org/project/tts-scores/"""

    # Define the path to save the training results
    file_path = f"{output_dir}/{model_title}_training_results.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
         
    shared_state = SharedState(file_path)
    
    print("Printing the first entry of the train data")
    print_dict_structure(train_data[0])
    # Preprocess the train_data using the data collator too many issues doing it on the fly in the Trainer
    data_collator = TTSDataCollatorWithPadding(processor=processor, model=model)
    preprocessed_train_data = [data_collator([example]) for example in tqdm(train_data, desc="Padding train data")]
    
    print("Printing the first entry of the Preprocessed train data")
    print_dict_structure(preprocessed_train_data[0])
    # Configure model
    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False
    # set language and task for generation and re-enable cache
    #model.generate = partial(model.generate, use_cache=True)
    # Initialize the Trainer with the preprocessed data
    trainer = Benchmark_TTS_Trainer(
        args=training_args,
        model=model,
        train_dataset=preprocessed_train_data,
        data_collator=None,  # No need for data collator as data is already preprocessed
        callbacks=[LoggingCallback(shared_state)],
        tokenizer=processor,
        shared_state=shared_state,
    )
    
    try:
        trainer.train()
    except Exception:
        # Handle exception and log error message
        print(traceback.format_exc())
        # Close the CSV file properly
    except KeyboardInterrupt:
        # Handle exception and log error message
        print("Training Loop Aborted")
        # Close the CSV file properlys
    
    return model

# def custom_train_tts_model(model, train_data, optimizer, epochs=3, batch_size=1, device='cuda', processor=None, checkpoint_dir='checkpoints', save_every_n_batches=100, save_every_n_epochs=1):
#     model = model.to(device)
#     # set language and task for generation and re-enable cache
#     model.generate = partial(model.generate, use_cache=True)
#     model.train()
#     train_data = train_data.select(range(100))
#     # Ensure checkpoint directory exists
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # Initialize the data collator
#     data_collator = TTSDataCollatorWithPadding(processor=processor, model=model)
#     preprocessed_train_data = [data_collator([example]) for example in tqdm(train_data, desc="Padding train data")]
#     print("Printing the first entry of the preprocessed train data")
#     print_dict_structure(preprocessed_train_data[0])


#     for epoch in range(epochs):
#         print(f"Starting epoch {epoch+1}/{epochs}")
#         epoch_start_time = time.time()
#         total_loss = 0

#         for batch_idx, batch in tqdm(enumerate(preprocessed_train_data), total=len(preprocessed_train_data), desc="Finetuning TTS Model"):
#             # Use the data collator to process the batch
#             # Extract tensors and move them to the device
#             input_ids = batch['input_values'].to(device)
#             speaker_embeddings = batch['speaker_embeddings'].to(device)
#             labels = batch['attention_mask'].to(device)

#             # Forward pass with checkpointing
#             outputs = model.forward(input_ids=input_ids, speaker_embeddings=speaker_embeddings, labels=labels)

#             loss = outputs.loss

#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             # Checkpoint model every n batches
#             if (batch_idx + 1) % save_every_n_batches == 0:
#                 batch_checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}_batch_{batch_idx+1}.pt')
#                 torch.save(model.state_dict(), batch_checkpoint_path)
#                 print(f"Saved checkpoint to {batch_checkpoint_path}")

#             if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
#                 print(f"Batch {batch_idx+1}, Loss: {loss.item()}")

#         avg_loss = total_loss / len(preprocessed_train_data)
#         epoch_time = time.time() - epoch_start_time
#         print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}. Time taken: {epoch_time}s")

#         # Checkpoint model every n epochs
#         if (epoch + 1) % save_every_n_epochs == 0:
#             epoch_checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
#             torch.save(model.state_dict(), epoch_checkpoint_path)
#             print(f"Saved checkpoint to {epoch_checkpoint_path}")

#     print("Training completed.")
#     return model

# loads a TTS model from the Hugging Face model hub, some of these are included in the transformers library.
# To accomidate this the function checks if its present in the transformers library and dynamically imports it for usage
# If the model is not present in the transformers library it is assumed to be a custom model and is loaded from the model hub.
def load_tts_model(model_name, model_class=None):
    """
    Load a TTS model from the Hugging Face `transformers` library based on the checkpoint.

    Args:
        model_name (str): The model checkpoint identifier from Hugging Face model hub.
        model_class (str, optional): The class name of the model to load. If None, AutoModel is used.

    Returns:
        model: An instance of the TTS model.
    """
    if model_class:
        # Dynamically import the model class from transformers
        module = __import__('transformers', fromlist=[model_class])
        model_class = getattr(module, model_class)
        model = model_class.from_pretrained(model_name, trust_remote_code=True)
    else:
        # Use AutoModel if no specific class is provided
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    return model

# Loads a processor from the Hugging Face model hub, some of these are included in the transformers library.
# To accomidate this the function checks if its present in the transformers library and dynamically imports it for usage
# If the processor is not present in the transformers library it is assumed to be a custom processor and is loaded from the model hub.
def load_processor(model_name, processor_class=None):
    """
    Load a processor from the Hugging Face `transformers` library dynamically based on the checkpoint.

    Args:
        checkpoint (str): The model checkpoint identifier from Hugging Face model hub.
        processor_class (str, optional): The class name of the processor to load. If None, AutoProcessor is used.

    Returns:
        processor: An instance of the processor.
    """
    if processor_class:
        # Dynamically import the processor class from transformers
        module = __import__('transformers', fromlist=[processor_class])
        processor_class = getattr(module, processor_class)
        processor = processor_class.from_pretrained(model_name, trust_remote_code=True)
    else:
        # Use AutoProcessor if no specific class is provided
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return processor

# Dynamically get the project directory, this is useful for loading data from the project directory.
def get_project_directory():
    # Get the absolute path of the script being executed
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Navigate up two levels to the project directory
    project_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return project_directory

# Get custom data collator for TTS training, this is useful for padding the data for TTS training.
def get_data_collator(processor, model):
    data_collator = TTSDataCollatorWithPadding(processor=processor, model=model)
    return data_collator

def create_training_args(output_dir, max_steps=4000, learning_rate=2e-4, num_train_epochs=3, batch_size=1, 
                         gradient_checkpointing=True, bf16=False, fp16=True, save_total_limit=3, 
                         eval_strategy="steps", save_strategy="steps", save_steps=1000, eval_steps=1000, 
                         logging_steps=1, eval_accumulation_steps=1, load_best_model_at_end=True, 
                         greater_is_better=False, label_names=['labels'], push_to_hub=False):
    """
    Create and return Seq2SeqTrainingArguments with default or specified values.

    Parameters:
    - output_dir (str): Directory where outputs will be saved.
    - max_steps (int): Maximum number of training steps.
    - learning_rate (float): Learning rate for the optimizer.
    - num_train_epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training and evaluation.
    - gradient_checkpointing (bool): Enable gradient checkpointing to save memory.
    - bf16 (bool): Use bfloat16 precision.
    - fp16 (bool): Use float16 precision.
    - fp32 (bool): Use float32 precision.
    - save_total_limit (int): Maximum number of checkpoints to keep.
    - eval_strategy (str): Evaluation strategy.
    - save_strategy (str): Saving strategy.
    - save_steps (int): Number of steps before saving a checkpoint.
    - eval_steps (int): Number of steps before performing evaluation.
    - logging_steps (int): Number of steps before logging.
    - eval_accumulation_steps (int): Number of steps for accumulating gradients during evaluation.
    - load_best_model_at_end (bool): Load the best model at the end of training.
    - greater_is_better (bool): Determines if a higher metric value is better.
    - label_names (list): Names of the labels.
    - push_to_hub (bool): Whether to push the model to the Hugging Face Hub.

    Returns:
    - Seq2SeqTrainingArguments: Configured training arguments.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        save_total_limit=save_total_limit,
        eval_strategy=eval_strategy,  # Corrected from eval_strategy
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        load_best_model_at_end=load_best_model_at_end,
        greater_is_better=greater_is_better,
        label_names=label_names,
        push_to_hub=push_to_hub
    )
    return training_args