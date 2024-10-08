# Author: Cody Sloan
# Project: GPU Benchmarking
# Model: Train semantic segmentation model (Segformer) on sidewalk dataset
# Backend: PyTorch

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
import evaluate
import torch
from torch import nn
import csv
import psutil
import math
import time
from typing import Dict, Union, Any
import os
import sys
import numpy as np

class SharedState:
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

class LoggingCallback(TrainerCallback):
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
        
    def on_train_begin(self, args: TrainingArguments, trainer_state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.shared_state.log_file_path, 'w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                'Timestamp (Unix)',
                'Epoch',
                'Batch',
                'Training Loss',
                'Execution Time (sec)',
                'Throughput (Seq/sec)',
                'Disk Read IOPS',
                'Disk Write IOPS'
                ])

        
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.shared_state.epoch_start_time = time.time()
        self.shared_state.disk_read_count = 0
        self.shared_state.disk_write_count = 0
        
        
class Benchmark_Image_Segmentation_Trainer(Trainer):
    """
    Parameters
    ----------
    shared_state : SharedState
        The shared state object that will be used to communicate information between this class and the
        LoggingCallback class.
    *args, **kwargs :
        Additional arguments that are passed to the Trainer class. This class accepts the same parameters as the
        Trainer class.
    
    Description
    -----------
    This is a custom Trainer class that overrides the training_step method to calculate and write the training results
    to the log file. It uses the shared_state object to communicate information with the LoggingCallback class,
    which is needed to track information at each epoch start.
    
    Link: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer
    """
    def __init__(self, *args, shared_state: SharedState, **kwargs):
        # Pass Trainer class arguments to the Trainer class
        super().__init__(*args, **kwargs)
        
        # Initialize the shared state object and the batch number
        self.shared_state = shared_state
        self.batch_num = 0
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        inputs : Dict[str, Union[torch.Tensor, Any]]
            The input data for the current batch training step.
            
        Returns
        -------
        loss : torch.Tensor
            The loss value for the current training step.
            
        Description
        -----------
        This function overrides the training_step method of the Trainer class to calculate and write the training
        results to the log file. It calls the normal training_step method of the Trainer class to perform the
        training, and then calculates the batch time, throughput, disk read IOPS, and disk write IOPS for the current
        batch and writes them to the log file. Some of the information needed for the calculations is stored in the
        shared_state object, which is updated in the LoggingCallback class.
        """
        
        # Perform the normal training step defined by the Trainer class
        loss = super().training_step(model, inputs)
        
        # vvv Calculate results for the benchmark vvv
        # Update disk IOPS counters
        disk_io_counters = psutil.disk_io_counters()
        disk_read_count = disk_io_counters.read_count
        disk_write_count = disk_io_counters.write_count
        
        # Calculate batch time
        end_time = time.time()
        batch_time = end_time - self.shared_state.epoch_start_time
        
        # Calculate throughput of the system for each batch
        batch_size = inputs['pixel_values'].shape[0]
        throughput = batch_size * (self.batch_num + 1) / batch_time
        
        # Calculate disk read and write IOPS for each batch
        disk_read_iops = disk_read_count / batch_time
        disk_write_iops = disk_write_count / batch_time
        
        # Get integer epoch number
        epoch_num = math.floor(self.state.epoch)
        
        # Write the results for each batch
        with open(self.shared_state.log_file_path, 'a', newline='') as log_file:
            writer = csv.writer(log_file)  
            writer.writerow([time.time(),
                            epoch_num,
                            self.batch_num,
                            loss.item(),
                            batch_time,
                            throughput,
                            disk_read_iops,
                            disk_write_iops])
        
        # Update the batch number
        self.batch_num += 1
        
        return loss
        

def train_image_segmentation_model(args, train_data, test_data, id2label):
    """
    Parameters
    args : argparse.ArgumentParser
        The arguments object that contains all of the necessary information to train an Image Segmentation model
    train_data : datasets.arrow_dataset.Dataset
        Training dataset for the model.
    test_data : datasets.arrow_dataset.Dataset
        Testing dataset for the model.
    id2label : dict
        Dictionary mapping label ids to label names.
        
    Returns
    -------
    model : transformers.models.segformer.modeling_segformer.SegformerForSemanticSegmentation
        The trained model.
            
    Description
    -----------
    This function trains an image segmentation model using the Segformer architecture on the sidewalk dataset.
    It uses the custom Trainer class Benchmark_Image_Segmentation_Trainer and the custom TrainerCallback class
    LoggingCallback to calculate and write the training results to a log file. All training is handled by the
    Trainer class from the transformers library. 
    """
    

    # Create reverse mapping of id2label, where the key is the label and the value is the id
    label2id = {v: k for k, v in id2label.items()}
    
    # Load the model
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    # Check if GPU is available
    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        # print("Using GPU:", torch.cuda.get_device_name(device.index))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Move the model to the device
    model.to(device)
    
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        push_to_hub=False,
        remove_unused_columns=False,
        use_cpu=(args.device=='cpu')
    )
    # Define the metric to be used for evaluation
    metric = evaluate.load("mean_iou")
    
    # The model outputs logits with dimensions height/4 and width/4,
    # so we need to upscale them before computing the metric.    
    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = torch.nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            # currently using _compute instead of compute
            # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
            metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=False,
            )

            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()

        return metrics
        
    # Define the path to save the training results
    
    file_path = os.path.join(args.output, "training_results.csv")
        
    shared_state = SharedState(file_path)
    
    # Define the trainer
    trainer = Benchmark_Image_Segmentation_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback(shared_state)],
        shared_state=shared_state
    )

    trainer.train()
    
    return model

    