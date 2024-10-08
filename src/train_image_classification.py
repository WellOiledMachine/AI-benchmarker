from datasetLoaders.ImageClassData import HF_DatasetLoader
from workloads.ImageClassW import train_classification_model
import Profiler as Profiler
from graphingScripts.PlotFunctions import all_inclusive_plots, measure_time
from utils import configure_output_dir, load_config_file
import time
import argparse
import os

def get_parsed_args(add_help=True):
    """
    Create an Argument Parser for PyTorch Classification Training

    This function sets up an argument parser using Python's argparse module, specifically tailored for configuring a PyTorch-based image classification training session. It defines and configures all necessary command-line arguments that control various aspects of the training process, such as model selection, device settings, training hyperparameters, and more.

    Parameters
    ----------
    add_help : bool, optional
        If True, the parser will include a help option (default is True).

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.

    Processes
    ----------
    - Initializes an argparse.ArgumentParser instance.
    - Adds arguments to the parser that allow the user to customize:
        - Model details (e.g., model type, device to use).
        - Training parameters (e.g., batch size, number of epochs, learning rate).
        - Optimization settings (e.g., optimizer type, weight decay).
        - Advanced training options (e.g., mixed precision training, distributed training).
        - Data augmentation and regularization techniques.
        - Checkpointing and result output configurations.
    - Each argument is clearly defined with a help message explaining its purpose and default value.

    Notes
    -----
    - This function is crucial for allowing flexible and configurable training runs. By using command-line arguments, users can easily adjust the training settings without modifying the code.
    - The parser supports a wide range of training configurations, making it suitable for various training scenarios and model architectures.
    - It is important to ensure that all default values align with the expected usage scenarios and that they are appropriately set based on the training context.
    """
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--output", default=None, type=str, help="path to save outputs")

    parser.add_argument('--config', type=str, default='', help='Path to config file')

    parser.add_argument('--sampling-rate', type=float, default=1.0, help='Sampling rate for updating the metrics in the benchmark. Default is 1.0 seconds, which is also the minimum value allowed.')

    # Added for HuggingFace DatasetLoader
    parser.add_argument("--dataset", default="cifar10", type=str, help="Either a path to a downloaded HF dataset, or the dataset-id to a dataset on the HF Hub")
    parser.add_argument("--image-column", type=str, help="the column name for the image in the dataset")
    parser.add_argument("--label-column", type=str, help="the column name for the label in the dataset")
    parser.add_argument("--train-split", default="train", type=str, help="the split for training shown in the datasets card on the HF hub")
    parser.add_argument("--test-split", type=str, help="the split for testing shown in the datasets card on the HF hub")
    parser.add_argument("--split-ratio", default=0.2, type=float, help="the ratio of the dataset to use for training. \
                        this will only be used if the train or test splits cannot be found in the dataset. \
                        Will error if set outside of (0.0, 1.0) range")
    
    # Original arguments
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--epochs", default=1, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    parser.add_argument(
        "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 1)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Whether to use distributed training")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    return parser.parse_args()

def main():
    # Retreive the arguments from the command line
    args = get_parsed_args()

    load_config_file(args)

    configure_output_dir(args.output) 
    print("Loading dataset...")
    start = time.time()
    # Load the dataset. Will error if the splits and column_titles are not correct for your dataset.
    loader = HF_DatasetLoader(args)
    train_dataloader, test_dataloader, train_sampler, num_classes = loader.load()
    print(f"Done. Dataset loaded in {time.time() - start:.2f} seconds.")

    monitor = Profiler.monitor_system_utilization(interval=args.sampling_rate, output_dir=args.output)
    with monitor:
        start_time = time.time()
        # Train the model using specifications from the args object
        train_classification_model(args, train_dataloader, test_dataloader, train_sampler, num_classes)
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    # Generate all the plots
    all_inclusive_plots(args.output, 'graphs')
    
if __name__ == "__main__":
    execution_time = measure_time(main)
    print(f"Total execution time: {execution_time:.2f} seconds")
    