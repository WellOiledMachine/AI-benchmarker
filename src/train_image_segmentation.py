import os
import argparse
import time

from datasetLoaders.ImageSegData import open_SidewalkSemantics, open_scene_parse_150
from workloads.ImageSegW import train_image_segmentation_model
from Profiler import monitor_system_utilization
from utils import load_config_file, configure_output_dir
from graphingScripts.PlotFunctions import all_inclusive_plots

def get_parsed_args():
    parser = argparse.ArgumentParser(description='Train a model for image segmentation')
    parser.add_argument('--config', type=str, default='', help='Path to a configuration file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--model', type=str, default='nvidia/mit-b0', help='Model id. CHANGING IS NOT RECOMMENDED')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('-lr' ,'--learning-rate', type=float, default=0.00006, help='Learning rate')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu', help='Device to use for training (gpu or cpu)')

    return parser.parse_args()
    


def main():
    args = get_parsed_args() # Get the arguments from the command line
    load_config_file(args)  # Load the configuration file if it exists
    configure_output_dir(args.output) # Create the output directory if it does not exist

    train_data, test_data, id2label = open_SidewalkSemantics()
    monitor = monitor_system_utilization(interval=1, output_dir=args.output)
    with monitor:
        start_time = time.time()
        model = train_image_segmentation_model(args, train_data, test_data, id2label)
        total_time = round(time.time() - start_time, 2)
        print("Training complete. Took", total_time, "seconds")

    save_dir = os.path.join(args.output, 'checkpoints')
    if model is not None:
        model.save_pretrained(save_directory=save_dir)
        print("Model saved")
    else:
        print("Model not saved")

    all_inclusive_plots(args.output, 'graphs')
    print("Program complete")

if __name__ == "__main__":
    main()
    
    