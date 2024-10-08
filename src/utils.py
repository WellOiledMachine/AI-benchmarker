# Authors: Cody Sloan
# project: AI Benchmarking
# Date: 08/22/2024

import os
import json
from argparse import ArgumentParser

def load_config_file(args: ArgumentParser):
    # Load the configuration file if it exists
    if args.config != '':
        with open(args.config, 'r') as f:
            config = json.load(f)
        for key in config:
            normalized_key = key.replace('-', '_') # Make sure user can use hyphens in the config file
            setattr(args, normalized_key, config[key])

def configure_output_dir(directory: str):
    """
    Parameters
    ----------
    directory : str
        The path to the output directory. Use whatever format your OS uses
        (e.g., forward slashes on Unix systems, backslashes on Windows).
    
    Create the output directory if it does not exist. If it does exist, check if it is empty.
    If it is not empty, raise a ValueError. If it is empty, add a 'checkpoints' subdirectory.
    """
    if not directory:
        raise ValueError("Output directory path required. Please provide a valid path, existing or not (must be empty).")
    if os.path.exists(directory):
        if os.listdir(directory):
            raise ValueError("Directory is not empty")

    os.makedirs(os.path.join(directory, 'checkpoints'), exist_ok=True)
    
def download_hf_dataset(dataset_name, output_dir, fraction=1.0):
    """
    Parameters
    ----------
    dataset_name : str
        The name of the Hugging Face dataset to download.
    output_dir : str
        The path to the output directory. Use whatever format your OS uses
        (e.g., forward slashes on Unix systems, backslashes on Windows).
    
    Download the Hugging Face dataset with the given name to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    from datasets import load_dataset, DatasetDict

    dataset = load_dataset(dataset_name)

    # If fraction is less than 1.0, take a subset of the dataset
    if fraction < 1.0:
        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                dataset[split] = dataset[split].train_test_split(test_size=1-fraction)['train']
        else:
            dataset = dataset.train_test_split(test_size=1-fraction)['train']

    dataset.save_to_disk(output_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download a Hugging Face dataset')
    parser.add_argument('--dataset', type=str, required=True, help='The ID of the Hugging Face dataset to download')
    parser.add_argument('--output', type=str, required=True, help='The path to the output directory')
    parser.add_argument('--fraction', type=float, default=1.0, help='The fraction of the dataset to download')
    args = parser.parse_args()
    download_hf_dataset(args.dataset, args.output, args.fraction)
