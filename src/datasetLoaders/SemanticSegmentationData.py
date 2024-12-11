# Author: Cody Sloan
# Contributor: Tyson Limato
# Project: GPU Benchmarking
# Model: Train semantic segmentation model (Segformer) on sidewalk dataset
# Backend: PyTorch

from datasets import load_dataset
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
import json
from huggingface_hub import hf_hub_download
import os
import numpy as np

class GeneralizedDatasetLoader: # Added by Tyson Limato, Requires Validation. Use https://huggingface.co/datasets/EduardoPacheco/FoodSeg103
    """
    A generalized dataset loader class.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load.
    label_map : str
        Path to label mapping file or Hugging Face repo (format: 'repo_id:filename')
    seed : int, optional
        The value used to initialize the random number generator for reproducible data splits.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    preprocess_params : dict, optional
        Parameters for preprocessing functions such as brightness, contrast, saturation, and hue.

    Description
    -----------
    This class loads a dataset, splits it into training and testing data, preprocesses the data,
    and can load additional resources like id2label mappings from a Hugging Face repository.
    """
    def __init__(self, dataset_name, label_map, seed=42, test_size=0.1, preprocess_params=None):
        self.data = load_dataset(dataset_name, split='train')
        self.data = self.data.shuffle(seed=seed)
        self.data = self.data.train_test_split(test_size=test_size)

        self.label_map = label_map

        self.jitter = ColorJitter(**(preprocess_params or {'brightness': 0.25, 'contrast': 0.25, 'saturation': 0.25, 'hue': 0.1}))
        self.feature_extractor = SegformerImageProcessor()

        

    def apply_transforms(self, example_batch, augment=False):
        images = [self.jitter(x) for x in example_batch["pixel_values"]] if augment else [x for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = self.feature_extractor.preprocess(images, labels)
        return inputs

    def get_train_data(self):
        train_ds = self.data["train"]
        train_ds.set_transform(lambda x: self.apply_transforms(x, augment=True))
        return train_ds

    def get_test_data(self):
        test_ds = self.data["test"]
        test_ds.set_transform(lambda x: self.apply_transforms(x, augment=False))
        return test_ds

    def get_id2label_mapping(self):
        if not self.label_map:
            raise ValueError("Label mapping file must be provided to fetch label mapping.")
        if ':' not in self.label_map:
            try:
                label_dict = json.loads(open(self.label_map, 'r').read())
            except:
                raise ValueError(f"Could not load label mapping from file: {self.label_map}")
        else:
            repo_id, filename = self.label_map.split(':')
            label_map_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
            with open(label_map_path, 'r') as file:
                label_dict = json.load(file)

        first_key = list(label_dict.keys())[0]
        first_value = label_dict[first_key]

        if (isinstance(first_key, str) and first_key.isdigit()) or isinstance(first_key, int):
            id2label = {int(k): v for k, v in label_dict.items()}
        elif (isinstance(first_value, str) and first_value.isdigit()) or isinstance(first_value, int):
            id2label = {int(v): k for k, v in label_dict.items()}
        
        return id2label

    def __len__(self):
        return len(self.data)

def open_dataset(dataset_name, label_map, seed=42, test_size=0.1, preprocess_params=None, ): # Added by Tyson Limato, Requires Validation. Use https://huggingface.co/datasets/EduardoPacheco/FoodSeg103
    """
    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load.
    seed : int, optional
        The value used to initialize the random number generator for reproducible data splits.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    preprocess_params : dict, optional
        Parameters for preprocessing functions such as brightness, contrast, saturation, and hue.
    repo_id : str, optional
        The repository ID on Hugging Face from which to download additional resources like id2label mapping.
    filename : str, optional
        The filename of the additional resource to download from the repository.

    Returns
    -------
    train_data : datasets.arrow_dataset.Dataset
        Dataset object containing the training data for the model.
    test_data : datasets.arrow_dataset.Dataset
        Dataset object containing the testing data for the model.
    id2label : dict
        Dictionary mapping label ids to label names.

    Description
    -----------
    This function loads and preprocesses the training and validation data files for an image segmentation model using
    a specified dataset. It performs the following steps: loads the data files, splits the data into training and testing sets,
    preprocesses the data, loads the labels of the dataset, and creates dictionaries to map label ids to label names.
    Returns the Dataset objects for training and testing data, and the label dictionaries.
    """
    
    print(f'Loading Training Data Files for {dataset_name}...')
    ds = GeneralizedDatasetLoader(dataset_name, label_map, seed=seed, test_size=test_size, preprocess_params=preprocess_params)
    train_data = ds.get_train_data()
    test_data = ds.get_test_data()
    id2label = ds.get_id2label_mapping()
    
    return train_data, test_data, id2label


class SidewalkSemanticsDataset:
    """
    Parameters
    ----------
    seed : int, optional
        The value used to initialize the random number generator for reproducible data splits.

    Description
    -----------
    This class is a dataset loader for the Sidewalk Semantics dataset. It loads the dataset, splits it into
    training and testing data, preprocesses the training and testing data splits, and loads a dictionary
    containing the mappings of every id and label.

    Link: https://huggingface.co/datasets/segments/sidewalk-semantic
    """
    def __init__(self, seed):
        # Load the sidewalk-semantics dataset. There is only a single split in
        # this dataset, which is the "train" split
        self.data = load_dataset('segments/sidewalk-semantic', split='train')
        
        # Split the data into training and testing sets
        self.data = self.data.shuffle(seed=seed)
        self.data = self.data.train_test_split(test_size=0.1)
        
        # Initialize preprocessing functions
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        self.feature_extractor = SegformerImageProcessor()
    
    def train_transforms(self, example_batch):
        # Apply color jitter to the images and preprocess the inputs
        images = [self.jitter(x) for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = self.feature_extractor.preprocess(images, labels)
        return inputs

    def val_transforms(self, example_batch):
        # Testing data doesn't need augmentation, so no color jitter is applied
        # Preprocess the inputs
        images = [x for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = self.feature_extractor.preprocess(images, labels)
        return inputs

    def get_train_data(self):
        # Apply the transforms and return the training dataset
        train_ds = self.data["train"]
        train_ds.set_transform(self.train_transforms)
        return train_ds
    
    def get_test_data(self):
        # Apply the transforms and return the testing dataset
        test_ds = self.data["test"]
        test_ds.set_transform(self.val_transforms)
        return test_ds
    
    def get_id2label(self):
        # Download the id2label file from the dataset repository
        id2label = json.load(
            open(
                hf_hub_download(
                    repo_id='segments/sidewalk-semantic', 
                    filename='id2label.json', 
                    repo_type='dataset'
                ),
                'r',
            )
        )
        
        # Create dictionaries of mapped labels and return them
        id2label = {int(k): v for k, v in id2label.items()}
        
        return id2label
        

    def __len__(self):
        # Return the total length of the dataset
        return len(self.data)
    

class scene_parse_150_Dataset:
    """
    Description
    -----------
    This class is a dataset loader for the scene parse 150 segmentation dataset. It loads the dataset, splits it into
    training and testing data, preprocesses the training and testing data splits, and loads a dictionary
    containing the mappings of every id and label.

    Link: https://huggingface.co/datasets/zhoubolei/scene_parse_150
    """
    def __init__(self):
        # Load the dataset with its splits
        self.train_data = load_dataset('zhoubolei/scene_parse_150', split='train', trust_remote_code=True)
        self.test_data = load_dataset('zhoubolei/scene_parse_150', split='validation', trust_remote_code=True)
        
        # Initialize preprocessing functions
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        self.feature_extractor = SegformerImageProcessor(do_reduce_labels=True)
    
    def train_transforms(self, example_batch):
        # Apply color jitter to the images and preprocess the inputs
        images = [self.jitter(x) for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = self.feature_extractor.preprocess(images, labels)
        return inputs

    def val_transforms(self, example_batch):
        # Testing data doesn't need augmentation, so no color jitter is applied
        # Preprocess the inputs
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["annotation"]]
        inputs = self.feature_extractor.preprocess(images, labels)
        return inputs

    def get_train_data(self):
        # Apply the transforms and return the training dataset
        self.train_data.set_transform(self.train_transforms)
        return self.train_data
    
    def get_test_data(self):
        # Apply the transforms and return the testing dataset
        self.test_data.set_transform(self.val_transforms)
        return self.test_data
    
    def get_id2label(self):
        # Download the id2label file from the dataset repository
        id2label = json.load(
            open(
                hf_hub_download(
                    repo_id='huggingface/label-files', 
                    filename='ade20k-id2label.json', 
                    repo_type='dataset'
                ),
                'r',
            )
        )
        
        # Create dictionaries of mapped labels and return them
        id2label = {int(k): v for k, v in id2label.items()}
        
        return id2label
        

    def __len__(self):
        # Return the total length of the dataset
        return len(self.data)


def open_SidewalkSemantics(seed=42):
    """
    Parameters
    ----------
    seed : int, optional
        The value used to initialize the random number generator for reproducible data splits.

    Returns
    -------
    train_data : datasets.arrow_dataset.Dataset
        Dataset object containing the training data for the model.
    test_Data : datasets.arrow_dataset.Dataset
        Dataset object containing the testing data for the model.
    id2label : dict
        Dictionary mapping label ids to label names.
    

    Description
    -----------
    This function loads and preprocesses the training and validation data files for an Image Segmentation model using
    the Sidewalk Semantics dataset. It performs the following steps: Loads the data files. Splits the data into training
    testing sets. Preprocesses the data. Loads the labels of the dataset. Creates dictionaries to map label ids to label names.
    Returns the Dataset objects for training and testing data, and the label dictionaries.
    """
    
    print('Loading Training Data Files...')
    ds = SidewalkSemanticsDataset(seed=seed)
    train_data = ds.get_train_data()
    test_data = ds.get_test_data()
    id2label = ds.get_id2label()
    
    return train_data, test_data, id2label 

def open_scene_parse_150():
    """
    Returns
    -------
    train_data : datasets.arrow_dataset.Dataset
        Dataset object containing the training data for the model.
    test_Data : datasets.arrow_dataset.Dataset
        Dataset object containing the testing data for the model.
    id2label : dict
        Dictionary mapping label ids to label names.
    

    Description
    -----------
    This function loads and preprocesses the training and validation data files for an Image Segmentation model using
    the scene_parse_150 dataset. It performs the following steps: Loads the data files. Splits the data into training
    testing sets. Preprocesses the data. Loads the labels of the dataset. Creates dictionaries to map label ids to label names.
    Returns the Dataset objects for training and testing data, and the label dictionaries.
    """
    
    print('Loading Training Data Files...')
    ds = scene_parse_150_Dataset()
    train_data = ds.get_train_data()
    test_data = ds.get_test_data()
    id2label = ds.get_id2label()
    
    return train_data, test_data, id2label 
