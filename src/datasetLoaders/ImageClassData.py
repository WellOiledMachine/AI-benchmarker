# Author: Cody Sloan
# Project: AI Benchmarking
# Model: Image Classifier
# Backend: PyTorch
# Last Modified: 08/22/2024

import os
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms
from torchvision.transforms.functional import InterpolationMode
from workloads.ImageClassificationTools.sampler import RASampler
from workloads.ImageClassificationTools import presets
from workloads.ImageClassificationTools import transforms

class HF_DatasetLoader:
    """
    Parameters:
    ----------
    args: argparse.Namespace
        The arguments object containing all the necessary parameters for the image classification workload.
        This object should be created using the get_args_parser() function from ImageClassW.py.
        It will contain all of the necessary parameters for this class to function properly as well.
    Description:
    -----------
    This class is used to load any image classification dataset using the HuggingFace Datasets library.
    The dataset will be downloaded from HF using the provided id, and will be split into training and 
    testing sets based on the dataset configuration. Train and test DataLoader objects are returned when the load() 
    method is called. This class will define the transformations that will be used on the data as well.
    """
    def __init__(self, args):
        # This is so the cifar10 dataset can be loaded without specifying too many defaults in the argument parser funciton
        if args.dataset=='cifar10':
            args.image_column = 'img'
            args.label_column = 'label'
            args.train_split = 'train'
            args.test_split = 'test'
        
        # Ensure that both column titles are provided and are non-empty strings.
        if not args.image_column or not args.label_column:
            raise ValueError("Both data column titles of the dataset must be provided. \
                Make sure to set image_column and label_column to the correct column titles \
                found on the dataset's page on HuggingFace.")
        
        self.args = args
        self.dataset = args.dataset
        self.train_split = args.train_split
        self.test_split = args.test_split
        if args.split_ratio <= 0.0 or args.split_ratio >= 1.0:
            raise ValueError("split_ratio must be a value between 0.0 and 1.0 (exclusive).")
        self.split_ratio = args.split_ratio
        self.image_column = args.image_column
        self.label_column = args.label_column
        self.val_resize_size = args.val_resize_size
        self.val_crop_size = args.val_crop_size
        self.train_crop_size = args.train_crop_size
        self.backend = args.backend
        self.interpolation = InterpolationMode(args.interpolation)
        self.train_transformations = self.create_train_transformations()
        self.test_transformations = self.create_test_transformations()
        self.dataset_train, self.dataset_test, self.num_classes = self.load_data()  # Load data and get num_classes

    def create_train_transformations(self):
        return presets.ClassificationPresetTrain(
            crop_size=self.train_crop_size,
            interpolation=self.interpolation,
            auto_augment_policy=getattr(self.args, "auto_augment", None),
            random_erase_prob=getattr(self.args, "random_erase", 0.0),
            ra_magnitude=getattr(self.args, "ra_magnitude", None),
            augmix_severity=getattr(self.args, "augmix_severity", None),
            backend=self.backend,
        )

    def create_test_transformations(self):
        return presets.ClassificationPresetEval(
            crop_size=self.val_crop_size,
            resize_size=self.val_resize_size,
            interpolation=self.interpolation,
            backend=self.backend,
        )

    def load_data(self):
        dataset_train = None
        dataset_test = None
        if os.path.isdir(self.dataset):
            dataset = load_from_disk(self.dataset)
        else:
            dataset = load_dataset(self.dataset, trust_remote_code=True)

        if isinstance(dataset, DatasetDict):
            if self.train_split in dataset and self.test_split in dataset:
                dataset_train = dataset[self.train_split]
                dataset_test = dataset[self.test_split]
            else:
                available_splits = list(dataset.keys())
                if len(available_splits) >= 2:
                    print(f"Could not find the specified training and testing splits in the dataset. Attempting "
                          f"to use pre-existing splits: \"{available_splits[0]}\" and \"{available_splits[1]}\"")
                    dataset_train = dataset[available_splits[0]]
                    dataset_test = dataset[available_splits[1]]
                else:
                    print(f"Could not find the specified training and testing splits in the dataset. Creating "
                          f"new splits using a ratio of {1-self.split_ratio} from the only available pre-existing split: {available_splits[0]}")
                    dataset = dataset[available_splits[0]]
                    split_datasets = dataset.train_test_split(test_size=self.split_ratio)
                    dataset_train = split_datasets['train']
                    dataset_test = split_datasets['test']
        
        if isinstance(dataset, Dataset):
            split_datasets = dataset.train_test_split(test_size=self.split_ratio)
            dataset_train = split_datasets['train']
            dataset_test = split_datasets['test']

        if not dataset_train or not dataset_test:
            raise ValueError("Could not find the training and testing splits in the dataset. \
                Make sure you have the correct split names and that the dataset has been loaded correctly.")

        # Set the transformations for the datasets
        dataset_train.set_transform(self.train_transformation_wrapper)
        dataset_test.set_transform(self.test_transformation_wrapper)

        # print(dataset_train.info)
        # Use label_column to index the features attribute of the dataset to get the number of classes.
        num_classes = dataset_train.features[self.label_column].num_classes  # Should work for most HF datasets

        return dataset_train, dataset_test, num_classes

    def train_transformation_wrapper(self, batch):
        images, labels = batch[self.image_column], batch[self.label_column]
        transformed_images = [self.train_transformations(image) for image in images]
        return {self.image_column: transformed_images, self.label_column: labels}

    def test_transformation_wrapper(self, batch):
        images, labels = batch[self.image_column], batch[self.label_column]
        transformed_images = [self.test_transformations(image) for image in images]
        return {self.image_column: transformed_images, self.label_column: labels}

    def create_dataloaders(self):
        if self.args.distributed:
            if self.args.ra_sampler:
                train_sampler = RASampler(self.dataset_train, shuffle=True, repetitions=self.args.ra_reps)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(self.dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(self.dataset_test)

        train_dataloader = DataLoader(
            self.dataset_train, 
            batch_size=self.args.batch_size, 
            sampler=train_sampler, 
            num_workers=self.args.workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            self.dataset_test, 
            batch_size=self.args.batch_size, 
            sampler=test_sampler, 
            num_workers=self.args.workers,
            pin_memory=True
        )

        return train_dataloader, test_dataloader, train_sampler, self.num_classes

    def collate_fn(self, batch):
        mixup_transforms = []
        if self.args.mixup_alpha > 0.0:
            mixup_transforms.append(transforms.RandomMixup(self.num_classes, p=1.0, alpha=self.args.mixup_alpha))
        if self.args.cutmix_alpha > 0.0:
            mixup_transforms.append(transforms.RandomCutmix(self.num_classes, p=1.0, alpha=self.args.cutmix_alpha))
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms) if mixup_transforms else default_collate
        return mixupcutmix(batch)

    def load(self):
        return self.create_dataloaders()





""" LEGACY CODE: The same dataset can be loaded by using the HF_DatasetLoader class with the following parameters:
    dataset_id = 'cifar10'
    train_split = 'train'
    test_split = 'test'
    image_column = 'img'
    label_column = 'label'
"""
# # Define wrapper functions to apply the transformations to every image in the provided batchs. 
# # Need to do this for both training and testing datasets because the transformations for each are different.
# class CIFAR10Loader:
#     """
#     Parameters:
#     ----------
#     args: argparse.Namespace
#         The arguments object containing all the necessary parameters for the image classification workload.
#         This object should be created using the get_args_parser() function from ImageClassW.py.
#         It will contain all of the necessary parameters for this class to function properly as well.

#     Description:
#     -----------
#     This class is used to load the CIFAR-10 dataset using the HuggingFace Datasets library.
#     Train and test DataLoader objects are returned when the load() method is called. This class will define
#     the transformations that will be used on the data as well.

#     Link: https://huggingface.co/datasets/uoft-cs/cifar10
#     """
#     def __init__(self, args):
#         self.args = args
#         self.val_resize_size = args.val_resize_size
#         self.val_crop_size = args.val_crop_size
#         self.train_crop_size = args.train_crop_size
#         self.backend = args.backend
#         self.interpolation = InterpolationMode(args.interpolation)
#         self.train_transformations = self.create_train_transformations()
#         self.test_transformations = self.create_test_transformations()
#         self.dataset_train, self.dataset_test, self.num_classes = self.load_data()  # Load data and get num_classes

#     def create_train_transformations(self):
#         return presets.ClassificationPresetTrain(
#             crop_size=self.train_crop_size,
#             interpolation=self.interpolation,
#             auto_augment_policy=getattr(self.args, "auto_augment", None),
#             random_erase_prob=getattr(self.args, "random_erase", 0.0),
#             ra_magnitude=getattr(self.args, "ra_magnitude", None),
#             augmix_severity=getattr(self.args, "augmix_severity", None),
#             backend=self.backend,
#         )

#     def create_test_transformations(self):
#         return presets.ClassificationPresetEval(
#             crop_size=self.val_crop_size,
#             resize_size=self.val_resize_size,
#             interpolation=self.interpolation,
#             backend=self.backend,
#         )

#     def load_data(self):
#         dataset_train = load_dataset('cifar10', split='train', trust_remote_code=True)
#         dataset_train.set_transform(self.train_transformation_wrapper)

#         dataset_test = load_dataset('cifar10', split='test', trust_remote_code=True)
#         dataset_test.set_transform(self.test_transformation_wrapper)
        
#         num_classes = dataset_train.features["label"].num_classes  # Assuming num_classes is accessible like this

#         return dataset_train, dataset_test, num_classes

#     def train_transformation_wrapper(self, batch):
#         images, labels = batch['img'], batch['label']
#         transformed_images = [self.train_transformations(image) for image in images]
#         return {'img': transformed_images, 'label': labels}

#     def test_transformation_wrapper(self, sample):
#         images, labels = sample['img'], sample['label']
#         transformed_images = [self.test_transformations(image) for image in images]
#         return {'img': transformed_images, 'label': labels}

#     def create_dataloaders(self):
#         if self.args.distributed:
#             if self.args.ra_sampler:
#                 train_sampler = RASampler(self.dataset_train, shuffle=True, repetitions=self.args.ra_reps)
#             else:
#                 train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_train)
#             test_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_test, shuffle=False)
#         else:
#             train_sampler = torch.utils.data.RandomSampler(self.dataset_train)
#             test_sampler = torch.utils.data.SequentialSampler(self.dataset_test)

#         train_dataloader = DataLoader(
#             self.dataset_train, 
#             batch_size=self.args.batch_size, 
#             sampler=train_sampler, 
#             num_workers=self.args.workers,
#             collate_fn=self.collate_fn,
#             pin_memory=True
#         )
        
#         test_dataloader = DataLoader(
#             self.dataset_test, 
#             batch_size=self.args.batch_size, 
#             sampler=test_sampler, 
#             num_workers=self.args.workers,
#             pin_memory=True
#         )

#         return train_dataloader, test_dataloader, train_sampler, self.num_classes

#     def collate_fn(self, batch):
#         mixup_transforms = []
#         if self.args.mixup_alpha > 0.0:
#             mixup_transforms.append(transforms.RandomMixup(self.num_classes, p=1.0, alpha=self.args.mixup_alpha))
#         if self.args.cutmix_alpha > 0.0:
#             mixup_transforms.append(transforms.RandomCutmix(self.num_classes, p=1.0, alpha=self.args.cutmix_alpha))
#         mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms) if mixup_transforms else default_collate
#         return mixupcutmix(batch)

#     def load(self):
#         return self.create_dataloaders()