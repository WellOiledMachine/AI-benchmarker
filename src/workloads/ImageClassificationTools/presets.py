# This code is based on an original implementation found at https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# Modifications have been made by Cody Sloan and Tyson Limato
# Project: AI Benchmarking
# Last Updated: 08/22/2024

import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

def ConvertToRGB(img, backend="pil"):
    """Makes sure the image is represented using RGB color space so the model will accept it."""
    if backend == "pil":
        if img.mode != 'RGB':
            img = img.convert('RGB')
    elif backend == "tensor":
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3, :, :]
    return img


class ClassificationPresetTrain:
    """
    Transformations for training image classification models. Initialize this class by providing at the very least the
    crop size. The other parameters are optional and have default values.
    Calling the object with an image will apply the transformations to the image and return them.
    """
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
    ):
        backend = backend.lower() # Convert to lowercase for easier comparison

        # Make sure that the images are in RGB format (have 3 channels) regardless of the backend
        color_convert = lambda image: ConvertToRGB(image, backend=backend) # Convert to RGB if grayscale, using the specified backend
        trans = [color_convert]
        
         # If the backend is tensor, add a PILToTensor transform to convert the image to a tensor immediately
        if backend == "tensor":
            trans.append(transforms.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        trans.append(transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            trans.append(transforms.PILToTensor())

        trans.extend(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    """
    Transformations for testing image classification models. Initialize this class by providing at the very least the
    crop size. The other parameters are optional and have default values.
    Calling the object with an image will apply the transformations to the image and return them.
    """
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
    ):
        backend = backend.lower() # Convert to lowercase for easier comparison

        # Make sure that the images are in RGB format (have 3 channels) regardless of the backend
        color_convert = lambda image: ConvertToRGB(image, backend=backend) # Convert to RGB if grayscale, using the specified backend
        trans = [color_convert]
        
        # If the backend is tensor, add a PILToTensor transform to convert the image to a tensor immediately
        if backend == "tensor":
            trans.append(transforms.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        trans += [
            transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
        ]

        if backend == "pil":
            trans.append(transforms.PILToTensor())

        trans += [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
