import os
import torch
from PIL import Image
import numpy as np

import torchvision.transforms as TT
import torchvision.transforms.functional as TF

import random

__all__ = ['VOC']

class Transform:
    """a custom transformer for segmentation image pairs"""
    def __init__(self, size): #size is a tuple representing the final image dimension
        self.size = size
        self.nrmlz = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target):

        #apply random resize crop
        rrcrop = TT.RandomResizedCrop(self.size)
        (i, j, h, w) = rrcrop.get_params(image, rrcrop.scale, rrcrop.ratio)
        image_augmented = TF.resized_crop(image, i, j, h, w, self.size, Image.NEAREST)
        target_augmented = TF.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)

        #apply random flips
        if random.random() > 0.5:
            image_augmented = TF.hflip(image_augmented)
            target_augmented = TF.hflip(target_augmented)

        if random.random() > 0.5:
            image_augmented = TF.vflip(image_augmented)
            target_augmented = TF.vflip(target_augmented)

        #Make tensor
        image_augmented_tensor = TF.to_tensor(image_augmented)
        #the target is a tensor of ints
        target_augmented_tensor = torch.as_tensor(
            np.array(target_augmented), dtype=torch.int64
        )

        #Normalize image
        image_augmented_tensor = self.nrmlz(image_augmented_tensor)

        return image_augmented_tensor, target_augmented_tensor

class VOC(torch.utils.data.Dataset):
    """VOC Segmentation dataset object, written by hand because I think the
    transforms are messed up so I am making my own

    I should rebalance the train and validation sets"""

    def __init__(self, root: str, split: str):

        #construct paths and file names
        image_path = os.path.join(root, 'VOC2012', 'JPEGImages')
        target_path = os.path.join(root, 'VOC2012', 'SegmentationClass')
        #extract files names from target because jpegimages has many unlabled images
        annotation_path = os.path.join(
            root, 'VOC2012', 'ImageSets', 'Segmentation', split+'.txt'
        )
        with open(annotation_path, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]

        #future optimization could be to build static name list input/target
        #instead of constructing the paths on the fly in __getitem__

        #build transformer (for data augmentations)
        self.transform = Transform((480, 480))
        self.image_path = image_path
        self.target_path = target_path
        self.file_names = file_names
        self.num_classes = 21

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        #extract images from disk and store as PIL images
        image_location = os.path.join(self.image_path, self.file_names[index] + '.jpg')
        target_location = os.path.join(self.target_path, self.file_names[index] + '.png')

        image = Image.open(image_location).convert('RGB')
        target = Image.open(target_location)

        #perform transforms
        image, target = self.transform(image, target)

        return image, target
