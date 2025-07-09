# data.py
# Defines how to load and transform the MRI images for the model.

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BrainTumourDataset(Dataset):
    """
    Custom PyTorch Dataset for loading brain tumour MRI slices.
    It handles loading the .mat files and applying image augmentations.
    """
    def __init__(self, data_folder, filenames, labels, is_train=True):
        """
        Args:
            data_folder (str): Path to the folder containing the .mat files.
            filenames (list): List of filenames to load.
            labels (list): List of corresponding labels.
            is_train (bool): If True, applies data augmentation.
        """
        self.data_folder = data_folder
        self.filenames = filenames
        self.labels = labels
        self.is_train = is_train

        # --- Define the image processing pipeline ---
        
        # Start with basic transformations for all images
        transform_steps = [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3), # ResNet needs 3 channels
            transforms.Resize((224, 224))
        ]

        # If this is the training set, add data augmentation
        # Augmentation makes the model more robust by showing it modified images
        if self.is_train:
            transform_steps.extend([
                transforms.RandomRotation(degrees=7),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Width/Height shifts of ±5%
                transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)), # Zoom of ±10%
                transforms.ColorJitter(brightness=0.1, contrast=0.1) # Brightness changes
    
            ])

        # Add final steps for all images
        transform_steps.extend([
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            # Normalize with ImageNet stats, as the pre-trained model expects this
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Gets a single image and its label by index."""
       
        filepath = os.path.join(self.data_folder, self.filenames[index])

        with h5py.File(filepath, "r") as f:
            image = f["cjdata"]["image"][()] # [()] extracts the data as a NumPy array

        # --- Pre-processing ---
        # Convert to float and scale pixel values to be between 0 and 1
        image = image.astype(np.float32)
        image /= image.max()
        
        # Apply the defined transformations (resizing, augmentation, etc.)
        image_tensor = self.transform(image)
        
        # Get the label and convert it to a tensor. The labels are 1, 2, 3,
        # but PyTorch expects them to start from 0, so we subtract 1.
        label = torch.tensor(self.labels[index] - 1, dtype=torch.long)
        
        return image_tensor, label