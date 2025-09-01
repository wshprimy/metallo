import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List, Tuple
import logging
import cv2
import numpy as np
def process(image_path):
    """
    Process metallographic images with preprocessing.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Processed image as numpy array (uint8)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = img.copy()
    mask = cv2.medianBlur(mask, 3)
    
    mask = np.array([135.0, 135.0, 135.0], dtype=np.float32) - mask.astype(np.float32)
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    return mask


def transform(image, image_transform):
    """
    Apply transformations to the image.
    
    Args:
        image: PIL Image or numpy array
        image_transform: Transform to apply
    
    Returns:
        Transformed image tensor
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    return image_transform(image)


class Metallographic(Dataset):
    """
    Dataset for metallographic images.
    Each DOS folder contains multiple .tif images.
    """

    def __init__(
        self,
        data_dir: str,
        image_transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        is_val: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        images_per_dos: int = 100,
        process_images: bool = False,
    ):
        """
        Initialize metallographic dataset.

        Args:
            data_dir: Directory containing DOS folders with images
            image_transform: Image transformations
            is_train: Whether this is training set
            is_val: Whether this is validation set
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            images_per_dos: Number of images per DOS value (default 100)
            process_images: Whether to apply preprocessing (default False)
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.is_val = is_val
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.images_per_dos = images_per_dos
        self.process_images = process_images

        # Get DOS folders and sort them
        self.dos_folders = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, f))
            ]
        )

        # Extract DOS values from folder names
        self.dos_values = []
        for folder in self.dos_folders:
            try:
                dos_value = float(folder)
                self.dos_values.append(dos_value)
            except ValueError:
                logging.warning(f"Skipping folder {folder} - not a valid DOS value")
                continue

        # Set up image transforms
        if image_transform is None:
            self.image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(336),
                    transforms.CenterCrop(300),
                ]
            )
        else:
            self.image_transform = image_transform

        # Calculate split indices
        if self.is_train:
            self.start_idx = 0
            self.end_idx = int(self.images_per_dos * self.train_ratio)
        elif self.is_val:
            self.start_idx = int(self.images_per_dos * self.train_ratio)
            self.end_idx = int(
                self.images_per_dos * (self.train_ratio + self.val_ratio)
            )
        else:  # test
            self.start_idx = int(
                self.images_per_dos * (self.train_ratio + self.val_ratio)
            )
            self.end_idx = self.images_per_dos

        self.images_per_split = self.end_idx - self.start_idx

        logging.info(
            f"Metallographic: {len(self.dos_values)} DOS values, "
            f"{self.images_per_split} images per DOS, "
            f"total samples: {len(self)}"
        )

    def __len__(self) -> int:
        return len(self.dos_values) * self.images_per_split

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        """
        Get image and corresponding DOS value.

        Args:
            index: Sample index

        Returns:
            Tuple of (image_tensor, dos_value)
        """
        # Calculate which DOS folder and which image within that folder
        dos_idx = index // self.images_per_split
        img_idx = index % self.images_per_split + self.start_idx

        dos_value = self.dos_values[dos_idx]
        dos_folder = self.dos_folders[dos_idx]

        # Get image files in the DOS folder
        dos_path = os.path.join(self.data_dir, dos_folder)
        image_files = sorted(
            [
                f
                for f in os.listdir(dos_path)
                if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
            ]
        )

        if img_idx >= len(image_files):
            # If we don't have enough images, cycle through available ones
            img_idx = img_idx % len(image_files)

        image_path = os.path.join(dos_path, image_files[img_idx])

        # Load and transform image
        if self.process_images:
            image = process(image_path)
            image_tensor = transform(image, self.image_transform)
        else:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)

        return image_tensor, dos_value
