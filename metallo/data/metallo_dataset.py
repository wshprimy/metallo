import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
import logging
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import torchvision.transforms as transforms


class MetalloDataset(Dataset):
    """
    Custom PyTorch dataset for metallographic (image) and spectral (CSV) data.
    Supports both image-only, spectral-only, and multimodal (image + spectral) tasks.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        spectral_data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        image_transform: Optional[transforms.Compose] = None,
        scale_spectral: bool = True,
        train_spectral_data: Optional[np.ndarray] = None,
        mode: str = "multimodal",  # "image", "spectral", "multimodal"
    ):
        """
        Initialize the dataset.

        Args:
            image_dir: Directory containing metallographic images
            spectral_data: Pre-loaded numpy array with spectral data (float32)
            labels: Target labels for the task
            image_transform: Torchvision transforms for images
            scale_spectral: Whether to apply StandardScaler to spectral data
            train_spectral_data: Training spectral data for fitting scaler
            mode: Dataset mode - "image", "spectral", or "multimodal"
        """
        self.image_dir = image_dir
        self.spectral_data = spectral_data
        self.labels = labels
        self.mode = mode
        self.scale_spectral = scale_spectral

        # Validate mode and data consistency
        if mode == "image" and image_dir is None:
            raise ValueError("image_dir must be provided for image mode")
        if mode == "spectral" and spectral_data is None:
            raise ValueError("spectral_data must be provided for spectral mode")
        if mode == "multimodal" and (image_dir is None or spectral_data is None):
            raise ValueError(
                "Both image_dir and spectral_data must be provided for multimodal mode"
            )

        # Set up image transforms
        if image_transform is None:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.image_transform = image_transform

        # Set up spectral data scaling
        self.spectral_scaler = None
        if self.spectral_data is not None:
            assert isinstance(
                spectral_data, np.ndarray
            ), f"spectral_data must be numpy array, got {type(spectral_data)}"
            assert (
                spectral_data.dtype == np.float32
            ), f"spectral_data must be float32, got {spectral_data.dtype}"

            if self.scale_spectral:
                self.spectral_scaler = StandardScaler()
                fit_data = (
                    train_spectral_data
                    if train_spectral_data is not None
                    else spectral_data
                )
                self.spectral_scaler.fit(fit_data)
                logging.info(f"Fitted spectral scaler on {len(fit_data)} samples")

        # Get image file list if using images
        self.image_files = []
        if self.image_dir is not None:
            self.image_files = sorted(
                [
                    f
                    for f in os.listdir(image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
                ]
            )
            logging.info(f"Found {len(self.image_files)} images in {image_dir}")

        # Determine dataset size
        if mode == "image":
            self.data_size = len(self.image_files)
        elif mode == "spectral":
            self.data_size = len(self.spectral_data)
        else:  # multimodal
            self.data_size = min(len(self.image_files), len(self.spectral_data))

        # Validate labels if provided
        if self.labels is not None:
            assert (
                len(self.labels) == self.data_size
            ), f"Labels length {len(self.labels)} doesn't match data size {self.data_size}"

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample at the given index.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Dictionary with data and labels for compatibility with Transformers
        """
        result = {}

        # Load image if needed
        if self.mode in ["image", "multimodal"]:
            image_path = os.path.join(self.image_dir, self.image_files[index])
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)
            result["image"] = image_tensor

        # Load spectral data if needed
        if self.mode in ["spectral", "multimodal"]:
            spectral_sample = self.spectral_data[index].copy()
            if self.scale_spectral and self.spectral_scaler is not None:
                spectral_sample = self.spectral_scaler.transform(
                    spectral_sample.reshape(1, -1)
                ).flatten()
            spectral_tensor = torch.tensor(spectral_sample, dtype=torch.float32)
            result["spectral"] = spectral_tensor

        # Add labels if available
        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)

        # For single-modal cases, also provide the data as 'x' for compatibility
        if self.mode == "image":
            result["x"] = result["image"]
        elif self.mode == "spectral":
            result["x"] = result["spectral"]

        return result

    def inverse_transform_spectral(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform spectral data if scaling was applied."""
        if not self.scale_spectral or self.spectral_scaler is None:
            return data
        return self.spectral_scaler.inverse_transform(data)

    def get_spectral_scaler(self) -> Optional[StandardScaler]:
        """Get the fitted spectral scaler object."""
        return self.spectral_scaler if self.scale_spectral else None


def create_datasets(
    image_dir: Optional[str] = None,
    spectral_csv_path: Optional[str] = None,
    spectral_columns: Optional[List[str]] = None,
    labels_csv_path: Optional[str] = None,
    label_column: str = "label",
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    scale_spectral: bool = True,
    mode: str = "multimodal",
    image_transform: Optional[transforms.Compose] = None,
) -> Dict[str, MetalloDataset]:
    """
    Create train, eval, and test datasets for metallographic and spectral data.

    Args:
        image_dir: Directory containing metallographic images
        spectral_csv_path: Path to CSV file with spectral data
        spectral_columns: List of spectral feature columns to use
        labels_csv_path: Path to CSV file with labels (can be same as spectral_csv_path)
        label_column: Name of the label column
        split_ratios: Train/eval/test split ratios
        scale_spectral: Whether to apply StandardScaler to spectral data
        mode: Dataset mode - "image", "spectral", or "multimodal"
        image_transform: Custom image transforms

    Returns:
        Dictionary with 'train', 'eval', 'test' datasets
    """
    # Load spectral data if provided
    spectral_data = None
    if spectral_csv_path is not None:
        if spectral_columns is None:
            raise ValueError(
                "spectral_columns must be provided when spectral_csv_path is given"
            )
        df_spectral = pd.read_csv(spectral_csv_path, usecols=spectral_columns)
        spectral_data = df_spectral.values.astype(np.float32)
        logging.info(
            f"Loaded spectral data: {spectral_data.shape[0]} samples, {spectral_data.shape[1]} features"
        )

    # Load labels if provided
    labels = None
    if labels_csv_path is not None:
        df_labels = pd.read_csv(labels_csv_path)
        labels = df_labels[label_column].values.astype(np.float32)
        logging.info(f"Loaded labels: {len(labels)} samples")

    # Determine total samples
    if mode == "image":
        if image_dir is None:
            raise ValueError("image_dir must be provided for image mode")
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]
        total_samples = len(image_files)
    elif mode == "spectral":
        if spectral_data is None:
            raise ValueError("spectral_csv_path must be provided for spectral mode")
        total_samples = len(spectral_data)
    else:  # multimodal
        if image_dir is None or spectral_data is None:
            raise ValueError(
                "Both image_dir and spectral_csv_path must be provided for multimodal mode"
            )
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]
        total_samples = min(len(image_files), len(spectral_data))

    # Create splits
    train_end = int(total_samples * split_ratios[0])
    eval_end = int(total_samples * (split_ratios[0] + split_ratios[1]))

    logging.info(f"Data splits:")
    logging.info(f"  Train: {train_end} samples")
    logging.info(f"  Eval:  {eval_end - train_end} samples")
    logging.info(f"  Test:  {total_samples - eval_end} samples")

    # Split data
    datasets = {}

    for split_name, (start_idx, end_idx) in [
        ("train", (0, train_end)),
        ("eval", (train_end, eval_end)),
        ("test", (eval_end, total_samples)),
    ]:
        split_spectral_data = None
        split_labels = None

        if spectral_data is not None:
            split_spectral_data = spectral_data[start_idx:end_idx]

        if labels is not None:
            split_labels = labels[start_idx:end_idx]

        # For training set, use its own data for scaler fitting
        # For eval/test sets, use training data for scaler fitting
        train_spectral_for_scaler = (
            spectral_data[:train_end] if spectral_data is not None else None
        )

        datasets[split_name] = MetalloDataset(
            image_dir=image_dir,
            spectral_data=split_spectral_data,
            labels=split_labels,
            image_transform=image_transform,
            scale_spectral=scale_spectral,
            train_spectral_data=(
                train_spectral_for_scaler if split_name != "train" else None
            ),
            mode=mode,
        )

    return datasets
