import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple
import logging

from .metallographic import Metallographic
from .spectral import Spectral


class UnimetalloDataset(Dataset):
    """
    Unified dataset that combines metallographic images and spectral data.
    Supports metallographic-only, spectral-only, and unified modes.
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "unified",  # "metallographic", "spectral", "unified"
        is_train: bool = True,
        is_val: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        images_per_dos: int = 30,
        spectra_per_dos: int = 2,
        image_transform: Optional[Any] = None,
        process_images: bool = False,
        normalize_spectral: bool = True,
    ):
        """
        Initialize unified dataset with mode support.

        Args:
            data_dir: Directory containing DOS folders
            mode: Dataset mode - "metallographic", "spectral", or "unified"
            is_train: Whether this is training set
            is_val: Whether this is validation set
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            images_per_dos: Number of images per DOS value (default 30)
            spectra_per_dos: Number of spectra per DOS value (default 2)
            image_transform: Image transformations
            process_images: Whether to preprocess images
            normalize_spectral: Whether to normalize spectral data
        """
        self.data_dir = data_dir
        self.mode = mode
        self.is_train = is_train
        self.is_val = is_val
        self.images_per_dos = images_per_dos
        self.spectra_per_dos = spectra_per_dos

        # Validate mode
        if mode not in ["metallographic", "spectral", "unified"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'metallographic', 'spectral', or 'unified'"
            )

        # Initialize sub-datasets based on mode
        self.image_dataset = None
        self.spectral_dataset = None

        if mode in ["metallographic", "unified"]:
            self.image_dataset = Metallographic(
                data_dir=data_dir,
                image_transform=image_transform,
                is_train=is_train,
                is_val=is_val,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                images_per_dos=images_per_dos,
                process_images=process_images,
            )

        if mode in ["spectral", "unified"]:
            self.spectral_dataset = Spectral(
                data_dir=data_dir,
                is_train=is_train,
                is_val=is_val,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                spectral_length=1600,  # Full spectrum length
                normalize=normalize_spectral,
            )

        # Calculate dataset size based on mode
        if mode == "metallographic":
            self.dataset_size = len(self.image_dataset)
        elif mode == "spectral":
            self.dataset_size = len(self.spectral_dataset)
        else:  # unified
            # Calculate dataset size based on unique image-spectrum pairing
            # Each image gets paired with exactly one spectrum
            num_dos_values = min(
                len(self.image_dataset.dos_values),
                len(self.spectral_dataset.dos_values),
            )
            
            # Calculate how many unique pairs we can create per DOS
            # Each spectrum can be paired with multiple images
            images_per_spectrum = self.images_per_dos // self.spectra_per_dos
            pairs_per_dos = self.images_per_dos  # Each image gets one spectrum
            
            self.dataset_size = num_dos_values * pairs_per_dos
            self.pairs_per_dos = pairs_per_dos
            self.images_per_spectrum = images_per_spectrum

        logging.info(
            f"UnimetalloDataset: mode={mode}, size={self.dataset_size}"
        )

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get sample(s) based on the dataset mode.

        Args:
            index: Sample index

        Returns:
            Dictionary containing the requested data modalities and labels
        """
        result = {}

        if self.mode == "metallographic":
            image, dos_value = self.image_dataset[index]
            result["image"] = image
            result["labels"] = torch.tensor(dos_value, dtype=torch.float32)
            result["x"] = image  # For backward compatibility

        elif self.mode == "spectral":
            spectral, dos_value = self.spectral_dataset[index]
            result["spectral"] = spectral
            result["labels"] = torch.tensor(dos_value, dtype=torch.float32)
            result["x"] = spectral  # For backward compatibility

        else:  # unified
            # Calculate which DOS folder and which pair within that DOS
            dos_idx = index // self.pairs_per_dos
            pair_idx_within_dos = index % self.pairs_per_dos

            # Get the image directly
            img_idx = dos_idx * self.image_dataset.images_per_split + pair_idx_within_dos
            img_idx = min(img_idx, len(self.image_dataset) - 1)
            image, dos_value = self.image_dataset[img_idx]

            # Map image to its corresponding spectrum
            # Each spectrum is shared by multiple images
            spectrum_idx_within_dos = pair_idx_within_dos // self.images_per_spectrum
            spectrum_idx_within_dos = min(spectrum_idx_within_dos, self.spectra_per_dos - 1)
            
            spec_idx = dos_idx * self.spectral_dataset.segments_per_split + spectrum_idx_within_dos
            spec_idx = min(spec_idx, len(self.spectral_dataset) - 1)
            spectral, _ = self.spectral_dataset[spec_idx]

            result["image"] = image
            result["spectral"] = spectral
            result["labels"] = torch.tensor(dos_value, dtype=torch.float32)

        return result

    def get_dos_values(self) -> list:
        """Get list of DOS values in the dataset."""
        if self.mode == "metallographic":
            return self.image_dataset.dos_values
        elif self.mode == "spectral":
            return self.spectral_dataset.dos_values
        else:  # unified
            # Return the intersection of DOS values from both modalities
            img_dos = set(self.image_dataset.dos_values)
            spec_dos = set(self.spectral_dataset.dos_values)
            return sorted(list(img_dos.intersection(spec_dos)))

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            "mode": self.mode,
            "size": self.dataset_size,
            "dos_values": self.get_dos_values(),
            "num_dos_values": len(self.get_dos_values()),
        }

        if self.mode in ["metallographic", "unified"]:
            info["images_per_dos"] = self.images_per_dos
            
        if self.mode in ["spectral", "unified"]:
            info["spectra_per_dos"] = self.spectra_per_dos
            
        if self.mode == "unified":
            info["images_per_spectrum"] = self.images_per_spectrum
            info["pairs_per_dos"] = self.pairs_per_dos

        if self.image_dataset:
            info["images_per_split"] = self.image_dataset.images_per_split
            info["image_transform"] = str(self.image_dataset.image_transform)

        if self.spectral_dataset:
            info["spectral_segments_per_split"] = self.spectral_dataset.segments_per_split
            info["spectral_length"] = self.spectral_dataset.spectral_length

        return info
