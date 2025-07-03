import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple
import logging

from .metallographic_dataset import MetallographicDataset
from .spectral_dataset import SpectralDataset


class UnifiedMetalloDataset(Dataset):
    """
    Unified dataset that combines metallographic images and spectral data.
    Supports image-only, spectral-only, and multimodal modes.
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = "multimodal",  # "image", "spectral", "multimodal"
        is_train: bool = True,
        is_val: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        images_per_dos: int = 100,
        spectral_length: int = 100,
        image_transform: Optional[Any] = None,
        preprocess_images: bool = False,
        normalize_spectral: bool = True
    ):
        """
        Initialize unified dataset.
        
        Args:
            data_dir: Directory containing DOS folders
            mode: Dataset mode - "image", "spectral", or "multimodal"
            is_train: Whether this is training set
            is_val: Whether this is validation set
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            images_per_dos: Number of images per DOS value
            spectral_length: Length of each spectral segment
            image_transform: Image transformations
            preprocess_images: Whether to preprocess images
            normalize_spectral: Whether to normalize spectral data
        """
        self.data_dir = data_dir
        self.mode = mode
        self.is_train = is_train
        self.is_val = is_val
        
        # Validate mode
        if mode not in ["image", "spectral", "multimodal"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'image', 'spectral', or 'multimodal'")
        
        # Initialize sub-datasets based on mode
        self.image_dataset = None
        self.spectral_dataset = None
        
        if mode in ["image", "multimodal"]:
            self.image_dataset = MetallographicDataset(
                data_dir=data_dir,
                image_transform=image_transform,
                is_train=is_train,
                is_val=is_val,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                images_per_dos=images_per_dos,
                preprocess=preprocess_images
            )
        
        if mode in ["spectral", "multimodal"]:
            self.spectral_dataset = SpectralDataset(
                data_dir=data_dir,
                is_train=is_train,
                is_val=is_val,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                spectral_length=spectral_length,
                normalize=normalize_spectral
            )
        
        # Determine dataset size
        if mode == "image":
            self.dataset_size = len(self.image_dataset)
        elif mode == "spectral":
            self.dataset_size = len(self.spectral_dataset)
        else:  # multimodal
            # For multimodal, we need to ensure alignment between image and spectral data
            # We'll use the minimum size to ensure we have both modalities for each sample
            img_size = len(self.image_dataset)
            spec_size = len(self.spectral_dataset)
            
            # Calculate samples per DOS for each modality
            img_samples_per_dos = img_size // len(self.image_dataset.dos_values)
            spec_samples_per_dos = spec_size // len(self.spectral_dataset.dos_values)
            
            # Use minimum samples per DOS to ensure alignment
            min_samples_per_dos = min(img_samples_per_dos, spec_samples_per_dos)
            num_dos_values = min(len(self.image_dataset.dos_values), 
                               len(self.spectral_dataset.dos_values))
            
            self.dataset_size = num_dos_values * min_samples_per_dos
            self.samples_per_dos = min_samples_per_dos
            
            logging.info(f"Multimodal dataset: {num_dos_values} DOS values, "
                        f"{min_samples_per_dos} samples per DOS, "
                        f"total: {self.dataset_size}")
        
        logging.info(f"UnifiedMetalloDataset initialized: mode={mode}, size={self.dataset_size}")
    
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
        
        if self.mode == "image":
            image, dos_value = self.image_dataset[index]
            result["image"] = image
            result["labels"] = torch.tensor(dos_value, dtype=torch.float32)
            result["x"] = image  # For backward compatibility
            
        elif self.mode == "spectral":
            spectral, dos_value = self.spectral_dataset[index]
            result["spectral"] = spectral
            result["labels"] = torch.tensor(dos_value, dtype=torch.float32)
            result["x"] = spectral  # For backward compatibility
            
        else:  # multimodal
            # For multimodal, we need to ensure we get corresponding samples
            dos_idx = index // self.samples_per_dos
            sample_idx_within_dos = index % self.samples_per_dos
            
            # Calculate corresponding indices for each modality
            img_idx = dos_idx * self.image_dataset.images_per_split + sample_idx_within_dos
            spec_idx = dos_idx * self.spectral_dataset.segments_per_split + sample_idx_within_dos
            
            # Ensure indices are within bounds
            img_idx = min(img_idx, len(self.image_dataset) - 1)
            spec_idx = min(spec_idx, len(self.spectral_dataset) - 1)
            
            image, dos_value_img = self.image_dataset[img_idx]
            spectral, dos_value_spec = self.spectral_dataset[spec_idx]
            
            # Use the DOS value from images (they should be the same)
            result["image"] = image
            result["spectral"] = spectral
            result["labels"] = torch.tensor(dos_value_img, dtype=torch.float32)
        
        return result
    
    def get_dos_values(self) -> list:
        """Get list of DOS values in the dataset."""
        if self.mode == "image":
            return self.image_dataset.dos_values
        elif self.mode == "spectral":
            return self.spectral_dataset.dos_values
        else:  # multimodal
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
            "num_dos_values": len(self.get_dos_values())
        }
        
        if self.image_dataset:
            info["images_per_dos"] = self.image_dataset.images_per_split
            info["image_transform"] = str(self.image_dataset.image_transform)
        
        if self.spectral_dataset:
            info["spectral_segments_per_dos"] = self.spectral_dataset.segments_per_split
            info["spectral_length"] = self.spectral_dataset.spectral_length
        
        return info