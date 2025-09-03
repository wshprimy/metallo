import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import logging
import csv


class MetalloDS(Dataset):
    """
    Unified dataset for metallography images and spectral data.

    Dataset structure:
    - 2 physical slices × 8 time points × 32 images = 512 total samples
    - Each folder contains: 32 PNG images + 1 spectrum.npy file
    - Each image corresponds to 24 spectra (768 spectra ÷ 32 images = 24)
    - Sample format: 1 image + 24 corresponding spectra
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "unified",  # "image", "spectral", "unified"
        split: str = "train",  # "train", "val", "test"
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        image_transform: Optional[transforms.Compose] = None,
        process_images: bool = False,
        normalize_spectral: bool = True,
        spectra_per_image: int = 24,
    ):
        """
        Initialize unified metallography dataset.

        Args:
            data_dir: Root directory containing slice folders (e.g., '.dataset/metallography')
            mode: Dataset mode - "image", "spectral", or "unified"
            split: Dataset split - "train", "val", or "test"
            train_ratio: Ratio for training split (default: 0.8)
            val_ratio: Ratio for validation split (default: 0.1)
            image_transform: Image transformations
            process_images: Whether to apply image preprocessing
            normalize_spectral: Whether to normalize spectral data
            spectra_per_image: Number of spectra per image (default: 24)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.split = split
        assert split in ["train", "eval", "test"], "Invalid split"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.process_images = process_images
        self.normalize_spectral = normalize_spectral
        self.spectra_per_image = spectra_per_image
        assert (
            self.spectra_per_image == 24
        ), "Currently only supports 24 spectra per image"

        # Validate mode
        if mode not in ["image", "spectral", "unified"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'image', 'spectral', or 'unified'"
            )

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

        # Discover all sample paths
        self.samples = self._discover_samples()

        # Calculate split indices
        total_samples = len(self.samples)
        if self.split == "train":
            self.start_idx = 0
            self.end_idx = int(total_samples * self.train_ratio)
        elif self.split == "eval":
            self.start_idx = int(total_samples * self.train_ratio)
            self.end_idx = int(total_samples * (self.train_ratio + self.val_ratio))
        elif self.split == "test":
            self.start_idx = int(total_samples * (self.train_ratio + self.val_ratio))
            self.end_idx = total_samples
        self.samples = self.samples[self.start_idx : self.end_idx]

    def _split_spectra_to_images(
        self, full_spectrum: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Split 768 spectra into 32 images (24 spectra per image).

        Args:
            full_spectrum: Full spectrum array of shape (768, 1600)

        Returns:
            Dictionary mapping image_number (1-32) to corresponding spectra (24, 1600)
        """
        image_spectra_dict = {}
        for image_num in range(0, 32):
            start_idx = image_num // 8 * 192 + image_num % 8 * 4
            image_spectra_idx = []
            image_spectra = []  # Shape: (24, 1600)

            for i in range(0, 6):
                for j in range(0, 4):
                    spectrum_idx = start_idx + i * 32 + j
                    image_spectra_idx.append(spectrum_idx)
                    image_spectra.append(full_spectrum[spectrum_idx, :])

            # Normalize if requested
            if self.normalize_spectral:
                # Normalize each spectrum individually
                max_vals = np.max(image_spectra, axis=1, keepdims=True)
                max_vals = np.where(
                    max_vals > 0, max_vals, 1.0
                )  # Avoid division by zero
                image_spectra = image_spectra / max_vals

            image_spectra_dict[image_num + 1] = image_spectra

        # print(image_spectra_dict)
        return image_spectra_dict

    def _discover_samples(self) -> List[Dict[str, Any]]:
        """
        Discover all samples in the dataset and load spectra during initialization.

        Returns:
            List of sample dictionaries containing paths, metadata, and pre-loaded spectra
        """
        samples = []

        # Expected structure: data_dir/slice/time_point/
        slice_dirs = sorted(
            [
                d
                for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
            ]
        )

        for slice_id in slice_dirs:
            slice_path = os.path.join(self.data_dir, slice_id)
            dos_csv_path = os.path.join(slice_path, "dos-dictionary.csv")

            dos_dict = {}
            with open(dos_csv_path, "r", encoding="utf-8-sig") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        time_point = row[0].strip()
                        dos_value = float(row[1].strip())
                        dos_dict[time_point] = dos_value
            logging.info(
                f"Loaded DOS dictionary for slice {slice_id}: {len(dos_dict)} time points"
            )

            # Get time point directories
            time_dirs = sorted(
                [
                    d
                    for d in os.listdir(slice_path)
                    if os.path.isdir(os.path.join(slice_path, d))
                ]
            )

            for time_point in time_dirs:
                time_path = os.path.join(slice_path, time_point)

                # Check if spectrum.npy exists
                spectrum_path = os.path.join(time_path, "spectrum.npy")
                if not os.path.exists(spectrum_path):
                    logging.warning(f"No spectrum.npy found in {time_path}")
                    continue

                # Load and split spectra for this time point
                logging.info(f"Loading spectra from {spectrum_path}")
                full_spectrum = np.load(spectrum_path)  # Shape: (768, 1600)
                image_spectra_dict = self._split_spectra_to_images(full_spectrum)

                # Get all PNG images
                image_files = sorted(
                    [f for f in os.listdir(time_path) if f.lower().endswith(".png")],
                    key=lambda x: int(os.path.splitext(x)[0]),
                )

                # Create samples for each image with pre-loaded spectra
                for img_idx, img_file in enumerate(image_files):
                    img_path = os.path.join(time_path, img_file)

                    # Extract image number from filename (e.g., "1.png" -> 1)
                    try:
                        img_num = int(os.path.splitext(img_file)[0])
                    except ValueError:
                        logging.warning(f"Cannot parse image number from {img_file}")
                        continue

                    # Get corresponding spectra for this image
                    if img_num in image_spectra_dict:
                        image_spectra = image_spectra_dict[img_num]  # Shape: (24, 1600)
                    else:
                        logging.warning(
                            f"No spectra found for image {img_num} in {time_path}"
                        )
                        continue

                    assert (
                        dos_dict.get(time_point, None) is not None
                    ), f"DOS value missing for time point {time_point} in slice {slice_id}"
                    sample = {
                        "slice_id": slice_id,
                        "time_point": time_point,
                        "image_number": img_num,
                        "image_path": img_path,
                        "spectra": torch.tensor(
                            image_spectra, dtype=torch.float32
                        ),  # Pre-loaded spectra (24, 1600)
                        "dos": torch.tensor(dos_dict[time_point], dtype=torch.float32),
                    }
                    samples.append(sample)

        logging.info(
            f"Discovered {len(samples)} samples across {len(slice_dirs)} slices with pre-loaded spectra"
        )
        return samples

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        if self.process_images:
            image = self._process_image(image_path)
            image = Image.fromarray(image)
        else:
            image = Image.open(image_path).convert("RGB")

        return self.image_transform(image)

    def _process_image(self, image_path: str) -> np.ndarray:
        """
        Process metallographic images with preprocessing.

        Args:
            image_path: Path to the image file

        Returns:
            Processed image as numpy array (uint8)
        """
        import cv2

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = img.copy()
        mask = cv2.medianBlur(mask, 3)

        mask = np.array([135.0, 135.0, 135.0], dtype=np.float32) - mask.astype(
            np.float32
        )
        mask = np.clip(mask, 0, 255).astype(np.uint8)

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get sample(s) based on the dataset mode.

        Args:
            index: Sample index

        Returns:
            Dictionary containing the requested data modalities
        """
        sample = self.samples[index]
        result = {
            # "slice_id": sample['slice_id'],
            # "time_point": sample['time_point'],
            # "image_number": sample['image_number'],
            "sample_id": f"{sample['slice_id']}_{sample['time_point']}_{sample['image_number']}",
        }

        if self.mode in ["image", "unified"]:
            result["image"] = self._load_image(sample["image_path"])
        if self.mode in ["spectral", "unified"]:
            result["spectral"] = sample["spectra"]
        result["labels"] = sample["dos"]
        return result
