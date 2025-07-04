import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging


class Spectral(Dataset):
    """
    Dataset for spectral data (LIBS).
    Each DOS folder contains a CSV file with spectral measurements.
    """

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        is_val: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        spectral_length: int = 100,
        normalize: bool = True,
    ):
        """
        Initialize spectral dataset.

        Args:
            data_dir: Directory containing DOS folders with CSV files
            is_train: Whether this is training set
            is_val: Whether this is validation set
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            spectral_length: Length of each spectral segment (default 100)
            normalize: Whether to normalize spectral data
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.is_val = is_val
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.spectral_length = spectral_length
        self.normalize = normalize

        # Get DOS folders and sort them
        self.dos_folders = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, f))
            ]
        )

        # Extract DOS values and load spectral data
        self.dos_values = []
        self.spectral_data = []

        for folder in self.dos_folders:
            try:
                dos_value = float(folder)

                # Find CSV file in the folder
                dos_path = os.path.join(data_dir, folder)
                csv_files = [f for f in os.listdir(dos_path) if f.endswith(".csv")]

                if not csv_files:
                    logging.warning(f"No CSV file found in {folder}")
                    continue

                # Use the first CSV file (should be named like {dos_value}.csv)
                csv_path = os.path.join(dos_path, csv_files[0])

                # Load spectral data
                df = pd.read_csv(csv_path, header=None)
                # Transpose the data since each column is a spectrum with 1600 dimensions
                spectral_matrix = df.values.T.astype(np.float32)  # Shape: (num_spectra, 1600)
                spectral_array = spectral_matrix.flatten()  # Flatten for compatibility

                self.dos_values.append(dos_value)
                self.spectral_data.append(spectral_array)

            except (ValueError, Exception) as e:
                logging.warning(f"Skipping folder {folder}: {e}")
                continue

        # Calculate how many spectral segments we can extract from each CSV
        if self.spectral_data:
            total_length = len(self.spectral_data[0])
            self.segments_per_dos = total_length // self.spectral_length

            # Calculate split indices for segments
            if self.is_train:
                self.start_segment = 0
                self.end_segment = int(self.segments_per_dos * self.train_ratio)
            elif self.is_val:
                self.start_segment = int(self.segments_per_dos * self.train_ratio)
                self.end_segment = int(
                    self.segments_per_dos * (self.train_ratio + self.val_ratio)
                )
            else:  # test
                self.start_segment = int(
                    self.segments_per_dos * (self.train_ratio + self.val_ratio)
                )
                self.end_segment = self.segments_per_dos

            self.segments_per_split = self.end_segment - self.start_segment
        else:
            self.segments_per_dos = 0
            self.segments_per_split = 0

        logging.info(
            f"Spectral: {len(self.dos_values)} DOS values, "
            f"{self.segments_per_split} segments per DOS, "
            f"spectral length: {self.spectral_length}, "
            f"total samples: {len(self)}"
        )

    def __len__(self) -> int:
        return len(self.dos_values) * self.segments_per_split

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        """
        Get spectral segment and corresponding DOS value.

        Args:
            index: Sample index

        Returns:
            Tuple of (spectral_tensor, dos_value)
        """
        # Calculate which DOS and which segment within that DOS
        dos_idx = index // self.segments_per_split
        segment_idx = index % self.segments_per_split + self.start_segment

        dos_value = self.dos_values[dos_idx]
        spectral_array = self.spectral_data[dos_idx]

        # Extract the specific segment
        start_idx = segment_idx * self.spectral_length
        end_idx = start_idx + self.spectral_length

        spectral_segment = spectral_array[start_idx:end_idx]

        # Normalize if requested
        if self.normalize:
            max_val = np.max(spectral_segment)
            if max_val > 0:
                spectral_segment = spectral_segment / max_val

        spectral_tensor = torch.tensor(spectral_segment, dtype=torch.float32)

        return spectral_tensor, dos_value

    def get_full_spectrum(self, dos_idx: int) -> Tuple[np.ndarray, float]:
        """
        Get the full spectrum for a given DOS index.

        Args:
            dos_idx: Index of the DOS value

        Returns:
            Tuple of (full_spectrum, dos_value)
        """
        if dos_idx >= len(self.dos_values):
            raise IndexError(f"DOS index {dos_idx} out of range")

        return self.spectral_data[dos_idx], self.dos_values[dos_idx]
