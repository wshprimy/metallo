#!/usr/bin/env python3
"""
Spectral Data Preprocessing Pipeline
====================================

This module processes CSV spectral data files from .dataset/libs/ and converts them
into structured numpy arrays for machine learning applications.

Key Features:
- Processes CSV files with spectral data (X,Y coordinates)
- Combines 768 CSV files into 768x1600 numpy matrices
- Maintains directory structure in output
- Robust error handling and progress tracking
- Memory-efficient processing

Author: Structured Data Architect
Environment: conda metallo
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocess.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SpectralDataProcessor:
    """
    Main class for processing spectral CSV data into structured numpy arrays.
    """

    def __init__(
        self,
        source_dir: str = ".dataset/libs",
        output_dir: str = ".dataset/metallography",
        target_files: int = 768,
        spectral_points: int = 1600,
    ):
        """
        Initialize the spectral data processor.

        Args:
            source_dir: Source directory containing CSV files
            output_dir: Output directory for processed numpy arrays
            target_files: Number of files to combine into each matrix (768)
            spectral_points: Number of spectral data points per file (1600)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_files = target_files
        self.spectral_points = spectral_points

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SpectralDataProcessor")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Target matrix size: {target_files}x{spectral_points}")

    def parse_csv_file(self, csv_path: Path) -> Optional[np.ndarray]:
        """
        Parse a single CSV file and extract spectral Y-values.

        Args:
            csv_path: Path to the CSV file

        Returns:
            numpy array of Y-values (1600 points) or None if parsing fails
        """
        try:
            # Read CSV file, skipping the first 2 header lines
            df = pd.read_csv(csv_path, skiprows=2, header=None, names=["X", "Y"])

            # Validate data shape
            if len(df) != self.spectral_points:
                logger.warning(
                    f"File {csv_path.name} has {len(df)} points, expected {self.spectral_points}"
                )
                return None

            # Extract Y-values (spectral intensity data)
            y_values = df["Y"].values.astype(np.float32)

            # Validate data integrity
            if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
                logger.warning(
                    f"File {csv_path.name} contains invalid values (NaN/Inf)"
                )
                return None

            return y_values

        except Exception as e:
            logger.error(f"Error parsing {csv_path.name}: {str(e)}")
            return None

    def discover_and_sort_csv_files(self, directory: Path) -> List[Path]:
        """
        Discover CSV files in directory and sort them numerically by ID.

        Args:
            directory: Directory to search for CSV files

        Returns:
            List of CSV file paths sorted by numerical ID
        """
        csv_files = []
        pattern = re.compile(r"Curve\((\d+)\)\.csv")

        for file_path in directory.glob("*.csv"):
            match = pattern.match(file_path.name)
            if match:
                file_id = int(match.group(1))
                csv_files.append((file_id, file_path))

        # Sort by numerical ID (not alphabetically)
        csv_files.sort(key=lambda x: x[0])

        # Return only the file paths
        sorted_paths = [path for _, path in csv_files]

        logger.info(f"Found {len(sorted_paths)} CSV files in {directory}")
        return sorted_paths

    def create_spectrum_matrix(self, csv_files: List[Path]) -> Optional[np.ndarray]:
        """
        Create a 768x1600 spectrum matrix from CSV files.

        Args:
            csv_files: List of CSV file paths (should be at least 768)

        Returns:
            numpy array of shape (768, 1600) or None if creation fails
        """
        if len(csv_files) < self.target_files:
            logger.warning(
                f"Only {len(csv_files)} files available, need {self.target_files}"
            )
            return None

        # Initialize matrix
        spectrum_matrix = np.zeros(
            (self.target_files, self.spectral_points), dtype=np.float32
        )
        successful_files = 0

        # Process first 768 files
        for i, csv_file in enumerate(csv_files[: self.target_files]):
            spectral_data = self.parse_csv_file(csv_file)

            if spectral_data is not None:
                spectrum_matrix[i] = spectral_data
                successful_files += 1
            else:
                logger.warning(f"Failed to process file {csv_file.name}, using zeros")

        logger.info(
            f"Successfully processed {successful_files}/{self.target_files} files"
        )

        if successful_files < self.target_files:
            logger.error(
                f"Too many failed files ({self.target_files - successful_files}), aborting matrix creation"
            )
            return None

        return spectrum_matrix

    def save_spectrum_matrix(self, matrix: np.ndarray, output_path: Path) -> bool:
        """
        Save spectrum matrix as numpy array.

        Args:
            matrix: Spectrum matrix to save
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as numpy array
            np.save(output_path, matrix)

            # Verify saved file
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(
                    f"Saved spectrum matrix: {output_path} ({file_size:.2f} MB)"
                )
                return True
            else:
                logger.error(f"Failed to save matrix to {output_path}")
                return False

        except Exception as e:
            logger.error(f"Error saving matrix to {output_path}: {str(e)}")
            return False

    def process_subdirectory(self, subdir_path: Path) -> bool:
        """
        Process a single subdirectory (e.g., 1/1d, 1/2d, etc.).

        Args:
            subdir_path: Path to subdirectory containing CSV files

        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing subdirectory: {subdir_path}")

        # Discover and sort CSV files
        csv_files = self.discover_and_sort_csv_files(subdir_path)

        if not csv_files:
            logger.warning(f"No CSV files found in {subdir_path}")
            return False

        # Create spectrum matrix
        spectrum_matrix = self.create_spectrum_matrix(csv_files)

        if spectrum_matrix is None:
            logger.error(f"Failed to create spectrum matrix for {subdir_path}")
            return False

        # Determine output path
        relative_path = subdir_path.relative_to(self.source_dir)
        output_subdir = self.output_dir / relative_path
        output_file = output_subdir / "spectrum.npy"

        # Save spectrum matrix
        success = self.save_spectrum_matrix(spectrum_matrix, output_file)

        if success:
            logger.info(f"Successfully processed {subdir_path} -> {output_file}")

        return success

    def validate_output_matrix(self, matrix_path: Path) -> Dict[str, any]:
        """
        Validate a saved spectrum matrix.

        Args:
            matrix_path: Path to saved numpy matrix

        Returns:
            Dictionary with validation results
        """
        try:
            matrix = np.load(matrix_path)

            validation_results = {
                "file_exists": True,
                "shape": matrix.shape,
                "dtype": matrix.dtype,
                "size_mb": matrix_path.stat().st_size / (1024 * 1024),
                "has_nan": np.any(np.isnan(matrix)),
                "has_inf": np.any(np.isinf(matrix)),
                "min_value": np.min(matrix),
                "max_value": np.max(matrix),
                "mean_value": np.mean(matrix),
                "std_value": np.std(matrix),
            }

            # Check if shape is correct
            expected_shape = (self.target_files, self.spectral_points)
            validation_results["correct_shape"] = matrix.shape == expected_shape

            return validation_results

        except Exception as e:
            return {"file_exists": False, "error": str(e)}

    def run_full_processing(self) -> Dict[str, bool]:
        """
        Run the complete processing pipeline on all subdirectories.

        Returns:
            Dictionary mapping subdirectory paths to success status
        """
        logger.info("Starting full processing pipeline")

        results = {}
        total_subdirs = 0
        successful_subdirs = 0

        # Process all subdirectories
        for lib_dir in self.source_dir.iterdir():
            if lib_dir.is_dir():
                for time_dir in lib_dir.iterdir():
                    if time_dir.is_dir():
                        total_subdirs += 1
                        subdir_key = str(time_dir.relative_to(self.source_dir))

                        success = self.process_subdirectory(time_dir)
                        results[subdir_key] = success

                        if success:
                            successful_subdirs += 1

        # Summary
        logger.info(
            f"Processing complete: {successful_subdirs}/{total_subdirs} subdirectories successful"
        )

        return results

    def validate_all_outputs(self) -> Dict[str, Dict]:
        """
        Validate all output spectrum matrices.

        Returns:
            Dictionary with validation results for each output file
        """
        logger.info("Validating all output matrices")

        validation_results = {}

        for spectrum_file in self.output_dir.rglob("spectrum.npy"):
            relative_path = str(spectrum_file.relative_to(self.output_dir))
            validation_results[relative_path] = self.validate_output_matrix(
                spectrum_file
            )

        return validation_results


def main():
    """
    Main function to run the spectral data preprocessing pipeline.
    """
    logger.info("=" * 60)
    logger.info("SPECTRAL DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Initialize processor
    processor = SpectralDataProcessor()

    # Run full processing
    processing_results = processor.run_full_processing()

    # Validate outputs
    validation_results = processor.validate_all_outputs()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)

    successful_processing = sum(1 for success in processing_results.values() if success)
    total_processing = len(processing_results)

    logger.info(
        f"Processing Results: {successful_processing}/{total_processing} successful"
    )

    for subdir, success in processing_results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {subdir}")

    logger.info("\nValidation Results:")
    for output_file, validation in validation_results.items():
        if validation.get("file_exists", False):
            shape_ok = "✓" if validation.get("correct_shape", False) else "✗"
            logger.info(
                f"  {shape_ok} {output_file}: {validation['shape']} ({validation['size_mb']:.2f} MB)"
            )
        else:
            logger.info(
                f"  ✗ {output_file}: {validation.get('error', 'File not found')}"
            )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
