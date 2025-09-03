#!/usr/bin/env python3
"""
Test script for the new unified MetalloDS dataset.
Run this in the metallo conda environment to test the implementation.
"""

import torch
from torch.utils.data import DataLoader
from metallo.data import MetalloDS


def test_dataset():
    """Test the new MetalloDS implementation."""
    print("=" * 60)
    print("Testing MetalloDS Implementation")
    print("=" * 60)

    # Test different modes
    modes = ["image", "spectral", "unified"]

    for mode in modes:
        print(f"\n--- Testing mode: {mode} ---")

        try:
            # Create dataset
            dataset = MetalloDS(
                data_dir=".dataset/metallography",
                mode=mode,
                is_train=True,
                train_ratio=0.8,
                val_ratio=0.1,
            )

            print(f"✓ Dataset created successfully")
            print(f"✓ Total samples: {len(dataset)}")

            # Test loading a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✓ Sample loaded successfully")
                print(f"✓ Sample keys: {list(sample.keys())}")

                if "image" in sample:
                    print(f"✓ Image shape: {sample['image'].shape}")
                if "spectral" in sample:
                    print(f"✓ Spectral shape: {sample['spectral'].shape}")

                print(f"✓ Sample ID: {sample['sample_id']}")

        except Exception as e:
            print(f"✗ Error in mode {mode}: {e}")
            import traceback

            traceback.print_exc()

    # Test train/val/test splits
    print(f"\n--- Testing data splits ---")
    try:
        train_ds = MetalloDS(
            ".dataset/metallography", mode="unified", is_train=True, is_val=False
        )
        val_ds = MetalloDS(
            ".dataset/metallography", mode="unified", is_train=False, is_val=True
        )
        test_ds = MetalloDS(
            ".dataset/metallography", mode="unified", is_train=False, is_val=False
        )

        print(f"✓ Train samples: {len(train_ds)}")
        print(f"✓ Val samples: {len(val_ds)}")
        print(f"✓ Test samples: {len(test_ds)}")
        print(f"✓ Total: {len(train_ds) + len(val_ds) + len(test_ds)}")

        # Check ratios
        total = len(train_ds) + len(val_ds) + len(test_ds)
        train_ratio = len(train_ds) / total
        val_ratio = len(val_ds) / total
        test_ratio = len(test_ds) / total

        print(
            f"✓ Actual ratios - Train: {train_ratio:.3f}, Val: {val_ratio:.3f}, Test: {test_ratio:.3f}"
        )

    except Exception as e:
        print(f"✗ Error in split testing: {e}")
        import traceback

        traceback.print_exc()

    # Test DataLoader
    print(f"\n--- Testing DataLoader ---")
    try:
        dataset = MetalloDS(".dataset/metallography", mode="unified", is_train=True)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        batch = next(iter(dataloader))
        print(f"✓ DataLoader created successfully")
        print(f"✓ Batch keys: {list(batch.keys())}")

        if "image" in batch:
            print(f"✓ Batch image shape: {batch['image'].shape}")
        if "spectral" in batch:
            print(f"✓ Batch spectral shape: {batch['spectral'].shape}")

    except Exception as e:
        print(f"✗ Error in DataLoader testing: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()
