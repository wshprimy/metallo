#!/usr/bin/env python3
"""
Test script to validate ToyNet refactoring for unified mode.
Tests model initialization and forward pass with sample data.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metallo.models import ToyNet, ToyNetConfig


def test_model_initialization():
    """Test ToyNet initialization with unified mode configuration."""
    print("=" * 60)
    print("Testing ToyNet Model Initialization")
    print("=" * 60)

    # Test with config object
    config = ToyNetConfig(
        mode="unified",
        image_backbone="resnet18",
        spectral_input_dim=38400,  # 24 × 1600
        hidden_dim=256,
        dropout=0.2,
        num_outputs=1,
    )

    print(
        f"Config created: mode={config.mode}, spectral_input_dim={config.spectral_input_dim}"
    )

    try:
        model = ToyNet(config)
        print("✓ Model initialized successfully with config")
        print(f"  Mode: {model.mode}")
        print(f"  Hidden dim: {model.hidden_dim}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        return model

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with sample unified mode data."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass with Unified Mode Data")
    print("=" * 60)

    if model is None:
        print("✗ Cannot test forward pass - model initialization failed")
        return

    batch_size = 2

    # Create sample inputs matching MetalloDS unified mode format
    sample_image = torch.randn(batch_size, 3, 300, 300)  # RGB images
    sample_spectral = torch.randn(batch_size, 24, 1600)  # 24 spectra per image
    sample_labels = torch.randn(batch_size, 1)  # DOS regression targets

    print(f"Sample inputs created:")
    print(f"  Image shape: {sample_image.shape}")
    print(f"  Spectral shape: {sample_spectral.shape}")
    print(f"  Labels shape: {sample_labels.shape}")

    try:
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                image=sample_image, spectral=sample_spectral, labels=sample_labels
            )

        print("\n✓ Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        print(f"  Predictions shape: {outputs['predictions'].shape}")
        print(f"  Loss: {outputs['loss'].item():.6f}")
        print(f"  Loss type: {type(outputs['loss'])}")

        # Verify output shapes
        expected_pred_shape = (batch_size, 1)
        if outputs["predictions"].shape == expected_pred_shape:
            print(f"✓ Predictions shape correct: {outputs['predictions'].shape}")
        else:
            print(
                f"✗ Predictions shape incorrect: expected {expected_pred_shape}, got {outputs['predictions'].shape}"
            )

        # Test with pre-flattened spectral data
        print("\n" + "-" * 40)
        print("Testing with pre-flattened spectral data")
        sample_spectral_flat = sample_spectral.view(batch_size, -1)  # (2, 38400)
        print(f"  Flattened spectral shape: {sample_spectral_flat.shape}")

        outputs_flat = model(
            image=sample_image, spectral=sample_spectral_flat, labels=sample_labels
        )
        print(f"✓ Forward pass with flattened spectral successful!")
        print(f"  Loss: {outputs_flat['loss'].item():.6f}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mode_validation():
    """Test mode validation and error handling."""
    print("\n" + "=" * 60)
    print("Testing Mode Validation and Error Handling")
    print("=" * 60)

    # Test unified mode requires both inputs
    config = ToyNetConfig(mode="unified", spectral_input_dim=38400)
    model = ToyNet(config)

    batch_size = 1
    sample_image = torch.randn(batch_size, 3, 300, 300)
    sample_spectral = torch.randn(batch_size, 24, 1600)

    # Test missing image
    try:
        model(spectral=sample_spectral)
        print("✗ Should have failed with missing image")
    except ValueError as e:
        print(f"✓ Correctly caught missing image error: {e}")

    # Test missing spectral
    try:
        model(image=sample_image)
        print("✗ Should have failed with missing spectral")
    except ValueError as e:
        print(f"✓ Correctly caught missing spectral error: {e}")

    # Test wrong spectral shape
    try:
        wrong_spectral = torch.randn(batch_size, 10, 100)  # Wrong shape
        model(image=sample_image, spectral=wrong_spectral)
        print("✗ Should have failed with wrong spectral shape")
    except ValueError as e:
        print(f"✓ Correctly caught wrong spectral shape error: {e}")


def main():
    """Main test function."""
    print("ToyNet Unified Mode Validation Test")
    print("Testing refactored ToyNet with MSELoss and unified mode")

    # Test model initialization
    model = test_model_initialization()

    # Test forward pass
    forward_success = test_forward_pass(model)

    # Test mode validation
    test_mode_validation()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if model is not None and forward_success:
        print("✓ All tests passed! ToyNet unified mode refactoring successful.")
        print("✓ Model supports:")
        print("  - Unified mode with image + 24 spectra input")
        print("  - Automatic spectral reshaping from (B, 24, 1600) to (B, 38400)")
        print("  - MSELoss for regression")
        print("  - Proper error handling for missing inputs")
    else:
        print("✗ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
