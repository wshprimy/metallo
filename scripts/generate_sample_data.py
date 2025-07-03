#!/usr/bin/env python3
"""
Generate sample data for testing the metallographic and spectral analysis pipeline.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import argparse


def generate_sample_images(output_dir: str, num_samples: int = 100):
    """Generate sample metallographic images."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        # Create a simple synthetic metallographic-like image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Add some random grain-like structures
        for _ in range(np.random.randint(10, 30)):
            x = np.random.randint(0, 224)
            y = np.random.randint(0, 224)
            size = np.random.randint(5, 20)
            color = tuple(np.random.randint(50, 200, 3))
            draw.ellipse(
                [x - size // 2, y - size // 2, x + size // 2, y + size // 2], fill=color
            )

        # Add some lines to simulate grain boundaries
        for _ in range(np.random.randint(5, 15)):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            color = tuple(np.random.randint(20, 100, 3))
            draw.line([x1, y1, x2, y2], fill=color, width=np.random.randint(1, 3))

        img.save(os.path.join(output_dir, f"sample_{i:03d}.png"))

    print(f"Generated {num_samples} sample images in {output_dir}")


def generate_sample_spectral_data(
    output_file: str, num_samples: int = 100, num_features: int = 50
):
    """Generate sample spectral data."""
    # Generate synthetic spectral data with some correlation structure
    np.random.seed(42)

    # Create base spectral patterns
    wavelengths = np.linspace(400, 800, num_features)

    data = []
    targets = []

    for i in range(num_samples):
        # Generate different types of spectral signatures
        spectrum_type = np.random.choice(["metallic", "oxide", "carbide"])

        if spectrum_type == "metallic":
            # Metallic signature: higher reflectance in red/IR
            base_spectrum = 0.3 + 0.4 * np.exp(-((wavelengths - 700) ** 2) / 5000)
            target = np.random.normal(0.8, 0.1)  # High hardness
        elif spectrum_type == "oxide":
            # Oxide signature: absorption bands
            base_spectrum = 0.5 - 0.3 * np.exp(-((wavelengths - 550) ** 2) / 3000)
            target = np.random.normal(0.3, 0.1)  # Low hardness
        else:  # carbide
            # Carbide signature: mixed pattern
            base_spectrum = 0.4 + 0.2 * np.sin((wavelengths - 400) / 50)
            target = np.random.normal(0.6, 0.1)  # Medium hardness

        # Add noise
        noise = np.random.normal(0, 0.05, num_features)
        spectrum = base_spectrum + noise

        # Ensure positive values
        spectrum = np.clip(spectrum, 0.01, 1.0)

        data.append(spectrum)
        targets.append(target)

    # Create DataFrame
    feature_names = [f"wavelength_{int(w)}" for w in wavelengths]
    df = pd.DataFrame(data, columns=feature_names)

    # Save spectral data
    df.to_csv(output_file, index=False)

    # Save targets
    target_file = output_file.replace(".csv", "_labels.csv")
    pd.DataFrame({"target": targets}).to_csv(target_file, index=False)

    print(f"Generated {num_samples} spectral samples with {num_features} features")
    print(f"Spectral data saved to: {output_file}")
    print(f"Labels saved to: {target_file}")

    return feature_names


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for testing")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_features", type=int, default=50, help="Number of spectral features"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="Output directory"
    )

    args = parser.parse_args()

    # Generate sample images
    image_dir = os.path.join(args.output_dir, "images")
    generate_sample_images(image_dir, args.num_samples)

    # Generate sample spectral data
    spectral_file = os.path.join(args.output_dir, "sample_spectral.csv")
    feature_names = generate_sample_spectral_data(
        spectral_file, args.num_samples, args.num_features
    )

    # Update default config with actual feature names
    config_file = "./configs/default.yaml"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_content = f.read()

        # Replace the spectral_columns section
        feature_list = "\n".join(
            [f'    - "{name}"' for name in feature_names[:10]]
        )  # Use first 10 features

        # Find and replace the spectral_columns section
        import re

        pattern = r"spectral_columns:.*?(?=\n  [a-zA-Z]|\n\n|\Z)"
        replacement = f"spectral_columns:\n{feature_list}"

        new_content = re.sub(pattern, replacement, config_content, flags=re.DOTALL)

        # Also update spectral_input_dim
        new_content = re.sub(
            r"spectral_input_dim: \d+", f"spectral_input_dim: 10", new_content
        )

        with open(config_file, "w") as f:
            f.write(new_content)

        print(f"Updated {config_file} with actual feature names")

    print("\nSample data generation complete!")
    print(f"Images: {image_dir}")
    print(f"Spectral data: {spectral_file}")
    print(f"Labels: {spectral_file.replace('.csv', '_labels.csv')}")


if __name__ == "__main__":
    main()
