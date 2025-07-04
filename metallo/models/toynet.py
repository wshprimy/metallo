import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig


class ToyNetConfig(PretrainedConfig):
    """Configuration class for ToyNet to work with transformers Trainer."""

    def __init__(
        self,
        mode="multimodal",
        image_backbone="resnet18",
        spectral_input_dim=100,
        hidden_dim=256,
        dropout=0.2,
        num_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.image_backbone = image_backbone
        self.spectral_input_dim = spectral_input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_outputs = num_outputs


class ToyNet(PreTrainedModel):
    """
    Neural network for DOS prediction from metallographic and/or spectral data.
    Supports image-only, spectral-only, and multimodal modes.
    Compatible with transformers Trainer.
    """

    config_class = ToyNetConfig

    def __init__(self, config=None, **kwargs):
        """
        Initialize ToyNet.

        Args:
            config: ToyNetConfig object or None for backward compatibility
            **kwargs: Direct parameters for backward compatibility
        """
        # Handle both config-based and direct parameter initialization
        if config is not None:
            super().__init__(config)
            self.mode = config.mode
            self.hidden_dim = config.hidden_dim
            image_backbone = config.image_backbone
            spectral_input_dim = config.spectral_input_dim
            dropout = config.dropout
            num_outputs = config.num_outputs
        else:
            # Backward compatibility: direct parameter initialization
            super().__init__(ToyNetConfig(**kwargs))
            self.mode = kwargs.get("mode", "multimodal")
            self.hidden_dim = kwargs.get("hidden_dim", 256)
            image_backbone = kwargs.get("image_backbone", "resnet18")
            spectral_input_dim = kwargs.get("spectral_input_dim", 100)
            dropout = kwargs.get("dropout", 0.2)
            num_outputs = kwargs.get("num_outputs", 1)

        # Image encoder
        if self.mode in ["image", "multimodal"]:
            self.image_encoder = self._build_image_encoder(
                image_backbone, self.hidden_dim, dropout
            )

        # Spectral encoder
        if self.mode in ["spectral", "multimodal"]:
            self.spectral_encoder = self._build_spectral_encoder(
                spectral_input_dim, self.hidden_dim, dropout
            )

        # Final regression head
        if self.mode == "multimodal":
            input_dim = self.hidden_dim * 2  # Concatenated features
        else:
            input_dim = self.hidden_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_outputs),
        )

    def _build_image_encoder(self, backbone_name: str, output_dim: int, dropout: float):
        """Build image feature extractor."""
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
            backbone_output_dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            backbone_output_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove the final classification layer
        backbone.fc = nn.Identity()

        return nn.Sequential(
            backbone,
            nn.Dropout(dropout),
            nn.Linear(backbone_output_dim, output_dim),
            nn.ReLU(),
        )

    def _build_spectral_encoder(self, input_dim: int, output_dim: int, dropout: float):
        """Build spectral feature extractor."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x=None, image=None, spectral=None, labels=None, **inputs):
        """
        Forward pass compatible with transformers Trainer.

        Args:
            x: Single input (for backward compatibility)
            image: Image tensor [B, C, H, W]
            spectral: Spectral tensor [B, spectral_dim]
            labels: Ground truth DOS values [B] or [B, 1]
            **inputs: Additional inputs from transformers Trainer

        Returns:
            Dictionary with loss and predictions
        """
        # Handle inputs from transformers Trainer
        if image is None and spectral is None and inputs:
            image = inputs.get("image", None)
            spectral = inputs.get("spectral", None)
            labels = inputs.get("labels", None)

        features = []

        # Handle backward compatibility
        if x is not None and image is None and spectral is None:
            if self.mode == "image":
                image = x
            elif self.mode == "spectral":
                spectral = x
            else:
                raise ValueError(
                    "For multimodal mode, provide 'image' and 'spectral' separately"
                )

        # Extract image features
        if self.mode in ["image", "multimodal"]:
            if image is None:
                raise ValueError(f"Image input required for mode: {self.mode}")
            img_features = self.image_encoder(image)
            features.append(img_features)

        # Extract spectral features
        if self.mode in ["spectral", "multimodal"]:
            if spectral is None:
                raise ValueError(f"Spectral input required for mode: {self.mode}")
            spec_features = self.spectral_encoder(spectral)
            features.append(spec_features)

        # Combine features
        if len(features) > 1:
            combined_features = torch.cat(features, dim=-1)
        else:
            combined_features = features[0]

        # Get predictions
        predictions = self.regressor(combined_features)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(-1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, labels)

        return {"loss": loss, "logits": predictions, "predictions": predictions}
