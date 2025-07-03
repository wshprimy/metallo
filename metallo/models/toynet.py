import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any


class SimpleToyNet(nn.Module):
    """
    Simple neural network for DOS prediction from metallographic and/or spectral data.
    Supports image-only, spectral-only, and multimodal modes.
    """
    
    def __init__(
        self,
        mode: str = "multimodal",  # "image", "spectral", "multimodal"
        image_backbone: str = "resnet18",
        spectral_input_dim: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_outputs: int = 1  # For DOS regression
    ):
        """
        Initialize ToyNet.
        
        Args:
            mode: Operating mode - "image", "spectral", or "multimodal"
            image_backbone: Backbone for image feature extraction
            spectral_input_dim: Input dimension for spectral data
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            num_outputs: Number of output values (1 for DOS regression)
        """
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        # Image encoder
        if mode in ["image", "multimodal"]:
            self.image_encoder = self._build_image_encoder(image_backbone, hidden_dim, dropout)
        
        # Spectral encoder
        if mode in ["spectral", "multimodal"]:
            self.spectral_encoder = self._build_spectral_encoder(
                spectral_input_dim, hidden_dim, dropout
            )
        
        # Final regression head
        if mode == "multimodal":
            input_dim = hidden_dim * 2  # Concatenated features
        else:
            input_dim = hidden_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs)
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
            nn.ReLU()
        )
    
    def _build_spectral_encoder(self, input_dim: int, output_dim: int, dropout: float):
        """Build spectral feature extractor."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x=None, image=None, spectral=None, labels=None):
        """
        Forward pass.
        
        Args:
            x: Single input (for backward compatibility)
            image: Image tensor [B, C, H, W]
            spectral: Spectral tensor [B, spectral_dim]
            labels: Ground truth DOS values [B] or [B, 1]
        
        Returns:
            Dictionary with loss and predictions
        """
        features = []
        
        # Handle backward compatibility
        if x is not None and image is None and spectral is None:
            if self.mode == "image":
                image = x
            elif self.mode == "spectral":
                spectral = x
            else:
                raise ValueError("For multimodal mode, provide 'image' and 'spectral' separately")
        
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
        
        return {
            "loss": loss,
            "logits": predictions,
            "predictions": predictions
        }


# Alias for backward compatibility
ToyNet = SimpleToyNet


def create_toynet(
    mode: str = "multimodal",
    image_backbone: str = "resnet18",
    spectral_input_dim: int = 100,
    hidden_dim: int = 256,
    dropout: float = 0.2
) -> SimpleToyNet:
    """
    Factory function to create a ToyNet model.
    
    Args:
        mode: Operating mode
        image_backbone: Image backbone architecture
        spectral_input_dim: Spectral input dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    
    Returns:
        Configured ToyNet model
    """
    return SimpleToyNet(
        mode=mode,
        image_backbone=image_backbone,
        spectral_input_dim=spectral_input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
