import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig


class ToyNetConfig(PretrainedConfig):
    """Configuration class for ToyNet to work with transformers Trainer."""

    def __init__(
        self,
        mode="unified",
        image_backbone="resnet18",
        hidden_dim=256,
        dropout=0.2,
        num_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.image_backbone = image_backbone
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_outputs = num_outputs


class SpectralEncoder(nn.Module):
    def __init__(self, config):
        super(SpectralEncoder, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=24, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, config.hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.max_pool1d(x, kernel_size=2)  # [batch_size, 32, 800]
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)  # [batch_size, 64, 400]

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)  # [batch_size, 128, 200]

        x = self.global_avg_pool(x)  # [batch_size, 128, 1]
        x = x.squeeze(-1)  # [batch_size, 128]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, config.hidden_dim]
        return x


class ToyNet(PreTrainedModel):
    """
    Neural network for DOS prediction from metallographic and spectral data.
    """

    config_class = ToyNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(ToyNetConfig(**kwargs))
        self.config = config
        assert (
            self.config.mode == "unified"
        ), "Currently only 'unified' mode is supported."

        self.image_encoder = self._build_image_encoder()
        self.spectral_encoder = SpectralEncoder(self.config)
        self.regressor = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_outputs),
        )

    def _build_image_encoder(self):
        """Build image feature extractor."""
        if self.config.image_backbone == "resnet18":
            backbone = models.resnet18(pretrained=True)
            backbone_output_dim = 512
        elif self.config.image_backbone == "resnet50":
            backbone = models.resnet50(pretrained=True)
            backbone_output_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.config.image_backbone}")

        # Remove the final classification layer
        backbone.fc = nn.Identity()
        return nn.Sequential(
            backbone,
            nn.Dropout(self.config.dropout),
            nn.Linear(backbone_output_dim, self.config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, image, spectral, labels=None, **inputs):
        img_enc = self.image_encoder(image)
        spec_enc = self.spectral_encoder(spectral)
        x = torch.cat([img_enc, spec_enc], dim=-1)
        pred = self.regressor(x)

        loss = None
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(-1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, labels)

        return {"loss": loss, "pred": pred}
