import torch
import torch.nn as nn
import torchvision.models as models
from transformers import PreTrainedModel, PretrainedConfig
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertIntermediate,
    BertOutput,
)


class MetalloNetConfig(PretrainedConfig):
    """Configuration class for ToyNet to work with transformers Trainer."""

    def __init__(
        self,
        mode="unified",
        image_backbone="resnet18",
        spectral_dim=1600,
        hidden_dim=128,
        dropout=0.2,
        num_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.image_backbone = image_backbone
        self.spectral_dim = spectral_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_outputs = num_outputs


class SpatialModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),  # Dimension reduction
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, kernel_size=1),  # Output single-channel weights
            nn.Sigmoid(),  # Ensure weights are in [0, 1] range
        )

    def forward(self, x):  # Input: [B, 128, 6, 4]
        # Generate attention weight map
        attention_weights = self.attention(x)  # Output: [B, 1, 6, 4]
        # Apply weights to original features
        return x * attention_weights  # [B, 128, 6, 4]


class SpectralEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Stage 1: Spectral dimension compression (depthwise separable)
        self.spectral_compress = nn.Sequential(
            # Depthwise convolution: independent convolution for each spectral channel
            nn.Conv2d(
                config.spectral_dim,
                config.spectral_dim,
                kernel_size=3,
                padding=1,
                groups=1600,
            ),
            nn.BatchNorm2d(config.spectral_dim),
            nn.ReLU(),
            # Pointwise convolution: feature fusion across spectral channels
            nn.Conv2d(config.spectral_dim, 400, kernel_size=1),
            nn.BatchNorm2d(400),
            nn.ReLU(),
        )

        # Stage 2: Further compression
        self.feature_extract = nn.Sequential(
            nn.Conv2d(400, 200, kernel_size=3, padding=1),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, config.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(),
        )

        self.spatial_module = SpatialModule(config.hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x):  # x: [B, 24, 1600]
        b, points, c = x.shape
        x = x.view(b, 6, 4, c).permute(0, 3, 1, 2)  # x: [B, 1600, 6, 4]
        x = self.spectral_compress(x)  # [B, 400, 6, 4]
        x = self.feature_extract(x)  # [B, 128, 6, 4]
        x = self.spatial_module(x)  # [B, 128, 6, 4] with attention
        x = self.global_pool(x).flatten(1)  # [B, 128]
        return self.fc(x)


class MetallographicDominantTransformerFusion(nn.Module):
    """
    Placeholder for a fusion module that emphasizes metallographic features.
    TBD.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.head = nn.Linear(self.config.hidden_dim // 2, self.config.num_outputs)

    def forward(self, img_feat, spec_feat):
        # img_feat & spec_feat: [B, hidden_dim]
        x = torch.cat([img_feat, spec_feat], dim=1)  # [B, 2 * hidden_dim]
        x = self.fusion(x)  # [B, hidden_dim // 2]
        x = self.head(x)  # [B, num_outputs]
        return x


class MetalloNet(PreTrainedModel):
    """
    Neural network for DOS prediction from metallographic and spectral data.
    """

    config_class = MetalloNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(MetalloNetConfig(**kwargs))
        self.config = config
        assert (
            self.config.mode == "unified"
        ), "Currently only 'unified' mode is supported."

        self.image_encoder = self._build_image_encoder()
        self.spectral_encoder = SpectralEncoder(self.config)
        self.fusion = MetallographicDominantTransformerFusion(self.config)

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
        backbone.fc = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(backbone_output_dim, self.config.hidden_dim),
        )
        return backbone

    def forward(self, image, spectral, labels=None, **inputs):
        img_enc = self.image_encoder(image)
        spec_enc = self.spectral_encoder(spectral)
        pred = self.fusion(img_enc, spec_enc)

        loss = None
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(-1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, labels)

        return {"loss": loss, "pred": pred}
