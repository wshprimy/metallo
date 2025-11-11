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

from .attention import SpatialCrossAttention, CrossModalAttention, DynamicGatingNetwork


class MetalloNetConfig(PretrainedConfig):
    """Configuration class for MetalloNet to work with transformers Trainer."""

    def __init__(
        self,
        mode="unified",
        image_backbone="resnet18",
        spectral_dim=1600,
        hidden_dim=128,
        dropout=0.2,
        num_outputs=1,
        fusion_type="progressive",  # "progressive" or "legacy"
        aux_loss_weight=0.1,  # Weight for auxiliary loss encouraging metallographic dominance
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.image_backbone = image_backbone
        self.spectral_dim = spectral_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_outputs = num_outputs
        self.fusion_type = fusion_type
        self.aux_loss_weight = aux_loss_weight


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


class ShallowFusionModule(nn.Module):
    """
    Shallow fusion: bidirectional cross-attention at feature map level.
    Enables spatial correspondence learning between metallographic and spectral features.
    """

    def __init__(self, img_channels=512, spec_channels=128, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Cross-attention: image attends to spectral
        self.img_to_spec_attn = SpatialCrossAttention(
            query_channels=img_channels,
            kv_channels=spec_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention: spectral attends to image
        self.spec_to_img_attn = SpatialCrossAttention(
            query_channels=spec_channels,
            kv_channels=img_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, img_feat, spec_feat):
        """
        Args:
            img_feat: [B, C_img, H_img, W_img] - e.g., [B, 512, 7, 7]
            spec_feat: [B, C_spec, H_spec, W_spec] - e.g., [B, 128, 6, 4]
        Returns:
            img_enhanced: [B, hidden_dim, H_img, W_img]
            spec_enhanced: [B, hidden_dim, H_spec, W_spec]
        """
        # Bidirectional cross-attention
        img_enhanced = self.img_to_spec_attn(img_feat, spec_feat)  # [B, D, H_img, W_img]
        spec_enhanced = self.spec_to_img_attn(spec_feat, img_feat)  # [B, D, H_spec, W_spec]

        return img_enhanced, spec_enhanced


class DeepFusionModule(nn.Module):
    """
    Deep fusion: co-attention + dynamic gating at semantic level.
    Learns to weight metallographic vs spectral features adaptively.
    Includes auxiliary loss to encourage metallographic dominance.
    """

    def __init__(self, hidden_dim=128, num_heads=4, gating_hidden_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Co-attention: image attends to spectral
        self.img_to_spec_attn = CrossModalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Co-attention: spectral attends to image
        self.spec_to_img_attn = CrossModalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Dynamic gating network
        self.gating = DynamicGatingNetwork(
            input_dim=hidden_dim * 2,
            num_modalities=2,
            hidden_dim=gating_hidden_dim,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, img_vec, spec_vec):
        """
        Args:
            img_vec: [B, hidden_dim] - pooled metallographic features
            spec_vec: [B, hidden_dim] - pooled spectral features
        Returns:
            fused: [B, hidden_dim] - fused features
            gate_weights: [B, 2] - [w_img, w_spec] for interpretability
        """
        # Add sequence dimension for attention: [B, D] -> [B, 1, D]
        img_seq = img_vec.unsqueeze(1)
        spec_seq = spec_vec.unsqueeze(1)

        # Co-attention
        img_attended = self.img_to_spec_attn(img_seq, spec_seq).squeeze(1)  # [B, D]
        spec_attended = self.spec_to_img_attn(spec_seq, img_seq).squeeze(1)  # [B, D]

        # Dynamic gating: learn importance weights
        gate_weights = self.gating(img_attended, spec_attended)  # [B, 2]

        # Weighted fusion
        w_img = gate_weights[:, 0:1]  # [B, 1]
        w_spec = gate_weights[:, 1:2]  # [B, 1]
        fused = w_img * img_attended + w_spec * spec_attended  # [B, hidden_dim]

        # Layer normalization
        fused = self.layer_norm(fused)

        return fused, gate_weights


class ProgressiveMultiLevelFusion(nn.Module):
    """
    Progressive multi-level fusion combining shallow cross-attention and deep co-attention gating.
    Learns both spatial correspondences and semantic importance adaptively.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Shallow fusion: feature-level cross-attention
        self.shallow_fusion = ShallowFusionModule(
            img_channels=512,  # ResNet18 output channels before final pooling
            spec_channels=128,  # SpectralEncoder output channels
            hidden_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
        )

        # Global pooling for deep fusion
        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.spec_pool = nn.AdaptiveAvgPool2d(1)

        # Deep fusion: semantic-level co-attention + gating
        self.deep_fusion = DeepFusionModule(
            hidden_dim=config.hidden_dim,
            num_heads=4,
            gating_hidden_dim=64,
            dropout=config.dropout,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_outputs),
        )

        # Auxiliary loss weight (encourages metallographic dominance)
        self.aux_loss_weight = 0.1

    def forward(self, img_feat, spec_feat):
        """
        Args:
            img_feat: [B, 512, 7, 7] - ResNet features (before final pooling)
            spec_feat: [B, 128, 6, 4] - Spectral features (before final pooling)
        Returns:
            pred: [B, num_outputs] - predictions
            aux_loss: scalar - auxiliary loss encouraging metallographic dominance
            gate_weights: [B, 2] - learned importance weights for interpretability
        """
        # Shallow fusion: cross-attention at feature map level
        img_enhanced, spec_enhanced = self.shallow_fusion(img_feat, spec_feat)
        # img_enhanced: [B, hidden_dim, 7, 7]
        # spec_enhanced: [B, hidden_dim, 6, 4]

        # Global pooling
        img_vec = self.img_pool(img_enhanced).flatten(1)  # [B, hidden_dim]
        spec_vec = self.spec_pool(spec_enhanced).flatten(1)  # [B, hidden_dim]

        # Deep fusion: co-attention + gating
        fused, gate_weights = self.deep_fusion(img_vec, spec_vec)
        # fused: [B, hidden_dim]
        # gate_weights: [B, 2] where [:, 0] is w_img, [:, 1] is w_spec

        # Prediction
        pred = self.head(fused)  # [B, num_outputs]

        # Auxiliary loss: encourage metallographic (image) dominance
        # Loss = -log(w_img) to maximize w_img
        # We want w_img > w_spec, so we penalize when w_img is small
        aux_loss = -torch.log(gate_weights[:, 0] + 1e-8).mean() * self.aux_loss_weight

        return pred, aux_loss, gate_weights


class MetallographicDominantTransformerFusion(nn.Module):
    """
    Legacy fusion module - kept for backward compatibility.
    Simple concatenation + MLP fusion.
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
    Supports both legacy fusion and progressive multi-level fusion.
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
        
        # Choose fusion module based on config
        if self.config.fusion_type == "progressive":
            self.fusion = ProgressiveMultiLevelFusion(self.config)
            self.use_progressive_fusion = True
        else:
            self.fusion = MetallographicDominantTransformerFusion(self.config)
            self.use_progressive_fusion = False

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

        # For progressive fusion, we need features before final pooling
        if self.use_progressive_fusion:
            # Remove avgpool and fc layers to get spatial features
            backbone.avgpool = nn.Identity()
            backbone.fc = nn.Identity()
        else:
            # Legacy: keep pooling, replace fc
            backbone.fc = nn.Sequential(
                nn.Dropout(self.config.dropout),
                nn.Linear(backbone_output_dim, self.config.hidden_dim),
            )
        return backbone

    def forward(self, image, spectral, labels=None, **inputs):
        """
        Forward pass with support for both fusion types.
        
        Returns:
            dict with keys:
                - loss: total loss (MSE + optional auxiliary loss)
                - pred: predictions
                - gate_weights: (optional) gating weights for progressive fusion
        """
        if self.use_progressive_fusion:
            # Get spatial features (before pooling)
            img_feat = self.image_encoder(image)  # [B, 512, 7, 7]
            spec_feat_spatial = self.spectral_encoder.spectral_compress(
                spectral.view(spectral.size(0), 6, 4, self.config.spectral_dim).permute(0, 3, 1, 2)
            )  # [B, 400, 6, 4]
            spec_feat_spatial = self.spectral_encoder.feature_extract(spec_feat_spatial)  # [B, 128, 6, 4]
            spec_feat_spatial = self.spectral_encoder.spatial_module(spec_feat_spatial)  # [B, 128, 6, 4]
            
            # Progressive fusion
            pred, aux_loss, gate_weights = self.fusion(img_feat, spec_feat_spatial)
            
            # Compute total loss
            loss = None
            if labels is not None:
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(-1)
                loss_fn = nn.MSELoss()
                mse_loss = loss_fn(pred, labels)
                loss = mse_loss + aux_loss
            
            return {
                "loss": loss,
                "pred": pred,
                "gate_weights": gate_weights,
                "aux_loss": aux_loss,
            }
        else:
            # Legacy fusion
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
