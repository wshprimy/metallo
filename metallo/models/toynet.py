from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torchvision.models as models


class ToyNetConfig(PretrainedConfig):
    def __init__(
        self,
        mode="multimodal",  # "image", "spectral", "multimodal"
        num_classes=1,  # For regression tasks, use 1; for classification, set accordingly
        task_type="regression",  # "regression" or "classification"
        # Image-related parameters
        image_backbone="resnet18",  # "resnet18", "resnet50", etc.
        image_feature_dim=512,
        # Spectral-related parameters
        spectral_input_dim=100,  # Number of spectral features
        spectral_hidden_dim=256,
        spectral_num_layers=2,
        # Fusion parameters (for multimodal)
        fusion_hidden_dim=256,
        fusion_num_layers=2,
        # Common parameters
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.num_classes = num_classes
        self.task_type = task_type

        self.image_backbone = image_backbone
        self.image_feature_dim = image_feature_dim

        self.spectral_input_dim = spectral_input_dim
        self.spectral_hidden_dim = spectral_hidden_dim
        self.spectral_num_layers = spectral_num_layers

        self.fusion_hidden_dim = fusion_hidden_dim
        self.fusion_num_layers = fusion_num_layers

        self.dropout = dropout


class ToyNet(PreTrainedModel):
    config_class = ToyNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Image encoder (if needed)
        if config.mode in ["image", "multimodal"]:
            self.image_encoder = self._build_image_encoder(config)

        # Spectral encoder (if needed)
        if config.mode in ["spectral", "multimodal"]:
            self.spectral_encoder = self._build_spectral_encoder(config)

        # Fusion and classification head
        self.classifier = self._build_classifier(config)

    def _build_image_encoder(self, config):
        """Build image feature extractor using pretrained CNN."""
        if config.image_backbone == "resnet18":
            backbone = models.resnet18(pretrained=True)
            backbone.fc = nn.Identity()  # Remove final classification layer
            feature_dim = 512
        elif config.image_backbone == "resnet50":
            backbone = models.resnet50(pretrained=True)
            backbone.fc = nn.Identity()
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported image backbone: {config.image_backbone}")

        # Update config with actual feature dimension
        config.image_feature_dim = feature_dim

        return nn.Sequential(
            backbone,
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, config.image_feature_dim),
        )

    def _build_spectral_encoder(self, config):
        """Build spectral feature extractor using MLP."""
        layers = []
        input_dim = config.spectral_input_dim

        for i in range(config.spectral_num_layers):
            layers.append(nn.Linear(input_dim, config.spectral_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = config.spectral_hidden_dim

        return nn.Sequential(*layers)

    def _build_classifier(self, config):
        """Build final classification/regression head."""
        # Determine input dimension for classifier
        if config.mode == "image":
            input_dim = config.image_feature_dim
        elif config.mode == "spectral":
            input_dim = config.spectral_hidden_dim
        else:  # multimodal
            input_dim = config.image_feature_dim + config.spectral_hidden_dim

        # Build fusion layers for multimodal case
        layers = []
        if config.mode == "multimodal":
            for i in range(config.fusion_num_layers):
                layers.append(nn.Linear(input_dim, config.fusion_hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
                input_dim = config.fusion_hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, config.num_classes))

        # Add activation for classification
        if config.task_type == "classification" and config.num_classes > 1:
            layers.append(nn.Softmax(dim=-1))

        return nn.Sequential(*layers)

    def forward(self, x=None, image=None, spectral=None, labels=None):
        """
        Forward pass supporting different input modes.

        Args:
            x: Single input tensor (for backward compatibility)
            image: Image tensor [B, C, H, W]
            spectral: Spectral tensor [B, spectral_dim]
            labels: Ground truth labels [B, num_classes]

        Returns:
            Dictionary with loss and logits
        """
        features = []

        # Handle backward compatibility
        if x is not None and image is None and spectral is None:
            if self.config.mode == "image":
                image = x
            elif self.config.mode == "spectral":
                spectral = x
            else:
                raise ValueError(
                    "For multimodal mode, must provide 'image' and 'spectral' separately"
                )

        # Extract image features
        if self.config.mode in ["image", "multimodal"]:
            if image is None:
                raise ValueError(f"Image input required for mode: {self.config.mode}")
            image_features = self.image_encoder(image)  # [B, image_feature_dim]
            features.append(image_features)

        # Extract spectral features
        if self.config.mode in ["spectral", "multimodal"]:
            if spectral is None:
                raise ValueError(
                    f"Spectral input required for mode: {self.config.mode}"
                )
            spectral_features = self.spectral_encoder(
                spectral
            )  # [B, spectral_hidden_dim]
            features.append(spectral_features)

        # Concatenate features for multimodal case
        if len(features) > 1:
            combined_features = torch.cat(features, dim=-1)
        else:
            combined_features = features[0]

        # Get predictions
        logits = self.classifier(combined_features)  # [B, num_classes]

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if self.config.task_type == "regression":
                loss_fn = nn.MSELoss()
                # Ensure shapes match for regression
                if logits.shape != labels.shape:
                    if len(labels.shape) == 1:
                        labels = labels.unsqueeze(-1)
                loss = loss_fn(logits, labels)
            else:  # classification
                loss_fn = nn.CrossEntropyLoss()
                if len(labels.shape) > 1 and labels.shape[-1] == 1:
                    labels = labels.squeeze(-1)
                loss = loss_fn(logits, labels.long())

        return {"loss": loss, "logits": logits}
