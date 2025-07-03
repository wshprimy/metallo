import wandb
import os
from datetime import datetime
from transformers import TrainerCallback


class WandbLogger(TrainerCallback):
    """
    Custom W&B callback for logging training progress and uploading config
    """

    def __init__(self, config, config_path=None):
        """
        Initialize W&B logger

        Args:
            config: Configuration object
            config_path: Path to YAML config file to upload
        """
        self.config = config
        self.config_path = config_path

        # Initialize W&B
        date_format = "%Y-%m-%d %H:%M:%S"
        current_time = datetime.now().strftime(date_format)
        wandb_name = (
            config.model.name
            + (f"-{config.wandb.name}" if config.wandb.name else "")
            + f"-{current_time}"
        )
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=self._config_to_dict(config),
            name=wandb_name,
            tags=config.wandb.tags,
        )

        # Upload config file if provided
        if config_path and os.path.exists(config_path):
            wandb.save(config_path)

    def _config_to_dict(self, config):
        """Convert config object to dictionary for W&B"""
        if hasattr(config, "__dict__"):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, "__dict__"):
                    result[key] = self._config_to_dict(value)
                else:
                    result[key] = value
            return result
        return config

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to W&B"""
        if logs:
            wandb.log(logs, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Finish W&B run"""
        wandb.finish()
