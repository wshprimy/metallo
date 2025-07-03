"""
Checkpoint management callback
"""

import os
from transformers import TrainerCallback


class CheckpointCallback(TrainerCallback):
    """
    Enhanced checkpoint callback with custom save logic
    """

    def __init__(self, save_best_only=True, metric_name="eval_loss", mode="min"):
        """
        Initialize checkpoint callback

        Args:
            save_best_only: Whether to save only the best model
            metric_name: Metric to monitor for best model
            mode: 'min' or 'max' for best metric comparison
        """
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        Check if current model is the best and save if needed
        """
        if not logs or self.metric_name not in logs:
            return

        current_metric = logs[self.metric_name]

        # Check if this is the best model
        is_best = False
        if self.mode == "min" and current_metric < self.best_metric:
            self.best_metric = current_metric
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            self.best_metric = current_metric
            is_best = True

        # Save best model
        if is_best and self.save_best_only:
            best_model_path = os.path.join(args.output_dir, "best_model")
            kwargs["model"].save_pretrained(best_model_path)
            if hasattr(kwargs, "tokenizer") and kwargs["tokenizer"]:
                kwargs["tokenizer"].save_pretrained(best_model_path)

            print(f"New best model saved with {self.metric_name}: {current_metric}")
