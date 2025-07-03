import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class TrainingLossCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if "train_loss" in logs:
                self.train_losses.append(logs["train_loss"])
                logger.info(
                    f"Step {state.global_step}: train_loss = {logs['train_loss']:.6f}"
                )
            if "eval_loss" in logs:
                logger.info(
                    f"Step {state.global_step}: eval_loss = {logs['eval_loss']:.6f}"
                )

    def on_epoch_end(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and "train_loss" in logs:
            logger.info(
                f"Epoch {state.epoch}: Final train_loss = {logs['train_loss']:.6f}"
            )

    def on_train_end(self, args, state, control, model=None, logs=None, **kwargs):
        if self.train_losses:
            final_loss = self.train_losses[-1]
            best_loss = min(self.train_losses)
            logger.info(f"Training completed!")
            logger.info(f"Final train_loss: {final_loss:.6f}")
            logger.info(f"Best train_loss: {best_loss:.6f}")
            logger.info(f"Total training steps: {len(self.train_losses)}")
