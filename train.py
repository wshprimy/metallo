import os
import logging
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import Trainer, TrainingArguments

import metallo as st
from metallo.data import MetalloDS
from metallo.models import MODEL_MAPPING

log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=log_format, datefmt=date_format, level=logging.INFO)
logger = logging.getLogger(__name__)


def create_image_transforms():
    """Create image transformations."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(336),
            transforms.CenterCrop(300),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_datasets_from_config(config):
    """Create datasets based on configuration."""
    image_transform = create_image_transforms()
    train_dataset = MetalloDS(
        data_dir=config.data.data_dir,
        mode=config.data.mode,
        split="train",
        image_transform=image_transform,
        process_images=getattr(config.data, "process_images", False),
        normalize_spectral=getattr(config.data, "normalize_spectral", True),
    )
    eval_dataset = MetalloDS(
        data_dir=config.data.data_dir,
        mode=config.data.mode,
        split="eval",
        image_transform=image_transform,
        process_images=getattr(config.data, "process_images", False),
        normalize_spectral=getattr(config.data, "normalize_spectral", True),
    )
    test_dataset = MetalloDS(
        data_dir=config.data.data_dir,
        mode=config.data.mode,
        split="test",
        image_transform=image_transform,
        process_images=getattr(config.data, "process_images", False),
        normalize_spectral=getattr(config.data, "normalize_spectral", True),
    )

    logger.info(f"Created datasets:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Eval: {len(eval_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    logger.info(f"  Mode: {config.data.mode}")

    return {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}


def main():
    """Main training function."""
    config, config_path = st.load_config_with_args()
    os.makedirs(config.training.output_dir, exist_ok=True)
    datasets = create_datasets_from_config(config)

    callbacks = []
    trainloss_callback = st.TrainingLossCallback()
    callbacks.append(trainloss_callback)
    wandb_callback = st.WandbLogger(config, config_path)
    callbacks.append(wandb_callback)

    model_config = MODEL_MAPPING[config.model.name]["config"](
        mode=config.model.mode,
        image_backbone=getattr(config.model, "image_backbone", "resnet18"),
        spectral_input_dim=getattr(config.model, "spectral_input_dim", 100),
        hidden_dim=getattr(config.model, "hidden_dim", 256),
        dropout=getattr(config.model, "dropout", 0.2),
        num_outputs=getattr(config.model, "num_outputs", 1),
    )

    if config.mode.test_only is False:
        logger.info("Running in training mode.")
        model = MODEL_MAPPING[config.model.name]["model"](model_config)
        logger.info(f"Model architecture:\n{model}")
        logger.info(f"Model mode: {config.model.mode}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        training_args = TrainingArguments(**config.training.__dict__)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            compute_metrics=st.compute_regression_metrics,
            callbacks=callbacks,
        )

        logger.info("Start training.")
        trainer.train()

        logger.info("Save final model.")
        trainer.save_model()

        # Optional: Evaluate on test set
        # if len(datasets["test"]) > 0:
        #     logger.info("Evaluating on test set.")
        #     test_results = trainer.evaluate(eval_dataset=datasets["test"])
        #     logger.info("Test Results:")
        #     for metric, value in test_results.items():
        #         if metric.startswith("eval_"):
        #             logger.info(f"  {metric}: {value:.6f}")

    else:
        logger.info("Running in test mode.")
        model = MODEL_MAPPING[config.model.name]["model"](model_config).from_pretrained(
            config.mode.checkpoint_path, config=model_config
        )
        logger.info(f"Model architecture:\n{model}")
        logger.info(f"Model mode: {config.model.mode}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        training_args_dict = config.training.__dict__.copy()
        training_args_dict.update(
            {
                "eval_strategy": "no",
                "save_strategy": "no",
                "logging_strategy": "no",
                "num_train_epochs": 0,
            }
        )
        training_args = TrainingArguments(**training_args_dict)
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=datasets["test"],
            compute_metrics=st.compute_regression_metrics,
        )

        with torch.no_grad():
            prediction_output = trainer.predict(test_dataset=datasets["test"])

            pred = prediction_output.predictions
            labels = prediction_output.label_ids
            logger.info(f"Raw Predictions: {pred}")
            logger.info(f"True Labels: {labels}")

            pred = pred.flatten()
            labels = labels.flatten()
            abs_error = np.abs(pred - labels)
            df = pd.DataFrame(
                {"Prediction": pred, "Target": labels, "Absolute Error": abs_error}
            )
            csv_path = os.path.join(
                "./", config.mode.checkpoint_path, "test_results.csv"
            )
            df.to_csv(csv_path, index=False)

        with torch.no_grad():
            test_results = trainer.evaluate()
            logger.info("Eval Results:")
            for metric, value in test_results.items():
                if metric.startswith("eval_"):
                    logger.info(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    main()
