import os
import logging
import torch
from transformers import Trainer, TrainingArguments
import metallo as st
from metallo import (
    load_config_with_args,
    create_datasets,
    ToyNetConfig,
    ToyNet,
)

log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=log_format, datefmt=date_format, level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "toynet": {"model": ToyNet, "config": ToyNetConfig},
}


def main():
    config, config_path = st.load_config_with_args()
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Create datasets based on mode
    if config.data.mode == "image":
        datasets = st.create_datasets(
            image_dir=config.data.image_dir,
            labels_csv_path=getattr(config.data, "labels_csv_path", None),
            label_column=getattr(config.data, "label_column", "target"),
            split_ratios=config.data.split_ratios,
            mode=config.data.mode,
        )
    elif config.data.mode == "spectral":
        datasets = st.create_datasets(
            spectral_csv_path=config.data.spectral_csv_path,
            spectral_columns=config.data.spectral_columns,
            labels_csv_path=getattr(config.data, "labels_csv_path", None),
            label_column=getattr(config.data, "label_column", "target"),
            split_ratios=config.data.split_ratios,
            scale_spectral=config.data.scale_spectral,
            mode=config.data.mode,
        )
    else:  # multimodal
        datasets = st.create_datasets(
            image_dir=config.data.image_dir,
            spectral_csv_path=config.data.spectral_csv_path,
            spectral_columns=config.data.spectral_columns,
            labels_csv_path=getattr(config.data, "labels_csv_path", None),
            label_column=getattr(config.data, "label_column", "target"),
            split_ratios=config.data.split_ratios,
            scale_spectral=config.data.scale_spectral,
            mode=config.data.mode,
        )

    # Set up callbacks
    callbacks = []
    trainloss_callback = st.TrainingLossCallback()
    callbacks.append(trainloss_callback)
    wandb_callback = st.WandbLogger(config, config_path)
    callbacks.append(wandb_callback)

    # Create model configuration
    model_config_kwargs = {
        "mode": config.model.mode,
        "num_classes": config.model.num_classes,
        "task_type": config.model.task_type,
        "dropout": config.model.dropout,
    }

    # Add mode-specific parameters
    if config.model.mode in ["image", "multimodal"]:
        model_config_kwargs.update(
            {
                "image_backbone": config.model.image_backbone,
                "image_feature_dim": config.model.image_feature_dim,
            }
        )

    if config.model.mode in ["spectral", "multimodal"]:
        # Update spectral input dim based on actual data
        if hasattr(config.data, "spectral_columns"):
            actual_spectral_dim = len(config.data.spectral_columns)
        else:
            actual_spectral_dim = config.model.spectral_input_dim

        model_config_kwargs.update(
            {
                "spectral_input_dim": actual_spectral_dim,
                "spectral_hidden_dim": config.model.spectral_hidden_dim,
                "spectral_num_layers": config.model.spectral_num_layers,
            }
        )

    if config.model.mode == "multimodal":
        model_config_kwargs.update(
            {
                "fusion_hidden_dim": config.model.fusion_hidden_dim,
                "fusion_num_layers": config.model.fusion_num_layers,
            }
        )

    config_model = MODEL_MAP[config.model.name]["config"](**model_config_kwargs)
    model = MODEL_MAP[config.model.name]["model"](config_model)

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Model mode: {config.model.mode}")
    logger.info(f"Task type: {config.model.task_type}")

    if config.eval_only:
        logger.info("Running in eval mode.")
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
            eval_dataset=datasets["eval"],
            compute_metrics=st.compute_regression_metrics,
        )
        with torch.no_grad():
            eval_results = trainer.evaluate()
            logger.info("Eval Results:")
            for metric, value in eval_results.items():
                if metric.startswith("eval_"):
                    logger.info(f"  {metric}: {value:.6f}")
    else:
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


if __name__ == "__main__":
    main()
