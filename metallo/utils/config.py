"""
Configuration management with YAML and argparse integration
"""

import logging
import argparse
import yaml
import os
from types import SimpleNamespace


def load_yaml_config(config_path):
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        SimpleNamespace: Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return dict_to_namespace(config_dict)


def dict_to_namespace(d):
    """
    Convert nested dict to nested SimpleNamespace

    Args:
        d: Dictionary to convert

    Returns:
        SimpleNamespace: Converted object
    """
    if isinstance(d, dict):
        namespace = SimpleNamespace()
        for key, value in d.items():
            setattr(namespace, key, dict_to_namespace(value))
        return namespace
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def load_config_with_args():
    """
    Load configuration from YAML and merge with command line arguments

    Returns:
        tuple: (config, config_path, args)
    """
    parser = argparse.ArgumentParser(description="SATemporal Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, help="Model name")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_yaml_config(args.config)
        config_path = args.config
    else:
        logging.error(f"Config file {args.config} not found, using default config")
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_train_epochs = args.num_epochs
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.model_name is not None:
        config.model.name = args.model_name
    return config, config_path
