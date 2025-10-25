# train.py
import yaml
import argparse
from pathlib import Path
from typing import Any
import logging
from src.tvp.trainer import train
from src.tvp.utils import set_seed

# Create a logger for this module
logger = logging.getLogger(__name__)

# Try to import IPEX, warn if not available
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    logger.warning("Intel Extension for PyTorch (IPEX) not found. XPU device will not be available.")
    pass


def set_nested_key(d: dict, key_string: str, value: Any):
    """
    Helper function to set a value in a nested dictionary using a dot-separated key.
    e.g., set_nested_key(config, "finetuning.lr", 0.01)
    """
    keys = key_string.split('.')
    current_level = d
    for key in keys[:-1]:
        # Create nested dictionaries if they don't exist
        current_level = current_level.setdefault(key, {})
    current_level[keys[-1]] = value


def main():
    """
    Main function to run the training process.
    Parses command-line arguments, loads and overrides config, and calls the trainer.
    """
    # 1. Setup argument parser to accept a config file path and optional overrides
    parser = argparse.ArgumentParser(description="Run a finetuning experiment for Task Vector Prediction.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--resume_id", type=str, help="Wandb run ID to resume a stopped training run.")
    
    # A generic override argument that can be used multiple times.
    parser.add_argument(
        "--set",
        dest="config_overrides",
        metavar="KEY=VALUE",
        nargs='+',
        action='append',
        help="Set a configuration value from the command line, e.g., --set data.batch_size=128 finetuning.lr=0.01"
    )
    
    args = parser.parse_args()

    # 2. Load the base configuration from the specified YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))

    # 3. Override config with any --set arguments from the command line
    override_parts = []
    if args.config_overrides:
        # The 'append' action creates a list of lists, so we need to flatten it.
        flat_overrides = [item for sublist in args.config_overrides for item in sublist]
        for override in flat_overrides:
            try:
                key, value_str = override.split('=', 1)
            except ValueError:
                raise ValueError(f"Invalid override format: {override}. Expected KEY=VALUE.")

            # Try to intelligently convert the value string to a Python type
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str # Keep as string if it's not a number or boolean
            
            set_nested_key(config, key, value)
            
            # For directory naming, use a sanitized key and value
            sanitized_key = key.split('.')[-1] # Use the last part of the key
            override_parts.append(f"{sanitized_key}{value}")

    # 4. Build the predictable "experiment group" directory name
    config_filename = Path(args.config).stem
    group_name_parts = [config_filename] + override_parts
    experiment_group_name = "_".join(group_name_parts)
    
    # The parent directory for all runs with these settings
    experiment_group_dir = Path(config.get('output_dir', 'outputs')) / experiment_group_name
    
    # 5. Call the main training function from the trainer module
    train(config, experiment_group_dir, args.resume_id)

if __name__ == "__main__":
    main()