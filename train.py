# train.py

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from ast import literal_eval

from models import build_model
from data import get_datasets
from trainers import get_trainer
from utils.paths import get_save_path
from utils.seed import set_seed
from utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--overrides', nargs='*', help='Override config entries. Format: key=value')
    args = parser.parse_args()

    override_dict = {}
    if args.overrides:
        for override in args.overrides:
            key, value = override.split('=', 1)
            try:
                # Attempt to evaluate the value as a Python literal (e.g., int, float, list, dict)
                _value = literal_eval(value)
            except:
                # If eval fails, keep it as a string
                _value = value
            override_dict[key] = _value

    return args.config, override_dict


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def merge_config(config: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        keys = k.split('.')
        d = config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return config


def get_input_size(model: nn.Module, config: dict) -> tuple:
    """
    Returns (Channels, Height, Width) of input that the model expects.
    Works well for image-based models (e.g., CNN, ViT).
    """
    model_type = config['model']['type']    # One of {'image', 'text', 'tabular', 'synthetic'}

    if model_type == 'image':
        # Use timm-style config is available
        if hasattr(model, 'default_cfg') and 'input_size' in model.default_cfg:
            return model.default_cfg['input_size']  # e.g., (3, 224, 224)
    
        # Check first top-level child module
        first_layer = next(model.children())
        if isinstance(first_layer, nn.Conv2d):
            return (first_layer.in_channels, 224, 224)  # Default to 224x224 for Conv2d layers
        elif isinstance(first_layer, nn.Linear):
            return (1, 1, first_layer.in_features)  # Linear layers expect 1D input

        # Fallback
        return (3, 224, 224)
    

    elif model_type == 'text':
        return (512,)  # Default for text models (e.g., BERT, GPT) / max sequence length
    

    elif model_type == 'tabular':
        first_layer = next(model.children())
        if isinstance(first_layer, nn.Linear):
            return (first_layer.in_features,)
        elif isinstance(first_layer, nn.Sequential):
            first_layer = next(first_layer.children())
            if isinstance(first_layer, nn.Linear):
                return (first_layer.in_features,)
        else:
            raise ValueError("Unsupported first layer type for tabular model")
        
    
    elif model_type == 'synthetic':
        return (1,)
    

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return (0,)


def main():
    # 1. Load config and overrides
    config_path, override_dict = parse_args()
    config = load_config(config_path)
    config = merge_config(config, override_dict)

    save_path = get_save_path(config_path=config_path, overrides=override_dict)
    config["save"]["path"] = save_path

    # 2. Set seed
    set_seed(config.get("seed", 42))

    # 3. Initialize logger (wandb + file + console)
    logger = init_logger(config)

    # 4. Build model
    logger.info("[1/6] Building model...")
    model = build_model(config)

    # 5. Get raw datasets
    logger.info("[2/6] Preparing dataset...")
    train_dataset, test_dataset = get_datasets(config)

    # 6. Build input transform or processor
    logger.info("[3/6] Configuring transforms...")
    model_type = config["model"]["type"]
    input_size = get_input_size(model, config)

    if model_type == "image":
        in_channels = input_size[0]
        resize = input_size[1:]

        transform_list = []
        transform_list.append(transforms.Resize(resize))
        if hasattr(train_dataset[0][0], 'mode') and train_dataset[0][0].mode == 'L' and in_channels == 3:
            transform_list.append(transforms.Grayscale(3))
        transform_list.append(transforms.ToTensor())

        transform = transforms.Compose(transform_list)
        train_dataset.transform = transform
        test_dataset.transform = transform

    elif model_type == "text":
        # Assuming the dataset is already tokenized by get_datasets
        logger.info("Text dataset assumed pre-tokenized.")
        pass

    elif model_type == "tabular":
        # Assuming the dataset is already scaled (e.g., StandardScaler) by get_datasets
        logger.info("Tabular dataset assumed pre-scaled.")
        pass

    elif model_type == "synthetic":
        # No specific preprocessing needed for synthetic datasets
        logger.info("Synthetic dataset assumed to require no transform.")
        pass

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 7. Create DataLoaders
    logger.info("[4/6] Creating dataloaders...")
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    # 8. Create trainer
    logger.info("[5/6] Creating trainer...")
    trainer = get_trainer(config, model, train_loader, test_loader, logger)

    # 9. Train
    logger.info("[6/6] Starting training loop...")
    trainer.train()


if __name__ == "__main__":
    main()
