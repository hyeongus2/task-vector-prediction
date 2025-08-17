# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models import build_model, get_input_size
from data import get_datasets
from trainers import get_trainer
from utils.config_utils import parse_args, load_config, merge_config
from utils.paths import get_save_path
from utils.seed import set_seed
from utils.logger import init_logger

def main():
    # 1. Load config and overrides
    config_path, override_dict = parse_args()
    config = load_config(config_path)
    config = merge_config(config, override_dict)

    # Add save path to config
    save_path = config["save"].get("path", get_save_path(config_path=config_path, overrides=override_dict))
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
    logger.finish_wandb()


if __name__ == "__main__":
    main()
