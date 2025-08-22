# src/tvp/data.py

import torchvision
from torch.utils.data import Dataset, random_split
from typing import Callable, Optional, Dict, List

__all__ = [
    "get_dataset",
    "VALID_DATASETS"
]

VALID_DATASETS: List[str] = [
    "CIFAR10", "CIFAR100", "EMNIST", "EuroSAT", "FashionMNIST", 
    "Flowers102", "Food101", "MNIST", "OxfordIIITPet", "SVHN"
]

def get_dataset(
    config: Dict,
    split: str = 'train',
    image_preprocess: Optional[Callable] = None
) -> Dataset:
    """
    Factory function for creating a torchvision dataset dynamically using getattr.
    Handles different constructor arguments and manual splitting for EuroSAT.
    """
    dataset_cfg = config["data"]
    dataset_name_lower = dataset_cfg["name"].lower()
    data_dir = dataset_cfg.get("path", "data")

    dataset_name = None
    for valid_name in VALID_DATASETS:
        if valid_name.lower() == dataset_name_lower:
            dataset_name = valid_name
            break
    
    if not dataset_name:
        raise ValueError(f"Unsupported dataset: '{dataset_cfg['name']}'.")

    try:
        dataset_class = getattr(torchvision.datasets, dataset_name)
    except AttributeError:
        raise ImportError(f"Could not import '{dataset_name}' from torchvision.datasets.")

    # --- Handle different constructor arguments ---
    dataset_args = {"root": data_dir, "download": True, "transform": image_preprocess}
    
    # Special case: EuroSAT needs manual splitting
    if dataset_name == "EuroSAT":
        full_dataset = dataset_class(**dataset_args)
        # Define split ratio, e.g., 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])
        return train_set if split == 'train' else val_set

    # Datasets using the `split` argument
    if dataset_name in ["SVHN", "OxfordIIITPet", "Flowers102", "Food101", "EMNIST"]:
        # EMNIST requires a specific split type from config
        if dataset_name == "EMNIST":
            dataset_args["split"] = dataset_cfg.get("emnist_split", "byclass")
        else:
            dataset_args["split"] = split if split != 'val' else 'train' # Defaulting val to train for simplicity
    
    # Datasets using the `train` argument (default case)
    else:
        dataset_args["train"] = (split == 'train')

    return dataset_class(**dataset_args)
