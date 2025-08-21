# src/tvp/data.py
import torchvision
from torch.utils.data import Dataset
from typing import Callable, Optional, Dict, List

__all__ = [
    "get_dataset",
    "VALID_DATASETS"
]

# A list of supported dataset class names from torchvision.datasets
VALID_DATASETS: List[str] = [
    "CIFAR10",
    "CIFAR100",
    "EMNIST",
    "EuroSAT",
    "FashionMNIST",
    "Flowers102",
    "Food101",
    "MNIST",
    "OxfordIIITPet",
    "SVHN"
]

def get_dataset(
    config: Dict,
    split: str = 'train',
    image_preprocess: Optional[Callable] = None
) -> Dataset:
    """
    Factory function for creating a torchvision dataset dynamically using getattr.
    Applies the provided preprocessing pipeline.
    """
    dataset_cfg = config["data"]
    dataset_name_from_config = dataset_cfg["name"]
    data_dir = dataset_cfg["path"]

    # Find the official, case-sensitive name for the dataset
    official_name = None
    for valid_name in VALID_DATASETS:
        if valid_name.lower() == dataset_name_from_config.lower():
            official_name = valid_name
            break
    
    if not official_name:
        raise ValueError(
            f"Unsupported or unknown dataset: '{dataset_name_from_config}'. "
            f"Supported datasets are: {VALID_DATASETS}"
        )

    try:
        # Dynamically get the dataset class from the torchvision.datasets module
        dataset_class = getattr(torchvision.datasets, official_name)
    except AttributeError:
        raise ImportError(f"Could not import '{official_name}' from torchvision.datasets.")

    # --- Handle different constructor arguments for datasets ---
    # Most datasets use `train=True/False`
    dataset_args = {
        "root": data_dir,
        "download": True,
        "transform": image_preprocess
    }
    
    # Some datasets like SVHN, OxfordIIITPet use `split='train'/'test'`
    if official_name in ["SVHN", "OxfordIIITPet", "Flowers102", "Food101", "EuroSAT"]:
        dataset_args["split"] = split
    else:
        dataset_args["train"] = (split == 'train')

    # Create an instance of the dataset class
    return dataset_class(**dataset_args)
