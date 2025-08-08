# data/image.py

import torchvision.datasets as tv_datasets
from torch.utils.data import Dataset

def get_image_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """
    Load raw train/test image datasets based on config.
    This function does NOT apply any transforms. Transforms (e.g., Resize, ToTensor)
    should be set externally (e.g., train.py) based on model input shape.

    Args:
        config (dict): Configuration dictionary containing "data" key with "name".

    Returns:
        train_dataset (Dataset): Raw training dataset.
        test_dataset (Dataset): Raw test dataset.
    """
    dataset_name = config["data"]["name"]  # e.g., "CIFAR10", "MNIST"

    # Dynamically load dataset class from torchvision.datasets
    try:
        DatasetClass = getattr(tv_datasets, dataset_name)
    except AttributeError:
        raise ValueError(f"[ERROR] torchvision.datasets has no dataset named '{dataset_name}'")

    # Load raw datasets (transform=None)
    train_dataset = DatasetClass(root="./data", train=True, download=True, transform=None)
    test_dataset = DatasetClass(root="./data", train=False, download=True, transform=None)

    return train_dataset, test_dataset
