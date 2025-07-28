# data/__init__.py

from .vision import get_vision_datasets
from torch.utils.data import Dataset
from typing import Tuple

def get_datasets(config: dict) -> Tuple[Dataset, Dataset]:
    """
    Load raw (train, test) datasets based on the model type and configuration.
    This function does NOT return DataLoaders, and does NOT apply transforms.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        Tuple[Dataset, Dataset]: train_dataset and test_dataset (raw, untransformed)
    """
    VALID_MODEL_TYPES = ['vision', 'text', 'tabular', 'synthetic']

    model_type = config['model']['type'].lower()

    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"[ERROR] Unsupported model type '{model_type}'. Supported types are: {VALID_MODEL_TYPES}")

    if model_type == 'vision':
        return get_vision_datasets(config)
    else:
        raise NotImplementedError(f"[ERROR] get_dataset for model type '{model_type}' is not yet implemented.")