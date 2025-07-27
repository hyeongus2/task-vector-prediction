# datasets/regression/numeric/__init__.py

from .synthetic import SyntheticDataset
from .tabular_loader import (
    load_california_housing_dataset,
    load_diabetes_dataset,
    load_energy_efficiency_dataset,
    load_bike_sharing_dataset
)

def get_dataset(name: str, **kwargs):
    """
    Factory function to get a dataset by name.

    Args:
        name (str): Name of the dataset to load.
        **kwargs: Additional arguments for dataset initialization.

    Returns:
        Dataset: An instance of the requested dataset.
    """
    if name == 'synthetic':
        return SyntheticDataset(**kwargs)
    elif name == 'california_housing':
        return load_california_housing_dataset(**kwargs)
    elif name == 'diabetes':
        return load_diabetes_dataset(**kwargs)
    elif name == 'energy_efficiency':
        return load_energy_efficiency_dataset(**kwargs)
    elif name == 'bike_sharing':
        return load_bike_sharing_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")