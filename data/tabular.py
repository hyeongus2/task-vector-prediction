# data/tabular.py

import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.utils._bunch import Bunch
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initializes the tabular dataset with features and targets.
        
        Args:
            X (np.ndarray): Feature matrix of shape (num_samples, num_features).
            y (np.ndarray): Target vector or matrix of shape (num_samples, num_outputs).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def load_california_housing_dataset():
    data = fetch_california_housing()
    return TabularDataset(data.data, data.target)

def load_diabetes_dataset():
    data = load_diabetes()
    return TabularDataset(data.data, data.target)

def load_boston_housing_dataset():
    """
    Loads the Boston Housing dataset from sklearn.
    
    Returns:
        TabularDataset: Dataset object containing features and targets.
    """
    data = fetch_openml(name='boston', version=1, as_frame=False)
    return TabularDataset(data.data, data.target)

def load_openml_dataset(name: str):
    """
    Loads a dataset from OpenML by name.
    
    Args:
        name (str): Name of the dataset to load.
        
    Returns:
        TabularDataset: Dataset object containing features and targets.
    """
    data = fetch_openml(name=name, as_frame=False)
    return TabularDataset(data.data, data.target)

def load_energy_efficiency_dataset():
    """
    Loads the Energy Efficiency dataset from OpenML.
    
    Returns:
        TabularDataset: Dataset object containing features and targets.
    """
    return load_openml_dataset('energy-efficiency')

def load_bike_sharing_dataset():
    """
    Loads the Bike Sharing dataset from OpenML.
    
    Returns:
        TabularDataset: Dataset object containing features and targets.
    """
    return load_openml_dataset('bike-sharing')