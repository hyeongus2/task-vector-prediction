# datasets/regression/numeric/synthetic.py
# This module generates synthetic data for numeric regression tasks.

import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self,
                 num_samples: int = 1000,
                 input_dim: int = 10,
                 output_dim: int = 1,
                 function: str = 'linear',
                 noise_std: float = 0.1,
                 seed: int = 42):
        """
        Initializes the synthetic dataset for scalar or vector regression.

        Args:
            num_samples (int): Number of samples in the dataset.
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output targets.
            function (str): Type of function to generate targets ('linear', 'quadratic', 'sin', etc.).
            noise_std (float): Standard deviation of the Gaussian noise added to the targets.
            seed (int): Random seed for reproducibility.

        y = f(x) + bias + noise, where f is a simple function (e.g., linear, quadratic, sin, etc.)
        """
        super().__init__()
        self.num_samples = num_samples

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate random inputs
        self.X = torch.randn(num_samples, input_dim)

        # Generate random weights and biases
        self.W = torch.randn(input_dim, output_dim)
        self.bias = torch.randn(1, output_dim)

        # Generate outputs based on function type
        if function == 'linear':
            self.y = self.X @ self.W
        elif function == 'quadratic':
            self.y = (self.X ** 2) @ self.W
        elif function == 'sin':
            self.y = torch.sin(self.X @ self.W)
        else:
            raise ValueError(f"Unsupported function type: {function}")
        
        # Add bias and Gaussian noise to the outputs
        self.y += self.bias + noise_std * torch.randn_like(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
