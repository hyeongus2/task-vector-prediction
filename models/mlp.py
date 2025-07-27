# models/mlp.py
# This module defines a Multi-Layer Perceptron (MLP) model with customizable
# architecture, activation functions, and dropout rates. It includes a function
# to build the MLP from a configuration dictionary, allowing for easy model
# instantiation and flexibility in design.

import torch.nn as nn

# Define the hidden dimensions & dropout for the MLP presets (base/large)
MLP_PRESETS = {
    "base": {"hidden_dims": [64, 64], "dropout": 0.2},
    "large": {"hidden_dims": [256, 256, 256], "dropout": 0.5}
}

def get_activation(name: str):
    if name == 'relu':
        return nn.ReLU
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'sigmoid':
        return nn.Sigmoid
    elif name == 'leaky_relu':
        return nn.LeakyReLU
    elif name == 'gelu':
        return nn.GELU
    else:
        raise ValueError(f"Unknown activation function: {name}")
    
class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.0):
        """
        Initializes a Multi-Layer Perceptron (MLP) model.
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list): List of integers representing the number of units in each hidden layer.
            output_dim (int): Dimension of the output features.
            activation (str): Activation function to use ('relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu').
            dropout (float): Dropout rate to apply after each layer.
        """
        assert isinstance(hidden_dims, list), "hidden_dims must be a list of integers"

        super().__init__()

        self.layers = nn.ModuleList()
        self.activation_cls = get_activation(activation)

        # Input & hidden layers
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))    # Linear layer
            self.layers.append(self.activation_cls())       # Activation function
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))     # Dropout layer
            prev_dim = dim

        # Output layer
        self.out_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)

def build_mlp(config: dict) -> MLP:
    """
    Builds an MLP model from a configuration dictionary.
    Args:
        config (dict): Configuration dictionary containing 'input_dim', 'output_dim', 'hidden_dims', 'activation', and 'dropout'.
    Returns:
        MLP: An instance of the MLP model.
    """
    variant = config.get('variant', 'custom')

    # If a preset variant is specified, use its parameters
    if variant in MLP_PRESETS:
        preset = MLP_PRESETS[variant]
        config['hidden_dims'] = preset['hidden_dims']
        config['dropout'] = preset['dropout']

    return MLP(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        activation=config.get('activation', 'relu'),
        dropout=config.get('dropout', 0.0)
    )