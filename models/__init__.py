# models/__init__.py

from .model_utils import *
from .mlp import build_mlp
from .pretrained import load_pretrained_model
import torch.nn as nn

def build_model(config: dict) -> nn.Module:
    """
    Builds a model based on the provided configuration dictionary.

    Args:
        config (dict): Must contain:
            - config["model"]["name"] (str): Backbone model name for pretrained models (e.g., "mlp", "resnet18", "vit_base_patch16_224", etc.)

    Returns:
        nn.Module: Constructed and initialized model
    """
    model_name = config["model"]["name"].lower()

    if model_name == "mlp":
        return build_mlp(config)

    # Assume all non-MLP types are pretrained models (vision/text)
    return load_pretrained_model(config)


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
