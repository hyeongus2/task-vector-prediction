# models/__init__.py

from .utils import *
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
