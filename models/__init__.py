# models/__init__.py

from .mlp import build_mlp
from .pretrained import load_pretrained_model
import torch.nn as nn

def build_model(config: dict) -> nn.Module:
    """
    Builds a model based on the provided configuration dictionary.

    Args:
        config (dict): Must contain:
            - config["model"]["type"] (str): Type of model ('mlp', 'resnet18', 'vit_base_patch16_224', etc.)
            - config["model"]["name"] (str): Backbone model name for pretrained models
            - config["data"]["task"] (str): Task type (classification, regression, feature_extraction)

    Returns:
        nn.Module: Constructed and initialized model
    """
    model_type = config["model"].get("type", "").lower()

    if not model_type:
        raise ValueError("[ERROR] Model type must be specified under config['model']['type'].")

    if model_type == "mlp":
        return build_mlp(config)

    # Assume all non-MLP types are pretrained models (vision/text)
    return load_pretrained_model(config)
