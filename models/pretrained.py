# models/pretrained.py

import torch.nn as nn
from .model_utils import get_head_in_features, reset_head

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None

try:
    import timm
except ImportError:
    timm = None

def load_pretrained_model(config: dict) -> nn.Module:
    """
    Load a pretrained model based on the provided config dictionary.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        config['model']['name'] (str): Name of the model to load.
        config['data']['task'] (str): Task type ('classification', 'regression', 'feature_extraction').
        config['model'].get('pretrained', True) (bool): Whether to load pretrained weights.
        config['data'].get('output_dim', 0) (int): Number of output classes for classification/regression tasks.
            
    Returns:
        nn.Module: The initialized model with modified head.
    """
    model_name: str = config['model']['name'].lower()
    task: str = config['data']['task'].lower()
    pretrained: bool = config['model'].get("pretrained", True)
    output_dim: int = config['data'].get("output_dim", 0)

    # Check validity of task
    VALID_TASKS = {'classification', 'regression', 'feature_extraction'}
    if task not in VALID_TASKS:
        raise ValueError(f"[ERROR] Invalid task '{task}'. Must be one of {VALID_TASKS}.")

    # Ensure output_dim is set properly for classification or regression tasks
    if task in ['classification', 'regression'] and output_dim <= 0:
        raise ValueError("[ERROR] output_dim must be positive for classification/regression tasks.")

    # Load torchvision model
    if tv_models is not None and hasattr(tv_models, model_name):
        model = getattr(tv_models, model_name)(pretrained=pretrained)

        # torchvision models need explicit handling for output layer modification
        # Remove head for feature extraction tasks
        if task == 'feature_extraction':
            reset_head(model, nn.Identity())
        else:
            in_features = get_head_in_features(model)
            reset_head(model, nn.Linear(in_features, output_dim))

    # Load timm model
    elif timm is not None and model_name in timm.list_models():
        num_classes = 0 if task == 'feature_extraction' else output_dim
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    else:
        raise ImportError("[ERROR] Model not found in torchvision or timm.")

    return model
