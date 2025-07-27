# models/pretrained.py
# This module loads a pretrained model and modifies its final layer based on the task.

import torch.nn as nn

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
        config (dict): Must contain keys:
            - model_name (str): Name of the model (e.g., 'resnet18', 'vit_tiny_patch16_224').
            - task (str): One of {'classification', 'regression', 'feature_extraction'}
            - pretrained (bool): Whether to load pretrained weights
            - output_dim (int): Number of output dimensions (e.g., num_classes or 1)
            
    Returns:
        nn.Module: The initialized model with modified head.
    """
    model_name: str = config["model_name"]
    task: str = config['task']
    pretrained: bool = config.get("pretrained", True)
    output_dim: int = config.get("output_dim", 0)

    # Check validity of task
    VALID_TASKS = ['classification', 'regression', 'feature_extraction']
    if task not in VALID_TASKS:
        raise ValueError(f"[ERROR] Invalid task '{task}'. Must be one of {VALID_TASKS}.")

    # Ensure output_dim is set properly for classification or regression tasks
    if task in ['classification', 'regression'] and output_dim <= 0:
        raise ValueError("[ERROR] output_dim must be a positive integer for classification or regression tasks.")

    if tv_models is not None and hasattr(tv_models, model_name):
        model = getattr(tv_models, model_name)(pretrained=pretrained)
    elif timm is not None and timm.is_model(model_name):
        model = timm.create_model(model_name, pretrained=pretrained)
    else:
        raise ImportError("[ERROR] Neither torchvision nor timm is available for loading pretrained models.")

    # Remove head for feature extraction tasks
    if task == 'feature_extraction':
        _replace_output_layer(model, nn.Identity())
    else:
        in_features = _get_in_features(model)
        _replace_output_layer(model, nn.Linear(in_features, output_dim))

    return model


def _get_in_features(model: nn.Module) -> int:
    """
    Get the number of input features for the final layer of the model.

    Args:
        model (nn.Module): The model to inspect.
    
    Returns:
        int: Number of input features for the final layer.
    """
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model.fc.in_features
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        return model.classifier.in_features
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        return model.head.in_features
    elif hasattr(model, 'get_classifier'):  # for timm models
        if not callable(model.get_classifier):
            raise TypeError("[ERROR] get_classifier is not callable.")
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Linear):
            return classifier.in_features
    raise NotImplementedError("[ERROR] Cannot extract in_features: unknown model structure.")


def _replace_output_layer(model: nn.Module, new_layer: nn.Module):
    """
    Replace the final layer of the model with a new layer.

    Args:
        model (nn.Module): The model to modify.
        new_layer (nn.Linear): The new output layer to replace the existing one.
    """
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = new_layer
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = new_layer
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = new_layer
    elif hasattr(model, 'reset_classifier'):  # for timm models
        if not callable(model.reset_classifier):
            raise TypeError("[ERROR] reset_classifier is not callable.")
        model.reset_classifier(num_classes=new_layer.out_features)
    else:
        raise NotImplementedError("[ERROR] Cannot replace output layer: unknown model structure.")
