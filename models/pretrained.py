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
    VALID_TASKS = ['classification', 'regression', 'feature_extraction']
    if task not in VALID_TASKS:
        raise ValueError(f"[ERROR] Invalid task '{task}'. Must be one of {VALID_TASKS}.")

    # Ensure output_dim is set properly for classification or regression tasks
    if task in ['classification', 'regression'] and output_dim <= 0:
        raise ValueError("[ERROR] output_dim must be positive for classification/regression tasks.")

    # Load model
    if tv_models is not None and hasattr(tv_models, model_name):
        model = getattr(tv_models, model_name)(pretrained=pretrained)

        # torchvision models need explicit handling for output layer modification
        # Remove head for feature extraction tasks
        if task == 'feature_extraction':
            _replace_output_layer(model, nn.Identity())
        else:
            in_features = _get_in_features(model)
            _replace_output_layer(model, nn.Linear(in_features, output_dim))

    elif timm is not None and model_name in timm.list_models():
        num_classes = 0 if task == 'feature_extraction' else output_dim
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    else:
        raise ImportError("[ERROR] Model not found in torchvision or timm.")

    return model


def _get_in_features(model: nn.Module) -> int:
    """
    Get the number of input features for the final layer of the model.

    Args:
        model (nn.Module): The model to inspect.
    
    Returns:
        int: Number of input features for the final layer.
    """
    for attr_name in ['fc', 'classifier', 'head']:
        if hasattr(model, attr_name):
            layer = getattr(model, attr_name)
            if isinstance(layer, nn.Linear):
                return layer.in_features
            elif isinstance(layer, nn.Sequential):
                # If the layer is a Sequential, inspect its last module
                return _get_in_features(layer[-1])
            
    if hasattr(model, 'get_classifier') and callable(model.get_classifier):  # for timm models
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Linear):
            return classifier.in_features
        elif isinstance(classifier, nn.Sequential):
            return _get_in_features(classifier[-1])
        
    raise NotImplementedError("[ERROR] Cannot extract in_features: unsupported model structure.")


def _replace_output_layer(model: nn.Module, new_layer: nn.Module):
    """
    Replace the final layer of the model with a new layer.

    Args:
        model (nn.Module): The model to modify.
        new_layer (nn.Linear): The new output layer to replace the existing one.
    """
    for attr_name in ['fc', 'classifier', 'head']:
        if hasattr(model, attr_name) and isinstance(getattr(model, attr_name), (nn.Linear, nn.Sequential)):
            setattr(model, attr_name, new_layer)
            return

    if hasattr(model, 'reset_classifier') and callable(model.reset_classifier):  # for timm models
        if not hasattr(new_layer, 'out_features'):
            raise ValueError("[ERROR] new_layer must have 'out_features' attribute for reset_classifier.")
        model.reset_classifier(num_classes=new_layer.out_features)
        return
    
    raise NotImplementedError("[ERROR] Cannot replace output layer: unsupported model structure.")
