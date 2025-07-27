# models/__init__.py

from .mlp import build_mlp
# from .resnet import build_resnet18
# from .vit import build_vit_tiny
# from .lm import build_minilm

def build_model(config: dict):
    """
    Builds a model based on the provided configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
        
    Returns:
        nn.Module: The constructed model.
    """
    model_type = config.get('type', '').lower()
    
    if model_type == 'mlp':
        return build_mlp(config)
    elif model_type == 'resnet18':
        raise NotImplementedError("ResNet18 model building is not implemented yet.")
    #     return build_resnet18(config)
    elif model_type == 'vit_tiny':
        raise NotImplementedError("ViT Tiny model building is not implemented yet.")
    #     return build_vit_tiny(config)
    elif model_type == 'minilm':
        raise NotImplementedError("MiniLM model building is not implemented yet.")
    #     return build_minilm(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")