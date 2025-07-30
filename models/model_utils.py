# models/model_utils.py

import torch.nn as nn

__all__ = [
    "get_head",
    "reset_head",
    "get_head_in_features",
    "get_lora_target_modules"
]

HEAD_NAME = {'fc', 'classifier', 'head', 'dense'}


def get_head(model: nn.Module):
    for attr in HEAD_NAME:
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, 'get_classifier') and callable(model.get_classifier):
        return model.get_classifier()
    raise NotImplementedError("[ERROR] Could not find model head.")


def get_head_in_features(model: nn.Module) -> int:
    head = get_head(model)
    if isinstance(head, nn.Linear):
        return head.in_features
    elif isinstance(head, nn.Sequential):
        last = head[-1]
        return last.in_features if isinstance(last, nn.Linear) else get_head_in_features(last)
    raise NotImplementedError("[ERROR] Unsupported head type for in_features.")


def reset_head(model: nn.Module, new_head: nn.Module):
    for attr in HEAD_NAME:
        if hasattr(model, attr) and isinstance(getattr(model, attr), (nn.Linear, nn.Sequential)):
            setattr(model, attr, new_head)
            return
    if hasattr(model, 'reset_classifier') and callable(model.reset_classifier):
        if not hasattr(new_head, 'out_features'):
            raise ValueError("[ERROR] new_head must have out_features for reset_classifier.")
        model.reset_classifier(num_classes=new_head.out_features)
        return
    raise NotImplementedError("[ERROR] Could not reset model head.")


def get_lora_target_modules(model: nn.Module) -> list[str]:
    """
    Extracts potential target modules for applying LoRA.
    Prioritizes common attention/feedforward components or final layers.
    """
    target_modules = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # (e.g., 'encoder.layer.0.attention.self.query' → 'query')
            last_name = name.split('.')[-1]
            if last_name in {'q', 'k', 'v', 'query', 'key', 'value'}.union(HEAD_NAME):
                target_modules.append(last_name)

    # Deduplication
    return list(set(target_modules)) or ["fc"]  # fallback
