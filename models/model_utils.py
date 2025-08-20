# models/model_utils.py

import re
from functools import lru_cache
import torch.nn as nn

__all__ = [
    "get_head",
    "reset_head",
    "get_head_in_features",
    "get_lora_target_modules",
    "is_head_param",
    "HEAD_TOKENS",
]

# Head tokens commonly used across libraries (timm/torchvision/etc.)
HEAD_TOKENS: set[str] = {"fc", "classifier", "head", "cls_head", "dense"}

@lru_cache(maxsize=8)
def _compile_head_regex(tokens_tuple: tuple[str, ...]) -> re.Pattern:
    """
    Build a segment-exact regex from tokens:
      matches if ANY path segment equals one of the tokens.
    Example match: "head.fc.weight", "classifier.bias", "model.cls_head.weight"
    """
    escaped = [re.escape(t) for t in tokens_tuple]
    inner = "|".join(escaped) if escaped else r"$^"  # match nothing if empty
    pattern = rf"(?:^|\.)(?:{inner})(?:\.|$)"
    return re.compile(pattern)

def _head_regex() -> re.Pattern:
    """
    Get a compiled regex reflecting the current HEAD_TOKENS.
    Because we pass a tuple of the *current* set, any mutation to HEAD_TOKENS
    will produce a different tuple → a new cached regex is compiled automatically.
    """
    return _compile_head_regex(tuple(sorted(HEAD_TOKENS)))

def is_head_param(name: str) -> bool:
    """
    Return True if a parameter name likely belongs to the classifier head.
    Fast path: top-level prefix check (e.g., 'fc.weight').
    Fallback: segment-exact regex match anywhere in the path.
    """
    # First, check for simple prefixes for performance (very common case)
    n = name.lower()
    prefix = n.split(".", 1)[0]
    if prefix in HEAD_TOKENS:
        return True
    # If not a simple prefix, use the more robust regex to find head tokens in any segment
    return bool(_head_regex().search(n))

def get_head(model: nn.Module) -> nn.Module:
    """
    Return the head module using common attribute names or model API.
    """
    for attr in HEAD_TOKENS:
        if hasattr(model, attr):
            return getattr(model, attr)
    # Handle timm-style models that have a get_classifier() method
    if hasattr(model, "get_classifier") and callable(getattr(model, "get_classifier")):
        return model.get_classifier()
    raise NotImplementedError("Could not find model head.")

def get_head_in_features(model: nn.Module) -> int:
    """
    Return input feature size of the head (Linear last layer).
    """
    head = get_head(model)
    if isinstance(head, nn.Linear):
        return head.in_features
    # Handle heads that are Sequential modules, finding the last Linear layer
    elif isinstance(head, nn.Sequential):
        for layer in reversed(head):
            if isinstance(layer, nn.Linear):
                return layer.in_features
    raise NotImplementedError("Unsupported head type for in_features.")

def reset_head(model: nn.Module, new_head: nn.Module):
    """
    Replace the classifier head with new_head, or use model.reset_classifier if available.
    """
    # First, try to find and replace the head attribute directly
    for attr in HEAD_TOKENS:
        if hasattr(model, attr) and isinstance(getattr(model, attr), (nn.Linear, nn.Sequential)):
            setattr(model, attr, new_head)
            return
    # If that fails, try the timm-style reset_classifier method
    if hasattr(model, "reset_classifier") and callable(model.reset_classifier):
        if not hasattr(new_head, "out_features"):
            raise ValueError("new_head must have out_features for reset_classifier.")
        # Assumes the new_head is a simple Linear layer or has a compatible structure
        model.reset_classifier(num_classes=new_head.out_features)
        return
    raise NotImplementedError("Could not reset model head.")

def get_lora_target_modules(
    model: nn.Module, 
    include_k: bool = False,
    include_ffn: bool = False
) -> list[str]:
    """
    Heuristically collect submodule LAST names suitable for LoRA injection.
    Defaults to Q/V only; optionally include K. Also cover common fused/projection names.
    Skips any module that belongs to the classifier head.
    """
    attn_names = {"q", "v", "qkv", "query", "value", "proj", "out_proj"}
    if include_k:
        attn_names.add("k")
        attn_names.add("key")

    target_names = attn_names.copy()
    if include_ffn:
        target_names.update({"fc1", "fc2", "up_proj", "down_proj"})

    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if is_head_param(name):   # skip head params entirely
            continue

        last = name.split(".")[-1]
        if last in target_names:
            targets.add(last)

    if not targets:
        # If no specific targets found, return a sensible default for attention layers
        default_targets = ["q", "v"]
        if include_k:
            default_targets.append("k")
        return default_targets
        
    return sorted(list(targets))
