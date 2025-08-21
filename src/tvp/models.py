# src/tvp/models.py
import torch.nn as nn
import open_clip
from typing import Dict, Tuple, Callable

def build_clip_model(config: Dict) -> Tuple[nn.Module, nn.Module, Callable, Callable]:
    """
    Builds and returns the core CLIP components (image & text encoders)
    and their associated preprocessors based on the configuration.
    This is the single factory function for getting all necessary parts of a CLIP model.

    Returns:
        A tuple containing:
        - image_encoder (nn.Module): The visual part of the CLIP model.
        - text_encoder (nn.Module): The text part of the CLIP model.
        - tokenizer (Callable): The tokenizer for creating text prompts.
        - image_preprocess (Callable): The transform pipeline for processing images.
    """
    model_cfg = config["model"]
    model_name_lower = model_cfg.get("name", "vit-b-32").lower()
    
    all_models = open_clip.list_pretrained()

    # --- Find the official, correctly-cased model name (case-insensitive) ---
    official_name_gen = (name for name, _ in all_models if name.lower() == model_name_lower)
    # Get the first item from the generator, or None if it's empty.
    model_name = next(official_name_gen, None)

    if model_name is None:
        raise ValueError(f"Could not find any CLIP model matching the name '{model_cfg['name']}'.")

    # --- Validate the 'pretrained' tag for the selected model ---
    # Get all valid pretrained tags for this specific model architecture
    valid_pretrained_tags = [
        tag for name, tag in all_models if name == model_name
    ]
    
    # Get the requested pretrained tag from config or use a default
    pretrained = model_cfg.get("pretrained", "laion2b_s34b_b79k")
    
    if pretrained not in valid_pretrained_tags:
        raise ValueError(
            f"Invalid pretrained tag '{pretrained}' for model '{model_name}'.\n"
            f"Available options for this model are: {valid_pretrained_tags}"
        )
    
    print(f"Loading CLIP components: {model_name} pretrained on {pretrained}...")

    # --- Load model using the validated names ---
    model, image_preprocess, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        jit=False
    )

    # Extract the core components needed for the experiment
    image_encoder = model.visual
    text_encoder = model.text
    tokenizer = open_clip.get_tokenizer(model_name)

    print("CLIP image and text encoders loaded successfully.")

    return image_encoder, text_encoder, tokenizer, image_preprocess
