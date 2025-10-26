# src/tvp/model.py
import logging
import torch
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig
from typing import List, Dict, Tuple
from .utils import set_seed

# Create a logger for this module
logger = logging.getLogger(__name__)


def create_model(config: Dict, class_names: List[str], processor: CLIPProcessor, device: torch.device) -> Tuple[CLIPModel, torch.Tensor]:
    """
    Creates the initial model (theta0) for the experiment. It loads a pre-trained
    CLIP model, applies the specified finetuning strategy (LoRA or full), 
    and separately returns the model and the text features that act as the classifier.

    Args:
        config (Dict): The configuration dictionary.
        class_names (List[str]): A list of class names for the dataset.
        processor (CLIPProcessor): The CLIP processor for tokenizing text.
        device (torch.device): The device to move the model to.

    Returns:
        Tuple[CLIPModel, torch.Tensor]: A tuple containing:
            - model (CLIPModel): The initialized CLIP model (theta0).
            - text_features (torch.Tensor): The normalized text features for classification.
    """
    set_seed(config.get('seed', 42))
    
    model_id = config['model_id']
    finetune_config = config['finetuning']
    
    logger.info(f"Loading pre-trained base CLIP model: {model_id}")
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True)

    # --- Apply finetuning strategy ---
    finetune_method = finetune_config.get('method', 'full')
    logger.info(f"Applying finetuning method: {finetune_method}")

    # In all cases, we freeze the text encoder as it acts as our semantic anchor.
    logger.info("Freezing the text encoder.")
    for param in model.parameters():
        param.requires_grad = False

    if finetune_method == 'lora':
        # For LoRA, we freeze the original vision model weights and only train
        # the added LoRA adapters and the final projection layer.            
        lora_config_dict = finetune_config['lora']
        lora_config = LoraConfig(**lora_config_dict)
        
        # Apply LoRA to the vision_model's transformer blocks
        model.vision_model = get_peft_model(model.vision_model, lora_config)

    elif finetune_method == 'full':
        # For full finetuning, the vision model is trainable by default.
        for param in model.vision_model.parameters():
            param.requires_grad = True
        logger.info("Vision model is set for full finetuning.")
    else:
        raise ValueError(f"Unsupported finetuning method: {finetune_method}")

    # --- Log trainable parameters ---
    logger.info("Trainable parameters summary:")
    # Use a try-except block to handle both PEFT and regular models gracefully
    try:
        # The print_trainable_parameters method exists on the PEFT-wrapped part of the model
        model.vision_model.print_trainable_parameters()
    except AttributeError:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"total params: {total_params:,}")
        logger.info(f"trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # --- Move the entire model to the specified device BEFORE using it ---
    model.to(device)

    # --- Create text prompts and generate text features (the "virtual" classifier head) ---
    logger.info("Creating text-based classifier weights.")
    prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        # Now both the model and the text_inputs are on the correct device
        text_features = model.get_text_features(**text_inputs.to(device))
    
    # Normalize the features, which will be our classifier weights
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return model, text_features