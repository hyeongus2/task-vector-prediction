# src/tvp/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Callable
from torch.utils.data import DataLoader

class CLIPFineTuner:
    """
    A trainer specifically for fine-tuning the image encoder of a CLIP model
    using a contrastive loss against frozen text features.
    """
    def __init__(self, config: Dict, image_encoder: nn.Module, text_encoder: nn.Module,
                 tokenizer: Callable, train_loader: DataLoader, val_loader: DataLoader, logger: object):
        
        self.config = config
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_encoder.to(self.device)
        self.text_encoder.to(self.device)

        # Create an optimizer that only targets the image_encoder's parameters.
        optim_cfg = config["optim"]
        self.optimizer = torch.optim.AdamW(
            self.image_encoder.parameters(),
            lr=optim_cfg.get("lr", optim_cfg.get("lr_backbone", 1e-5)), # Use a general lr or a specific one
            weight_decay=optim_cfg.get("weight_decay", 0.1)
        )
        
        # This is a learnable parameter from the original CLIP model, crucial for scaling logits.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale.to(self.device)

    def _calculate_loss(self, image_features: torch.Tensor, text_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the contrastive loss.
        """
        # Normalize features for cosine similarity calculation
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Text features are assumed to be pre-normalized.

        # Calculate logits as scaled cosine similarities
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        
        # Calculate cross-entropy loss between image-text similarity scores and the true labels
        return F.cross_entropy(logits, labels)
        
    def train_one_epoch(self, epoch: int):
        """
        The main training loop for one epoch.
        """
        self.image_encoder.train() # Set the image encoder to training mode
        
        # Pre-compute text features for all classes once per epoch (as they are frozen)
        class_names = self.train_loader.dataset.classes
        text_prompts = [f"a photo of a {name}" for name in class_names]
        text_tokens = self.tokenizer(text_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.text_encoder(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # --- Forward pass ---
            image_features = self.image_encoder(images)
            loss = self._calculate_loss(image_features, text_features, labels)

            # --- Backward pass and optimization ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch+1} average training loss: {avg_loss:.4f}")

        # Here you would typically run validation and save checkpoints/tau vectors.
        # self.validate(epoch)

    def run_training(self):
        """
        Starts and manages the overall training process.
        """
        num_epochs = self.config["training"]["epochs"]
        self.logger.info(f"Starting training for {num_epochs} epochs on device '{self.device}'...")
        
        # Here you would typically load the pretrained weights (theta_0) before training
        # and save the initial task vector (tau_0, which is all zeros).

        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            # After each epoch, calculate and save the new tau vector
            # e.g., save_tau(get_tau(...), epoch=epoch, ...)
        
        self.logger.info("Training complete.")
