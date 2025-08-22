# src/tvp/trainers.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
from typing import Dict, Any, Tuple

from src.tvp.model import build_clip_model
from src.tvp.data import DataModule
from src.tvp.utils import get_tau, save_tau # We assume these are in utils.py

class CLIPFineTuner:
    """
    Orchestrates the fine-tuning process for the CLIP image encoder.
    Handles training, validation, checkpointing, and task vector generation.
    """
    def __init__(self, config: Dict, logger: Any):
        self.config = config
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Build model components and data module
        self.image_encoder, self.text_encoder, self.tokenizer, preprocess = build_clip_model(config)
        self.data_module = DataModule(config, preprocess=preprocess)

        self.image_encoder.to(self.device)
        self.text_encoder.to(self.device)

        # 2. Store a deep copy of the pretrained weights (theta_0) before training begins
        self.theta_0_sd = copy.deepcopy(self.image_encoder.state_dict())

        # 3. Setup optimizer to target ONLY the image_encoder's parameters
        optim_cfg = config["optim"]
        self.optimizer = torch.optim.AdamW(
            self.image_encoder.parameters(),
            lr=optim_cfg.get("lr", 1e-5),
            weight_decay=optim_cfg.get("weight_decay", 0.1)
        )
        
        # 4. Prepare text features and logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale.to(self.device)
        self.text_features = self._precompute_text_features()

    def _precompute_text_features(self) -> torch.Tensor:
        """Computes and normalizes text features for all classes once."""
        class_names = self.data_module.classnames
        text_prompts = [f"a photo of a {name}" for name in class_names]
        text_tokens = self.tokenizer(text_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.text_encoder(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _calculate_loss(self, image_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates the contrastive loss against the pre-computed text features."""
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ self.text_features.t()
        return F.cross_entropy(logits, labels)

    def train_one_epoch(self, epoch: int) -> float:
        """Runs a single training epoch."""
        self.image_encoder.train()
        total_loss = 0
        pbar = tqdm(self.data_module.train_loader, desc=f"Epoch {epoch+1} Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            image_features = self.image_encoder(images)
            loss = self._calculate_loss(image_features, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.data_module.train_loader)
        self.logger.info(f"Epoch {epoch+1} Avg Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Runs a single validation epoch."""
        self.image_encoder.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        pbar = tqdm(self.data_module.test_loader, desc=f"Epoch {epoch+1} Validation")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            image_features = self.image_encoder(images)
            loss = self._calculate_loss(image_features, labels)
            total_loss += loss.item() * images.size(0)

            logits = self.logit_scale.exp() * image_features @ self.text_features.t()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self.logger.info(f"Epoch {epoch+1} Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")
        return avg_loss, accuracy

    def run_training(self):
        """Starts and manages the overall training process."""
        num_epochs = self.config["training"]["epochs"]
        out_dir = self.config["save"]["path"]
        best_val_acc = -1.0
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # --- Tau Vector Calculation and Saving ---
            tau, meta = get_tau(self.image_encoder, self.theta_0_sd, exclude_head=False)
            save_tau(tau, meta, epoch=epoch, mode="epoch", out_dir=out_dir)
            self.logger.info(f"Saved tau_epoch_{epoch+1:03d}.pt")

            # --- Checkpoint Saving ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.logger.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving best model.")
                # Save the best model checkpoint
                torch.save(self.image_encoder.state_dict(), os.path.join(out_dir, "best_model.pt"))
                # Save the corresponding tau*
                save_tau(tau, meta, mode="star", out_dir=out_dir)

        self.logger.info("Training complete.")
