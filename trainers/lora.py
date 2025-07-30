# trainers/lora.py

import torch
import torch.nn as nn
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from trainers.base_trainer import BaseTrainer

class LoRATrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader, logger):
        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION if config["data"]["task"] == "feature_extraction" else TaskType.REGRESSION,
            inference_mode=False,
            r=config["lora"].get("r", 8),
            lora_alpha=config["lora"].get("alpha", 16),
            lora_dropout=config["lora"].get("dropout", 0.1),
            bias="none",
            target_modules=config["lora"].get("target_modules", ["q", "v", "fc"])
        )
        model = get_peft_model(model, peft_config)

        super().__init__(config, model, train_loader, test_loader, logger)

        self.logger.info("LoRA applied. Trainable params:")
        self.model.print_trainable_parameters()


    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            # Early stopping check
            if avg_loss + self.delta < self.best_loss:
                self.best_loss = avg_loss
                self.no_improve = 0
                self.save_model()
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break


    def eval(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        self.logger.info(f"[Evaluation] Test Loss: {avg_loss:.4f}")
