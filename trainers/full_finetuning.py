# trainers/full_finetuning.py

import torch
import torch.nn as nn
import torch.optim as optim
from .base_trainer import BaseTrainer


class FullFinetuneTrainer(BaseTrainer):
    def train(self):
        epochs = self.config["train"]["epochs"]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["train"]["lr"])

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.logger.info(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        acc = correct / total
        self.logger.info(f"[Evaluation] Accuracy: {acc:.4f}")
