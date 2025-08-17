# trainers/base_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from collections import OrderedDict
from typing import Optional

from utils.seed import set_seed
from utils.tau_utils import get_tau, save_tau

class BaseTrainer():
    def __init__(self, config: dict, model: nn.Module, train_loader, test_loader, logger):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pretrained_state: OrderedDict = copy.deepcopy(OrderedDict(model.state_dict()))

        self.criterion, self.optimizer = self.get_criterion_and_optimizer()
        self.epochs = config["finetuning"].get("epochs", 10)

        # Early stopping params
        self.early_stop_enabled = config.get("early_stop", {}).get("enabled", False)
        self.patience = config.get("early_stop", {}).get("patience", 3)
        self.delta = config.get("early_stop", {}).get("delta", 1e-4)
        self.best_loss = float('inf')
        self.no_improve = 0

        # Save
        self.save_enabled = config["save"]["enabled"]
        self.save_every = config["save"].get("every", 10)
        self.save_max = self.save_every * config["save"]["max"] if config["save"].get("max", 0) else float('inf')
        self.save_path = config["save"]["path"]

        # Set seed
        set_seed(config.get("seed", 42))


    def get_criterion_and_optimizer(self):
        task = self.config["data"]["task"]
        opt_name = self.config["finetuning"].get("optimizer", "adam")
        lr = self.config["finetuning"].get("lr", 1e-4)
        momentum = self.config["finetuning"].get("momentum", 0.9)

        criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        if opt_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_name}")

        return criterion, optimizer


    def train(self):
        tau, _ = get_tau(self.model, self.pretrained_state)
        self.logger.log_wandb(tau=torch.zeros_like(tau), step=0, mode="step", path=None)
        self.logger.log_wandb(tau=torch.zeros_like(tau), step=0, mode="epoch", path=None)
        step = 1

        last_epoch = -1

        for epoch in range(self.epochs):
            self.model.train()
            correct = 0
            n = 0
            total_loss = 0.0

            # tqdm wrapper for step progress bar
            pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.epochs}] Train", leave=False)

            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    pred = pred.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    n += x.size(0)

                # Save tau_t for some steps
                if self.save_enabled and step < self.save_max and step % self.save_every == 0:
                    tau, meta = get_tau(self.model, self.pretrained_state)
                    save_tau(tau, meta, step=step, mode="step", out_dir=self.save_path)
                    self.logger.log_wandb(tau=tau, step=step, mode="step", path=None)
                step += 1

                # update tqdm progress bar with loss
                pbar.set_postfix(loss=loss.item())

            train_acc = None
            train_loss = total_loss / len(self.train_loader)

            if isinstance(self.criterion, nn.CrossEntropyLoss) and n > 0:
                train_acc = correct / n
                self.logger.info(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}")
            self.logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")

            tau, meta = get_tau(self.model, self.pretrained_state)

            # Save tau_t for every epoch
            if self.save_enabled:
                save_tau(tau, meta, epoch=epoch, mode="epoch", out_dir=self.save_path)

            # Validation after every epoch
            val_acc, val_loss = self.eval(epoch)

            self.logger.log_wandb(tau=tau, step=step, mode="epoch", path=None)
            if train_acc is not None and val_acc is not None:
                self.logger.log_wandb_scalar({
                    "acc/train": train_acc,
                    "loss/train": train_loss,
                    "acc/val": val_acc,
                    "loss/val": val_loss,
                }, step=step)
            else:
                self.logger.log_wandb_scalar({
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                }, step=step)

            # Early stopping check (based on val_loss)
            if val_loss + self.delta < self.best_loss:
                self.best_loss = val_loss
                self.no_improve = 0
                if self.save_enabled:
                    self.save_model(filename="best_model.pt")
                    save_tau(tau, meta, epoch=epoch, mode="star", out_dir=self.save_path)
                    self.logger.info(f"Saved best model and corresponding tau_star at epoch {epoch+1}")
            else:
                self.no_improve += 1
                if self.early_stop_enabled and self.no_improve >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    last_epoch = epoch
                    break

            last_epoch = epoch

        if self.save_enabled:
            self.save_model(filename=f"last_model_epoch_{last_epoch+1:03d}.pt")


    def eval(self, epoch=None) -> tuple[Optional[float], float]:
        self.model.eval()
        correct = 0
        n = 0
        total_loss = 0.0

        pbar = tqdm(self.test_loader, desc="[Evaluation]", leave=False)
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    pred = pred.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    n += x.size(0)

                pbar.set_postfix(loss=loss.item())

        acc = None
        avg_loss = total_loss / len(self.test_loader)

        if epoch is not None:
            if isinstance(self.criterion, nn.CrossEntropyLoss) and n > 0:
                acc = correct / n
                self.logger.info(f"[Epoch {epoch+1}] Validation Acc: {acc:.4f}")
            self.logger.info(f"[Epoch {epoch+1}] Validation Loss: {avg_loss:.4f}")
        else:
            if isinstance(self.criterion, nn.CrossEntropyLoss) and n > 0:
                acc = correct / n
                self.logger.info(f"[Evaluation] Test Acc: {acc:.4f}")
            self.logger.info(f"[Evaluation] Test Loss: {avg_loss:.4f}")

        return acc, avg_loss


    def save_model(self, path=None, filename="best_model.pt"):
        """
        Save model state dict to disk.

        Args:
            path (str): Directory where the model should be saved.
            filename (str): Name of the model file (default: best_model.pt).
        """
        path = path or self.save_path
        os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, filename)
        torch.save(self.model.state_dict(), full_path)
        self.logger.info(f"Model saved to {full_path}")


    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.logger.info(f"Loaded checkpoint from {path}")
        else:
            self.logger.warning(f"Checkpoint not found at {path}")
