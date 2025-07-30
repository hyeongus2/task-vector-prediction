# trainers/base_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from utils.seed import set_seed
from utils.tau_utils import get_tau, save_tau

class BaseTrainer():
    def __init__(self, config: dict, model: nn.Module, train_loader, test_loader, logger):
        self.config = config
        self.model = model
        self.pretrained_state: OrderedDict = OrderedDict(model.state_dict())
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion, self.optimizer = self.get_criterion_and_optimizer()
        self.epochs = config["train"].get("epochs", 10)

        # Early stopping params
        self.patience = config["train"].get("patience", 5)
        self.delta = config["train"].get("delta", 1e-4)
        self.best_loss = float('inf')
        self.no_improve = 0

        # Save
        self.save_enabled = config["save"]["enable"]
        self.save_every = config["save"].get("every", 10)
        self.save_max = self.save_every * config["save"].get("max", float('inf'))
        self.save_path = config["save"]["path"]

        # Set seed
        set_seed(config.get("seed", 42))


    def get_criterion_and_optimizer(self):
        task = self.config["data"]["task"]
        opt_name = self.config["train"].get("optimizer", "adam")
        lr = self.config["train"].get("lr", 1e-4)
        momentum = self.config["train"].get("momentum", 0.9)

        criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        if opt_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_name}")

        return criterion, optimizer


    def train(self):
        step = 0

        for epoch in range(self.epochs):
            self.model.train()
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

                # Save tau_t for some steps
                if self.save_enabled and step < self.save_max and step % self.save_every == 0:
                    tau, meta = get_tau(self.model, self.pretrained_state)
                    save_tau(tau, meta, step=step, mode="step", out_dir=self.save_path)
                    self.logger.log_wandb(tau=tau, step=step, path=None)
                step += 1

                # update tqdm progress bar with loss
                pbar.set_postfix(loss=loss.item())

            train_loss = total_loss / len(self.train_loader)
            self.logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")

            # Save tau_t for every epoch
            if self.save_enabled:
                tau, meta = get_tau(self.model, self.pretrained_state)
                save_tau(tau, meta, epoch=epoch, mode="epoch", out_dir=self.save_path)

            # Validation after every epoch
            val_loss = self.eval(epoch)

            self.logger.log_wandb_scalar({
                "loss/train": train_loss,
                "loss/val": val_loss,
            }, step=epoch+1)

            # Early stopping check (based on val_loss)
            if val_loss + self.delta < self.best_loss:
                self.best_loss = val_loss
                self.no_improve = 0
                if self.save_enabled:
                    self.save_model()
                    save_tau(tau, meta, epoch=epoch, mode="star", out_dir=self.save_path)
                    self.logger.info(f"Saved best model and corresponding tau_star at epoch {epoch+1}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break


    def eval(self, epoch=None) -> float:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(self.test_loader, desc="[Evaluation]", leave=False)
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.test_loader)

        if epoch is not None:
            self.logger.info(f"[Epoch {epoch+1}] Validation Loss: {avg_loss:.4f}")
        else:
            self.logger.info(f"[Evaluation] Test Loss: {avg_loss:.4f}")

        return avg_loss


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
