# trainers/base_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.seed import set_seed
import abc

class BaseTrainer(abc.ABC):
    def __init__(self, config: dict, model: nn.Module, train_loader, test_loader, logger):
        self.config = config
        self.model = model
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

        self.save_path = config["train"].get("save_path", "checkpoints/best_model.pt")

        set_seed(config.get("seed", 42))


    def get_criterion_and_optimizer(self):
        task = self.config["data"]["task"]
        opt_name = self.config["train"].get("optimizer", "adam")
        lr = self.config["train"].get("lr", 1e-4)
        momentum = self.config["train"].get("momentum", 0.9)

        criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
        if opt_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_name}")

        return criterion, optimizer


    @abc.abstractmethod
    def train(self):
        pass


    @abc.abstractmethod
    def eval(self):
        pass


    def save_model(self, path=None):
        path = path or self.save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")


    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.logger.info(f"Loaded checkpoint from {path}")
        else:
            self.logger.warning(f"Checkpoint not found at {path}")
