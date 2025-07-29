# trainers/__init__.py

from .full_finetuning import FullFinetuneTrainer
from .lora import LoRATrainer
from .pretraining import PretrainTrainer


def get_trainer(config, model, train_loader, val_loader, logger):
    strategy = config["train"]["strategy"]

    if strategy == "full_finetuning":
        return FullFinetuneTrainer(config, model, train_loader, val_loader, logger)
    elif strategy == "lora":
        return LoRATrainer(config, model, train_loader, val_loader, logger)
    elif strategy == "pretraining":
        return PretrainTrainer(config, model, train_loader, val_loader, logger)
    else:
        raise ValueError(f"[ERROR] Unsupported training strategy: {strategy}")
