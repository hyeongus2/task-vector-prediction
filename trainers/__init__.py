# trainers/__init__.py

from trainers.full_finetuning import FullFinetuneTrainer
from trainers.lora import LoRATrainer

def get_trainer(config, model, train_loader, test_loader, logger):
    method = config["finetuning"].get("method", "full")
    if method == "full":
        return FullFinetuneTrainer(config, model, train_loader, test_loader, logger)
    elif method == "lora":
        return LoRATrainer(config, model, train_loader, test_loader, logger)
    else:
        raise ValueError(f"Unknown method: {method}")
