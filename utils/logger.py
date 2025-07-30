import logging
import os
import wandb
from datetime import datetime

def init_logger(config):
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("logging", {}).get("run_name", f"run_{timestamp}")
    log_path = os.path.join(log_dir, f"{run_name}.log")

    # File logger
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))

    # Console logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # WandB
    if config.get("logging", {}).get("use_wandb", False):
        wandb.init(project=config["logging"]["wandb_project"],
                   name=run_name,
                   config=config)

    return logger
