# utils/logger.py

import os
import torch
import logging
import wandb
from datetime import datetime
from typing import Optional
from utils import tau_logger
from utils.config_utils import flatten_config

class DummyLogger:
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def log_wandb(self, *args, **kwargs): pass


def init_logger(config):
    if not config.get("logging", {}).get("enabled", False):
        return DummyLogger()
    
    log_dir = config["logging"].get("dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_name = os.path.basename(os.path.normpath(config["save"]["path"])) + f"_run_{timestamp}"
    run_name = config["logging"].get("run_name", default_run_name)
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

    # Avoid duplicated logs if re-running inside Jupyter or shell
    logger.propagate = False

    # WandB
    if config["logging"].get("use_wandb", False):
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=config["logging"].get("run_name", run_name),
            config=flatten_config(config)
        )

        def log_wandb(*, tau: torch.Tensor, step: int, path: Optional[str] = None):
            tau_logger.log_tau_scalar(tau, step)
            tau_logger.log_tau_histogram(tau, step)
            tau_logger.log_tau_table(tau, step)
            if config["logging"].get("log_tau_plot", False):
                tau_logger.log_tau_plot(tau, step)
            if config["logging"].get("log_tau_artifact", False) and path:
                tau_logger.log_tau_artifact(path, step)

        def log_wandb_scalar(metrics: dict, step: Optional[int] = None):
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        def finish_wandb():
            wandb.finish()

        logger.log_wandb = log_wandb
        logger.log_wandb_scalar = log_wandb_scalar
        logger.finish_wandb = finish_wandb
        
    else:
        logger.log_wandb = lambda *args, **kwargs: None         # Dummy
        logger.log_wandb_scalar = lambda *args, **kwargs: None  # Dummy
        logger.finish_wandb = lambda *args, **kwargs: None      # Dummy

    return logger
