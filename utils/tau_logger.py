# utils/tau_logger.py

import os
import torch
import wandb
import matplotlib.pyplot as plt

from typing import Optional
from wandb import Table, Histogram, Artifact, Image

def log_tau_scalar(tau: torch.Tensor, step: int):
    """Logs individual tau[i] values as scalars."""
    jump = tau.numel() // 100
    metrics = {f"tau/val_{i}": v for i, v in enumerate(tau.cpu().tolist()[:tau.numel():jump])}
    wandb.log(metrics, step=step)

def log_tau_histogram(tau: torch.Tensor, step: int):
    """Logs tau as a histogram."""
    wandb.log({"tau/hist": Histogram(tau.cpu().numpy())}, step=step)

def log_tau_table(tau: torch.Tensor, step: int):
    """Logs tau as a wandb Table with index and value."""
    table = Table(columns=["index", "value"])
    for i, v in enumerate(tau.cpu().tolist()):
        table.add_data(i, v)
    wandb.log({"tau/table": table}, step=step)

def log_tau_plot(tau: torch.Tensor, step: int):
    """Logs a matplotlib line plot of tau."""
    fig, ax = plt.subplots(figsize=(10, 3))
    tau_np = tau.cpu().numpy()
    ax.plot(range(len(tau_np)), tau_np)
    ax.set_title(f"Tau @ step {step}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    wandb.log({"tau/plot": Image(fig)}, step=step)
    plt.close(fig)

def log_tau_artifact(tau_path: str, step: int, artifact_dir: Optional[str] = "tau_artifacts"):
    """Logs a saved tau file as an artifact."""
    os.makedirs(artifact_dir, exist_ok=True)
    filename = os.path.basename(tau_path)
    artifact = Artifact(f"tau_{filename}", type="tau")
    artifact.add_file(tau_path)
    wandb.log_artifact(artifact)
