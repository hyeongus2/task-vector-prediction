# utils/tau_utils.py

import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional


def get_flat_params(model: nn.Module, only_require_grad: bool = False) -> torch.Tensor:
    """
    Returns a flattened 1D tensor of all model parameters.

    Args:
        model (nn.Module): The model.
        only_require_grad (bool): Whether to include only parameters that require gradients.

    Returns:
        torch.Tensor: Flattened parameters.
    """
    if only_require_grad:
        return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    else:
        return torch.cat([p.detach().flatten() for p in model.parameters()])


def get_tau(model: nn.Module, pretrained_state: OrderedDict) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Compute tau = theta_ft - theta_pre as a flattened vector and record metadata.

    Args:
        model (nn.Module): Fine-tuned model.
        pretrained_state (OrderedDict): Pretrained model state dict.

    Returns:
        Tuple of:
            - tau (torch.Tensor): Flattened delta vector.
            - meta (List): List of (name, shape, start_idx, end_idx)
    """
    flat_params = []
    meta = []
    current_idx = 0

    for name, param in model.named_parameters():
        if name not in pretrained_state:
            continue
        if param.shape != pretrained_state[name].shape:
            continue  # skip incompatible layers

        delta = (param.detach() - pretrained_state[name].detach()).flatten()
        flat_params.append(delta)

        start_idx = current_idx
        end_idx = current_idx + delta.numel()
        meta.append((name, param.shape, start_idx, end_idx))
        current_idx = end_idx

    tau = torch.cat(flat_params)
    return tau, meta


def save_tau(
        tau: torch.Tensor,
        meta: list[tuple[str, torch.Size, int, int]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        mode: str = "step",
        out_dir: str = "checkpoints"
    ) -> str:
    """
    Save tau vector and its metadata to disk.

    Args:
        tau (torch.Tensor): Flattened tau.
        meta (List): Metadata to reconstruct parameters.
        step (int): Optional step index.
        epoch (int): Optional epoch index.
        mode (str): One of {'step', 'epoch', 'star'}.
        out_dir (str): Base directory to save.

    Returns:
        str: Full path where tau was saved.
    """
    assert mode in ["step", "epoch", "star"]
    assert (step is not None) if mode == "step" else (epoch is not None)

    subdir = {
        "step": "tau_early",
        "epoch": "tau_epoch",
        "star": ""
    }[mode]
    
    save_dir = os.path.join(out_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    if mode == "step":
        fname = f"tau_step_{step:04d}.pt"
    elif mode == "epoch":
        fname = f"tau_epoch_{epoch+1:03d}.pt"
    else:
        fname = f"tau_star.pt"

    path = os.path.join(save_dir, fname)

    torch.save({"tau": tau, "meta": meta}, path)
    return path


def load_tau(tau_path: str) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Load tau and meta from file.

    Args:
        tau_path (str): Path to .pt file

    Returns:
        Tuple of tau and meta
    """
    data = torch.load(tau_path)
    return data["tau"], data["meta"]


def reconstruct_model(pretrained_state: OrderedDict, tau: torch.Tensor, meta: list[tuple[str, torch.Size, int, int]]) -> OrderedDict:
    """
    Reconstruct fine-tuned parameters from pretrained state and tau.

    Args:
        pretrained_state (OrderedDict): Pretrained model state dict.
        tau (torch.Tensor): Flattened tau.
        meta (List): Metadata for reconstruction.

    Returns:
        OrderedDict: New state_dict of reconstructed model.
    """
    new_state = pretrained_state.copy()
    for name, shape, start, end in meta:
        delta = tau[start:end].view(shape)
        new_state[name] = pretrained_state[name] + delta
    return new_state


def tau_distance(tau1: torch.Tensor, tau2: torch.Tensor, p: int = 2) -> float:
    """
    Compute Lp distance between two tau vectors.

    Args:
        tau1 (torch.Tensor): First tau vector.
        tau2 (torch.Tensor): Second tau vector.
        p (int): Norm order (default=2 for Euclidean)

    Returns:
        float: Distance value.
    """
    return torch.norm(tau1 - tau2, p=p).item()


def tau_magnitude(tau: torch.Tensor, p:int = 2) -> float:
    """
    Compute the magnitude (L2 norm) of a tau vector.

    Args:
        tau (torch.Tensor): Tau vector.
        p (int): Norm order (default=2 for Euclidean)

    Returns:
        float: L-p norm of tau.
    """
    return torch.norm(tau, p=p).item()


def tau_cosine_similarity(tau1: torch.Tensor, tau2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two tau vectors.

    Args:
        tau1 (torch.Tensor): First tau vector.
        tau2 (torch.Tensor): Second tau vector.

    Returns:
        float: Cosine similarity value in [-1, 1].
    """
    return torch.nn.functional.cosine_similarity(tau1, tau2, dim=0).item()
    