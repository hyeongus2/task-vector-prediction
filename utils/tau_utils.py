# utils/tau_utils.py

import os
import torch
import torch.nn as nn
from collections import OrderedDict


def get_flat_params(model: nn.Module, only_require_grad: bool = False) -> torch.Tensor:
    """
    Returns a flat tensor of all model parameters.
    """
    if only_require_grad:
        return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    else:
        return torch.cat([p.detach().flatten() for p in model.parameters()])


def get_tau(model: nn.Module, pretrained_state: OrderedDict) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Compute tau = theta_ft - theta_pre as a flattened 1D vector, along with metadata to reconstruct full parameter shapes.

    Returns:
        tau: Flattened 1D tau vector
        meta: List of (name, shape, start_idx, end_idx) to reconstruct full tensor
    """
    flat_params = []
    meta = []
    current_idx = 0

    for name, param in model.named_parameters():
        if name not in pretrained_state:
            continue
        if param.shape != pretrained_state[name].shape:
            continue  # Skip mismatched layers (e.g., classifier)

        tau_tensor = (param.detach() - pretrained_state[name].detach()).flatten()
        flat_params.append(tau_tensor)

        start_idx = current_idx
        end_idx = current_idx + tau_tensor.numel()
        meta.append((name, param.shape, start_idx, end_idx))
        current_idx = end_idx

    tau = torch.cat(flat_params)
    return tau, meta


def reconstruct_model(pretrained_state: OrderedDict, tau: torch.Tensor, meta: list[tuple[str, torch.Size, int, int]]) -> OrderedDict:
    """
    Reconstruct theta_ft = theta_pre + tau using flat tau vector and metadata.

    Returns:
        new_state_dict: OrderedDict with reconstructed parameters
    """
    new_state_dict = pretrained_state.copy()

    for name, shape, start, end in meta:
        tau_slice = tau[start:end].view(shape)
        new_state_dict[name] = pretrained_state[name] + tau_slice

    return new_state_dict


def save_tau(tau: torch.Tensor, meta: list[tuple[str, torch.Size, int, int]], epoch: int, tau_dir: str):
    """
    Save tau vector and metadata to disk.
    """
    os.makedirs(tau_dir, exist_ok=True)
    torch.save({
        'tau': tau,
        'meta': meta,
    }, os.path.join(tau_dir, f"tau_epoch_{epoch:03d}.pt"))


def load_tau(tau_path: str) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Load tau vector and metadata from disk.
    """
    data = torch.load(tau_path)
    return data['tau'], data['meta']