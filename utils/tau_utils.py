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

        # Ensure both tensors are on the same device before subtraction
        pre = pretrained_state[name].detach()
        if pre.device != param.device:
            pre = pre.to(param.device)

        delta = (param.detach() - pre).flatten()
        flat_params.append(delta)

        start_idx = current_idx
        end_idx = current_idx + delta.numel()
        meta.append((name, param.shape, start_idx, end_idx))
        current_idx = end_idx

    tau = torch.cat(flat_params) if flat_params else torch.empty(0, device=next(model.parameters()).device)
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
    if mode == "step":
        assert step is not None, "step must be provided when mode='step'"
    elif mode == "epoch":
        assert epoch is not None, "epoch must be provided when mode='epoch'"

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
    torch.save({"tau": tau.detach().cpu(), "meta": meta}, path)
    return path


def safe_torch_load(path: str, map_location: str | torch.device = "cpu"):
    """
    Safe-ish loader for PyTorch 2.6+ default (weights_only=True).
    1) Try safe load (weights_only=True).
    2) If that fails due to restricted globals, fall back to weights_only=False
       (only do this if you trust the file - here we assume it's our own artifact).
    """
    try:
        # Try safe mode first (PyTorch 2.6+)
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch < 2.6: no weights_only arg
        return torch.load(path, map_location=map_location)
    except Exception:
        # Trusted fallback: allow full unpickling if needed
        return torch.load(path, map_location=map_location, weights_only=False)


def load_tau(tau_path: str, device: Optional[str | torch.device] = None) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Load tau and meta from file.

    Args:
        tau_path (str): Path to .pt file

    Returns:
        Tuple of tau and meta
    """
    # Resolve desired map_location
    if device is None:
        map_loc = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        map_loc = torch.device(device) if isinstance(device, str) else device
        # If user asked for CUDA but it's unavailable, fall back to CPU
        if map_loc.type == 'cuda' and not torch.cuda.is_available():
            map_loc = torch.device('cpu')

    data = safe_torch_load(tau_path, map_location=map_loc)
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

        # Match device and dtype of the target tensor to avoid mismatches
        target = pretrained_state[name]
        if delta.device != target.device:
            delta = delta.to(target.device)
        if delta.dtype != target.dtype:
            delta = delta.to(dtype=target.dtype)

        new_state[name] = target + delta
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
    