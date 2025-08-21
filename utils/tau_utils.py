# utils/tau_utils.py

import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from models.model_utils import is_head_param

# ---------------------------
# Flattening utilities
# ---------------------------

def get_flat_params(model: nn.Module, only_require_grad: bool = False) -> torch.Tensor:
    """
    Returns a flattened 1D tensor of all model parameters.
    """
    params_iter = (p.detach().flatten() for p in model.parameters() if not only_require_grad or p.requires_grad)
    return torch.cat(tuple(params_iter))

# ---------------------------
# Tau construction
# ---------------------------

def get_tau(
    model: nn.Module,
    pretrained_state: OrderedDict,
    exclude_head: bool = True
) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Compute tau = theta_ft - theta_pre as a flattened vector and record metadata.
    Meta entries: (name, shape, start_idx, end_idx)
    """
    flat_params: list[torch.Tensor] = []
    meta: list[tuple[str, torch.Size, int, int]] = []
    current_idx = 0

    for name, param in model.named_parameters():
        if exclude_head and is_head_param(name):
            continue

        if (name not in pretrained_state or
            param.shape != pretrained_state[name].shape or
            not param.dtype.is_floating_point):
            continue

        pre = pretrained_state[name].detach()
        if pre.device != param.device or pre.dtype != param.dtype:
            pre = pre.to(param, non_blocking=True)

        delta = (param.detach() - pre).flatten()
        flat_params.append(delta)

        start_idx = current_idx
        end_idx = current_idx + delta.numel()
        meta.append((name, param.shape, start_idx, end_idx))
        current_idx = end_idx

    if not flat_params:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.empty(0, device=device), []
        
    tau = torch.cat(flat_params)
    return tau, meta

# ---------------------------
# Tau IO
# ---------------------------

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
    """
    assert mode in ["step", "epoch", "star"]
    if mode == "step":
        assert step is not None, "step must be provided when mode='step'"
    elif mode == "epoch":
        assert epoch is not None, "epoch must be provided when mode='epoch'"

    subdir = {"step": "tau_early", "epoch": "tau_epoch", "star": ""}[mode]
    save_dir = os.path.join(out_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    if mode == "step":
        fname = f"tau_step_{step:04d}.pt"
    elif mode == "epoch":
        fname = f"tau_epoch_{epoch+1:03d}.pt"
    else:
        fname = "tau_star.pt"

    path = os.path.join(save_dir, fname)
    torch.save({"tau": tau.detach().cpu(), "meta": meta}, path)
    return path

def safe_torch_load(path: str, map_location: str | torch.device = "cpu"):
    """
    Safe-ish loader for PyTorch 2.6+ default (weights_only=True).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)

def load_tau(tau_path: str, device: Optional[str | torch.device] = None) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int, int]]]:
    """
    Load tau and meta from file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data = safe_torch_load(tau_path, map_location=device)
    return data["tau"].to(device), data["meta"]

# ---------------------------
# Reconstruction (strict, buffer-safe)
# ---------------------------

def _validate_meta_against_state(pretrained_state: dict[str, torch.Tensor],
                                 meta: list[tuple[str, torch.Size, int, int]]) -> None:
    """
    Ensure every meta entry exists in state, has matching shape, and contiguous indices.
    """
    if not meta:
        raise RuntimeError("Meta is empty.")
    # Contiguity check
    expected_start = 0
    for name, shape, start, end in meta:
        if start != expected_start:
            raise RuntimeError(f"Meta indices not contiguous at '{name}': start={start}, expected={expected_start}")
        if end <= start:
            raise RuntimeError(f"Meta end must be > start at '{name}': {start}..{end}")
        expected_start = end
        if name not in pretrained_state:
            raise KeyError(f"Meta name '{name}' not found in pretrained_state.")
        tgt = pretrained_state[name]
        if tgt.shape != shape:
            raise RuntimeError(f"Shape mismatch at '{name}': meta {shape} vs state {tgt.shape}")
        if not tgt.dtype.is_floating_point:
            raise RuntimeError(f"Target tensor '{name}' is not floating type (dtype={tgt.dtype}).")
    # OK

def reconstruct_model(pretrained_state: dict[str, torch.Tensor],
                      tau: torch.Tensor,
                      meta: list[tuple[str, torch.Size, int, int]]) -> OrderedDict:
    """
    Reconstruct fine-tuned parameters from pretrained state and tau.
    Only updates parameters referenced by meta (buffers are not touched).
    """
    _validate_meta_against_state(pretrained_state, meta)

    # Verify tau length
    last_end = meta[-1][3] if meta else 0
    if tau.numel() != last_end:
        raise RuntimeError(f"Tau length {tau.numel()} does not match meta size {last_end}.")

    # Start from cloned state to avoid aliasing issues
    new_state = OrderedDict((k, v.clone()) for k, v in pretrained_state.items())

    # Apply delta to each parameter slice
    for name, shape, start, end in meta:
        target = new_state[name]
        delta = tau[start:end].view(shape).to(target.device, target.dtype)
        target.add_(delta)

    return new_state

# ---------------------------
# Diagnostics
# ---------------------------

def flatten_by_meta(sd: dict[str, torch.Tensor], meta: list[tuple[str, torch.Size, int, int]]) -> torch.Tensor:
    """
    Flatten parameters from state_dict according to meta ordering.
    """
    parts: list[torch.Tensor] = []
    for name, shape, start, end in meta:
        t = sd[name]
        parts.append(t.reshape(-1).to(torch.float32).cpu())
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.float32)

def tau_distance(tau1: torch.Tensor, tau2: torch.Tensor, p: int = 2) -> float:
    """Compute Lp distance between two tau vectors."""
    return torch.norm(tau1 - tau2, p=p).item()

def tau_magnitude(tau: torch.Tensor, p: int = 2) -> float:
    """Compute the magnitude (L2 norm by default) of a tau vector."""
    return torch.norm(tau, p=p).item()

def tau_cosine_similarity(tau1: torch.Tensor, tau2: torch.Tensor) -> float:
    """Compute cosine similarity between two tau vectors."""
    return torch.nn.functional.cosine_similarity(tau1, tau2, dim=0).item()
