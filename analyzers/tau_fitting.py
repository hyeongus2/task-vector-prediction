# utils/tau_fitting.py
# Exponential fitting utilities for task-vector trajectories.
#
# Notation:
#   Single-exp model: tau_t  ≈  A * (1 - exp(-B * t))
#   Two-exp model   : tau_t  ≈  A1*(1 - exp(-B1*t)) + A2*(1 - exp(-B2*t))
# Where A, A1, A2 are vectors in R^d and B, B1, B2 are positive vectors in R^d.
#
# Both fits are vectorized over dimensions; each parameter (A, B, ...) has shape [d].

from __future__ import annotations

import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import trange
from typing import Optional, Tuple


# -----------------------------
# Internal helpers
# -----------------------------

def _as_float_tensor(x, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input (Tensor/ndarray/list) to float32 tensor on `device`.
    If already a Tensor, clone to avoid accidental in-place sharing.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().clone()
        if device is not None:
            t = t.to(device)
        return t.float()
    t = torch.tensor(x, dtype=torch.float32, device=device)
    return t


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable inverse of softplus for y>0:
      softplus(x) = log(1 + exp(x))  =>  x = log(exp(y) - 1)
    """
    # clamp to avoid log(0) for tiny y
    eps = torch.finfo(y.dtype).eps
    return torch.log(torch.clamp(torch.exp(y) - 1.0, min=eps))


# -----------------------------
# Single-exponential fitting
# -----------------------------

def fit_exp_vector(
    t_steps,
    tau_seq,
    num_iters: int = 1000,
    lr: float = 1e-1,
    device: Optional[str] = None,
    weight_decay: float = 0.0,
    progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit tau_t = A * (1 - exp(-B * t)) for all dimensions in parallel (vectorized).

    Args:
        t_steps: shape [k], e.g., [1, 2, ..., k]
        tau_seq: shape [k, d], observed tau sequence
        num_iters: number of Adam steps
        lr: learning rate for Adam
        device: 'cuda' | 'cpu' | None (auto)
        weight_decay: optional L2 on A to discourage overgrowth
        progress: show tqdm progress bar

    Returns:
        A_fit (Tensor): shape [d]
        B_fit (Tensor): shape [d], positive
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Inputs
    t = _as_float_tensor(t_steps, device=device).view(-1, 1)  # [k, 1]
    tau = _as_float_tensor(tau_seq, device=device)            # [k, d]
    k, d = tau.shape

    # Parameters to learn (initialize)
    # A starts from the last observed tau (reasonable terminal estimate)
    A = torch.nn.Parameter(tau[-1].clone())  # [d]

    # Heuristic for B initialization based on t scale
    t_mean = float(t.mean().item())
    if t_mean <= 0:
        B0 = torch.full((d,), 0.1, device=device)
    else:
        # A decent starting decay rate
        B0 = torch.full((d,), 1.0 / t_mean, device=device)
    raw_B = torch.nn.Parameter(_inv_softplus(B0))  # so that softplus(raw_B) ≈ B0

    optimizer = Adam([A, raw_B], lr=lr)

    iters = trange(num_iters, desc="fit_exp_vector", disable=not progress)
    for _ in iters:
        B = F.softplus(raw_B)                  # [d], ensure positivity
        tau_hat = A * (1 - torch.exp(-B * t))  # [k, d]
        loss = F.mse_loss(tau_hat, tau)
        if weight_decay > 0:
            loss = loss + 0.5 * weight_decay * (A.norm(2) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    A_fit = A.detach()
    B_fit = F.softplus(raw_B.detach())
    return A_fit, B_fit


# -----------------------------
# Two-exponential fitting
# -----------------------------

def fit_exp2_vector(
    t_steps,
    tau_seq,
    num_iters: int = 1500,
    lr: float = 1e-1,
    device: Optional[str] = None,
    init_split: Tuple[float, float] = (0.7, 0.3),
    b_scales: Tuple[float, float] = (2.0, 0.2),
    weight_decay: float = 0.0,
    progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit a two-term exponential model per dimension:
        tau_t ≈ A1*(1 - exp(-B1*t)) + A2*(1 - exp(-B2*t))

    This allows one fast and one slow timescale, which often matches real training traces better.

    Args:
        t_steps: shape [k]
        tau_seq: shape [k, d]
        num_iters: Adam steps
        lr: learning rate
        device: 'cuda' | 'cpu' | None (auto)
        init_split: (alpha1, alpha2) so that A1≈alpha1*tau_last, A2≈alpha2*tau_last, alpha1+alpha2≈1
        b_scales: (s_fast, s_slow). B_fast≈s_fast/t_mean, B_slow≈s_slow/t_mean
        weight_decay: optional L2 on A1,A2
        progress: show tqdm

    Returns:
        A1_fit, B1_fit, A2_fit, B2_fit  (all shape [d], with B1_fit,B2_fit > 0)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Inputs
    t = _as_float_tensor(t_steps, device=device).view(-1, 1)  # [k,1]
    tau = _as_float_tensor(tau_seq, device=device)            # [k,d]
    k, d = tau.shape

    # Initialize A1, A2 from the last observation
    tau_last = tau[-1].clone()  # [d]
    a1, a2 = init_split
    s = max(a1 + a2, 1e-6)
    a1, a2 = a1 / s, a2 / s
    A1 = torch.nn.Parameter((a1 * tau_last))
    A2 = torch.nn.Parameter((a2 * tau_last))

    # Initialize B1 (fast) and B2 (slow) from t scale
    t_mean = float(t.mean().item())
    if t_mean <= 0:
        B1_0 = torch.full((d,), 0.2, device=device)
        B2_0 = torch.full((d,), 0.05, device=device)
    else:
        s_fast, s_slow = b_scales
        B1_0 = torch.full((d,), s_fast / t_mean, device=device)
        B2_0 = torch.full((d,), s_slow / t_mean, device=device)

    raw_B1 = torch.nn.Parameter(_inv_softplus(B1_0))
    raw_B2 = torch.nn.Parameter(_inv_softplus(B2_0))

    params = [A1, A2, raw_B1, raw_B2]
    optimizer = Adam(params, lr=lr)

    iters = trange(num_iters, desc="fit_exp2_vector", disable=not progress)
    for _ in iters:
        B1 = F.softplus(raw_B1)  # [d] positive
        B2 = F.softplus(raw_B2)  # [d] positive

        # tau_hat: [k,d] via broadcasting
        tau_hat = A1 * (1 - torch.exp(-B1 * t)) + A2 * (1 - torch.exp(-B2 * t))

        loss = F.mse_loss(tau_hat, tau)
        if weight_decay > 0:
            loss = loss + 0.5 * weight_decay * ((A1.norm(2) ** 2) + (A2.norm(2) ** 2))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    A1_fit = A1.detach()
    A2_fit = A2.detach()
    B1_fit = F.softplus(raw_B1.detach())
    B2_fit = F.softplus(raw_B2.detach())
    return A1_fit, B1_fit, A2_fit, B2_fit


# -----------------------------
# (Optional) Predictors
# -----------------------------

@torch.no_grad()
def predict_exp_vector(A: torch.Tensor, B: torch.Tensor, t_steps) -> torch.Tensor:
    """
    Predict tau_hat(t) for a single-exp model.
    Args:
        A, B: shape [d]
        t_steps: [k]
    Returns:
        tau_hat: [k, d]
    """
    device = A.device
    t = _as_float_tensor(t_steps, device=device).view(-1, 1)  # [k,1]
    return A * (1 - torch.exp(-F.softplus(B) * t)) if (B.min() <= 0).item() else A * (1 - torch.exp(-B * t))


@torch.no_grad()
def predict_exp2_vector(A1: torch.Tensor, B1: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor, t_steps) -> torch.Tensor:
    """
    Predict tau_hat(t) for a two-exp model.
    Args:
        A1,B1,A2,B2: shape [d]
        t_steps: [k]
    Returns:
        tau_hat: [k, d]
    """
    device = A1.device
    t = _as_float_tensor(t_steps, device=device).view(-1, 1)  # [k,1]
    # Accept either raw-B or already-positive B
    B1p = F.softplus(B1) if (B1.min() <= 0).item() else B1
    B2p = F.softplus(B2) if (B2.min() <= 0).item() else B2
    return A1 * (1 - torch.exp(-B1p * t)) + A2 * (1 - torch.exp(-B2p * t))
