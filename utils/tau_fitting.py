# utils/tau_fitting.py

import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import trange

def fit_exp_vector(t_steps, tau_seq, num_iters=1000, lr=1e-1, device=None):
    """
    Fit tau_t = A * (1 - exp(-B * t)) for all dimensions in parallel (vectorized).

    Args:
        t_steps (array-like): shape [T], e.g., [0, 1, 2, ..., T-1]
        tau_seq (array-like): shape [T, d], full tau sequence
        num_iters (int): number of optimization iterations
        lr (float): learning rate for Adam
        device (str or None): 'cuda', 'cpu', or None to auto-detect

    Returns:
        tau_pred (ndarray): shape [d], predicted A (tau at t→∞)
        A_fit (Tensor): learned A (on CPU)
        B_fit (Tensor): learned B (on CPU, positive)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert inputs to tensors
    t = torch.tensor(t_steps, dtype=torch.float32, device=device).view(-1, 1)  # [T, 1]
    tau = torch.tensor(tau_seq, dtype=torch.float32, device=device)            # [T, d]
    T, d = tau.shape

    # Parameters to learn
    A = torch.nn.Parameter(tau[-1].clone())

    # Dynamically set initial B based on t scale
    t_mean = t.mean().item()
    if t_mean == 0:
        raw_B_init = torch.tensor(0.1, device=device)  # fallback
    else:
        # Heuristic: Set B_init ≈ 1 / t_mean (for good decay rate)
        B_init = 1.0 / t_mean
        raw_B_init = torch.log(torch.expm1(torch.tensor(B_init, device=device)))  # inverse softplus
    raw_B = torch.nn.Parameter(raw_B_init.expand(d).clone())

    optimizer = Adam([A, raw_B], lr=lr)

    for iter in trange(num_iters, desc="Fitting Exponential function:"):
        B = torch.nn.functional.softplus(raw_B)  # ensure positivity
        tau_hat = A * (1 - torch.exp(-B * t))    # [T, d]
        loss = F.mse_loss(tau_hat, tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    A_fit = A.detach().cpu()
    B_fit = torch.nn.functional.softplus(raw_B.detach()).cpu()

    return A_fit, B_fit
