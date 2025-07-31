# utils/tau_fitting.py

import numpy as np
from scipy.optimize import curve_fit
import torch

def exp_func(t, a, b):
    return a * (1 - np.exp(-b * t))


def fit_tau_curve_fit(t_steps: np.ndarray, tau_steps: np.ndarray) -> np.ndarray:
    """
    Fit tau values (k x d) to exponential function elementwise using curve_fit.

    Args:
        t_steps (np.ndarray): Shape (k,), time steps (e.g., [1, 2, 3])
        tau_steps (np.ndarray): Shape (k, d), tau vectors at each step

    Returns:
        np.ndarray: Predicted final tau (tau_pred), shape (d,)
    """
    k, d = tau_steps.shape
    tau_pred = np.zeros(d)
    for i in range(d):
        y = tau_steps[:, i]
        try:
            popt, _ = curve_fit(exp_func, t_steps, y, bounds=(0, [np.inf, np.inf]), maxfev=5000)
            a, b = popt
            tau_pred[i] = a  # As t -> inf, tau -> a
        except RuntimeError:
            tau_pred[i] = y[-1]  # Fallback: use last observed value
    return tau_pred


class ExpFitter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, t):
        return self.a * (1 - torch.exp(-self.b * t))


def fit_tau_torch(t_steps: np.ndarray, tau_steps: np.ndarray, num_iters: int = 500, lr: float = 0.01) -> np.ndarray:
    """
    Fit tau values (k x d) to exponential function elementwise using PyTorch.

    Args:
        t_steps (np.ndarray): Shape (k,), time steps
        tau_steps (np.ndarray): Shape (k, d), tau vectors
        num_iters (int): Number of optimization steps per dimension
        lr (float): Learning rate

    Returns:
        np.ndarray: Predicted tau vector (d,)
    """
    k, d = tau_steps.shape
    tau_pred = np.zeros(d)

    for i in range(d):
        y = torch.tensor(tau_steps[:, i], dtype=torch.float32)
        t = torch.tensor(t_steps, dtype=torch.float32)
        model = ExpFitter()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(num_iters):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(t), y)
            loss.backward()
            optimizer.step()

        tau_pred[i] = model.a.item()  # a is the limit value as t -> inf

    return tau_pred
