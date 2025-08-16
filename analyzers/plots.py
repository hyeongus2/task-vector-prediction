# analyzers/plots.py
# Plot helpers that take raw sequences (tau_t, tau_hat_t) and reference vectors
# (tau_star, tau_pred, tau_final) and save informative figures.

import os
import torch
import numpy as np

from utils.tau_utils import tau_magnitude, tau_cosine_similarity
from .analyzer_utils import line_plot

__all__ = [
    "plot_combo_magnitude",
    "plot_cos_hat_refs",
    "plot_cos_obs_refs",
    "plot_l2_hat_refs",
    "plot_l2_obs_refs",
    "plot_refs_pairwise"
]


def plot_combo_magnitude(indices: np.ndarray,
                         taus: torch.Tensor,
                         tau_hat_list: list[torch.Tensor],
                         tau_star: torch.Tensor,
                         tau_pred: torch.Tensor,
                         label: str,
                         out_dir: str) -> str:
    """
    Plot magnitudes of tau_t and tau_hat_t.
    Add horizontal dashed lines for ||tau_star|| (red) and ||tau_pred|| (magenta).
    """
    mags_true = [float(tau_magnitude(t)) for t in taus]
    mags_hat  = [float(tau_magnitude(h)) for h in tau_hat_list]
    mag_star  = float(tau_magnitude(tau_star))
    mag_pred  = float(tau_magnitude(tau_pred))

    out = os.path.join(out_dir, f"mag_combo_{label.lower()}.png")
    line_plot(indices, [mags_true, mags_hat], ["||tau_t||", "||tau_hat_t||"],
              f"Magnitude (tau_t vs tau_hat_t) over {label}s", label, "||tau||",
              out, hlines=[(mag_star, "||tau*||", "r", "--"),
                          (mag_pred, "||tau_pred||", "m", "--")])
    return out


def plot_cos_obs_refs(indices: np.ndarray,
                      taus: torch.Tensor,
                      tau_star: torch.Tensor,
                      tau_pred: torch.Tensor,
                      tau_final: torch.Tensor,
                      label: str,
                      out_dir: str) -> str:
    """
    Cosine similarities of observed tau_t against references: tau*, tau_pred, tau_final.
    """
    cos_obs_star  = [float(tau_cosine_similarity(taus[i], tau_star))   for i in range(len(indices))]
    cos_obs_pred  = [float(tau_cosine_similarity(taus[i], tau_pred))   for i in range(len(indices))]
    cos_obs_final = [float(tau_cosine_similarity(taus[i], tau_final))  for i in range(len(indices))]

    out = os.path.join(out_dir, f"cos_obs_refs_{label.lower()}.png")
    line_plot(indices, [cos_obs_star, cos_obs_pred, cos_obs_final],
              ["cos(tau_t, tau*)", "cos(tau_t, tau_pred)", "cos(tau_t, tau_final)"],
              f"Cosine: tau_t vs references over {label}s", label, "Cosine Similarity", out)
    return out


def plot_cos_hat_refs(indices: np.ndarray,
                      tau_hat_list: list[torch.Tensor],
                      taus: torch.Tensor,
                      tau_star: torch.Tensor,
                      tau_pred: torch.Tensor,
                      tau_final: torch.Tensor,
                      label: str,
                      out_dir: str) -> str:
    """
    Cosine similarities of predicted tau_hat_t against references: tau*, tau_pred, tau_final, and tau_t.
    The last one (cos(tau_hat_t, tau_t)) is the fit residual direction similarity.
    """
    cos_hat_star  = [float(tau_cosine_similarity(tau_hat_list[i], tau_star))   for i in range(len(indices))]
    cos_hat_pred  = [float(tau_cosine_similarity(tau_hat_list[i], tau_pred))   for i in range(len(indices))]
    cos_hat_final = [float(tau_cosine_similarity(tau_hat_list[i], tau_final))  for i in range(len(indices))]
    cos_hat_obs   = [float(tau_cosine_similarity(tau_hat_list[i], taus[i]))    for i in range(len(indices))]

    out = os.path.join(out_dir, f"cos_hat_refs_{label.lower()}.png")
    line_plot(indices, [cos_hat_star, cos_hat_pred, cos_hat_final, cos_hat_obs],
              ["cos(tau_hat_t, tau*)", "cos(tau_hat_t, tau_pred)", "cos(tau_hat_t, tau_final)", "cos(tau_hat_t, tau_t)"],
              f"Cosine: tau_hat_t vs references (incl. residual) over {label}s", label, "Cosine Similarity", out)
    return out


def plot_l2_obs_refs(indices: np.ndarray,
                     taus: torch.Tensor,
                     tau_star: torch.Tensor,
                     tau_pred: torch.Tensor,
                     tau_final: torch.Tensor,
                     label: str,
                     out_dir: str) -> str:
    """
    L2 distances of observed tau_t to references: tau*, tau_pred, tau_final.
    """
    l2_obs_star  = [float(torch.norm(taus[i] - tau_star).item())  for i in range(len(indices))]
    l2_obs_pred  = [float(torch.norm(taus[i] - tau_pred).item())  for i in range(len(indices))]
    l2_obs_final = [float(torch.norm(taus[i] - tau_final).item()) for i in range(len(indices))]

    out = os.path.join(out_dir, f"l2_obs_refs_{label.lower()}.png")
    line_plot(indices, [l2_obs_star, l2_obs_pred, l2_obs_final],
              ["L2(tau_t - tau*)", "L2(tau_t - tau_pred)", "L2(tau_t - tau_final)"],
              f"L2: tau_t vs references over {label}s", label, "L2 Distance", out)
    return out


def plot_l2_hat_refs(indices: np.ndarray,
                     tau_hat_list: list[torch.Tensor],
                     taus: torch.Tensor,
                     tau_star: torch.Tensor,
                     tau_pred: torch.Tensor,
                     tau_final: torch.Tensor,
                     label: str,
                     out_dir: str) -> str:
    """
    L2 distances of predicted tau_hat_t to references: tau*, tau_pred, tau_final, and tau_t.
    The last one (L2(tau_hat_t, tau_t)) is the fit residual magnitude.
    """
    l2_hat_star  = [float(torch.norm(tau_hat_list[i] - tau_star).item())  for i in range(len(indices))]
    l2_hat_pred  = [float(torch.norm(tau_hat_list[i] - tau_pred).item())  for i in range(len(indices))]
    l2_hat_final = [float(torch.norm(tau_hat_list[i] - tau_final).item()) for i in range(len(indices))]
    l2_hat_obs   = [float(torch.norm(tau_hat_list[i] - taus[i]).item())   for i in range(len(indices))]

    out = os.path.join(out_dir, f"l2_hat_refs_{label.lower()}.png")
    line_plot(indices, [l2_hat_star, l2_hat_pred, l2_hat_final, l2_hat_obs],
              ["L2(tau_hat_t - tau*)", "L2(tau_hat_t - tau_pred)", "L2(tau_hat_t - tau_final)", "L2(tau_hat_t - tau_t)"],
              f"L2: tau_hat_t vs references (incl. residual) over {label}s", label, "L2 Distance", out)
    return out


def plot_refs_pairwise(tau_star: torch.Tensor,
                       tau_pred: torch.Tensor,
                       tau_final: torch.Tensor,
                       label: str,
                       out_dir: str) -> str:
    """
    Simple bar summary of pairwise cosine and L2 among (tau_pred, tau_star, tau_final).
    Useful to check how close tau_pred(∞) is to tau_final and tau*.
    """
    import matplotlib.pyplot as plt

    pairs = [("pred,star", tau_pred, tau_star),
             ("pred,final", tau_pred, tau_final),
             ("star,final", tau_star, tau_final)]

    cos_vals = [float(tau_cosine_similarity(a, b)) for _, a, b in pairs]
    l2_vals  = [float(torch.norm(a - b).item()) for _, a, b in pairs]
    labels_bars = [p[0] for p in pairs]
    bar_colors = ['#ffb3ba', '#baffc9', '#bae1ff']

    out = os.path.join(out_dir, f"refs_pairwise_{label.lower()}.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bars0 = axes[0].bar(labels_bars, cos_vals, color=bar_colors)
    axes[0].set_title("Pairwise Cosine among (tau_pred, tau_final, tau*)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].bar_label(bars0, fmt="%.3f", label_type="center")

    bars1 = axes[1].bar(labels_bars, l2_vals, color=bar_colors)
    axes[1].set_title("Pairwise L2 among (tau_pred, tau_final, tau*)")
    axes[1].set_ylabel("L2 Distance")
    axes[1].bar_label(bars1, fmt="%.3f", label_type="center")

    fig.suptitle(f"Reference pairwise summary ({label})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
