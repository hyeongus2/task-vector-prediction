# src/tvp/plotting.py
import logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Create a logger for this module
logger = logging.getLogger(__name__)


def plot_magnitude_comparison(
    steps: torch.Tensor,
    actual_mags: torch.Tensor,
    predicted_mags: torch.Tensor,
    star_mag: float,
    title_suffix: str,
    save_path: Path
):
    """Plots the L2 magnitude of actual and predicted trajectories over time."""
    fig, ax = plt.subplots()
    ax.plot(steps, actual_mags, 'o-', color='blue', label=r'Actual $\|\tau_t\|$')
    ax.plot(steps, predicted_mags, 'o-', color='green', label=r'Predicted $\|\hat{\tau}_t\|$')
    ax.axhline(y=star_mag, c='red', ls='--', label=r'$\|\tau_*\|$')
    ax.set_title(f'Magnitude {title_suffix}')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("L2 Magnitude")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path / "1_magnitude_comparison.png")
    logger.info(f"Saved plot: {save_path.name}/1_magnitude_comparison.png")
    plt.close(fig)


def plot_cosine_similarity_between_trajectories(
    steps: torch.Tensor,
    cos_sims: torch.Tensor,
    final_cos_sim: float,
    title_suffix: str,
    save_path: Path
):
    """Plots the cosine similarity between actual and predicted trajectories."""
    fig, ax = plt.subplots()
    ax.plot(steps, cos_sims, 'o-', color='purple', label=r'cos($\hat{\tau}_t$, $\tau_t$)')
    ax.axhline(y=final_cos_sim, c='plum', ls='--', label=rf'Final cos($\hat{{\tau}}_\infty$, $\tau_\infty$): {final_cos_sim:.4f}')
    ax.set_xlim(left=-50)
    ax.set_title(r'cos($\hat{\tau}_t$, $\tau_t$) ' + f'{title_suffix}')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cosine Similarity")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path / "2_cosine_similarity_between.png")
    logger.info(f"Saved plot: {save_path.name}/2_cosine_similarity_between.png")
    plt.close(fig)


def plot_l2_distance_between_trajectories(
    steps: torch.Tensor,
    l2_dists: torch.Tensor,
    final_l2_dist: float,
    title_suffix: str,
    save_path: Path
):
    """Plots the L2 distance between actual and predicted trajectories."""
    fig, ax = plt.subplots()
    ax.plot(steps, l2_dists, 'o-', color='purple', label=r'$\|\hat{\tau}_t - \tau_t\|$')
    ax.axhline(y=final_l2_dist, c='plum', ls='--', label=rf'Final L2($\hat{{\tau}}_\infty$, $\tau_\infty$): {final_l2_dist:.4f}')
    ax.set_xlim(left=-50)
    ax.set_title(r'$\|\hat{\tau}_t - \tau_t\|$ ' + f'{title_suffix}')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("L2 Distance")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path / "3_l2_distance_between.png")
    logger.info(f"Saved plot: {save_path.name}/3_l2_distance_between.png")
    plt.close(fig)


def plot_alignment_to_star(
    steps: torch.Tensor,
    actual_sims: torch.Tensor,
    predicted_sims: torch.Tensor,
    title_suffix: str,
    save_path: Path
):
    """Plots the cosine alignment of actual and predicted trajectories with tau_star."""
    fig, ax = plt.subplots()
    ax.plot(steps, actual_sims, 'o-', color='blue', label=r'cos($\tau_t$, $\tau_*$)')
    ax.plot(steps, predicted_sims, 'o-', color='green', label=r'cos($\hat{\tau}_t$, $\tau_*$)')
    ax.set_title(r'Cosine Alignment with $\tau_*$ ' + f'{title_suffix}')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cosine Similarity")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path / "4_alignment_to_star.png")
    logger.info(f"Saved plot: {save_path.name}/4_alignment_to_star.png")
    plt.close(fig)


def plot_distance_to_star(
    steps: torch.Tensor,
    actual_dists: torch.Tensor,
    predicted_dists: torch.Tensor,
    title_suffix: str,
    save_path: Path
):
    """Plots the L2 distance of actual and predicted trajectories to tau_star."""
    fig, ax = plt.subplots()
    ax.plot(steps, actual_dists, 'o-', color='blue', label=r'$\|\tau_t - \tau_*\|$')
    ax.plot(steps, predicted_dists, 'o-', color='green', label=r'$\|\hat{\tau}_t - \tau_*\|$')
    ax.set_title(r'L2 Distance to $\tau_*$ ' + f'{title_suffix}')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("L2 Distance")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path / "5_distance_to_star.png")
    logger.info(f"Saved plot: {save_path.name}/5_distance_to_star.png")
    plt.close(fig)


def plot_final_performance_comparison(
    top_candidates: List[Dict],
    best_val_acc: float,
    acc_star: float,
    acc_zero: float,
    title_suffix: str,
    save_path: Path
):
    """
    Plots a bar chart comparing the final accuracies of theta_0 and the top predicted candidates.
    """
    fig, ax = plt.subplots()

    # --- 1. Prepare data for the bar chart ---
    # Labels for the x-axis
    labels = [r'$\theta_0$'] + [f"Cand. (Trial {c['trial_num']})" for c in top_candidates]
    # Corresponding accuracies
    accuracies = [acc_zero] + [c['val_acc'] for c in top_candidates]
    # Colors for the bars
    colors = ['orange'] + ['green'] * len(top_candidates)

    # --- 2. Create the bar chart ---
    ax.bar(labels, accuracies, color=colors, zorder=2)
    ax.set_xticklabels(labels, rotation=45, ha="right") # Rotate labels to prevent overlap

    # --- 3. Add horizontal lines and legend ---
    # Actual best model (tau_*)
    ax.axhline(y=acc_star, c='red', ls='--', label=rf'Actual Best ($\tau_*$): {acc_star:.4f}', zorder=3)

    # Baseline model (tau_0)
    ax.axhline(y=acc_zero, c='sandybrown', ls='--', label=rf'Baseline ($\theta_0$): {acc_zero:.4f}', zorder=3)

    # Predicted best model (tau_hat_infty from the best candidate)
    best_pred_acc = best_val_acc
    if best_pred_acc > 0:
        ax.axhline(
            y=best_pred_acc, 
            c='limegreen', 
            ls='--', 
            label=rf'Predicted Best ($\hat{{\tau}}_\infty$): {best_pred_acc:.4f}',
            zorder=3
        )

    # --- 4. Set titles and labels ---
    ax.set_title(f'Final Model Performance Comparison {title_suffix}')
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation Accuracy")
    ax.set_ylim(bottom=min(accuracies + [acc_star]) - 0.05, top=1.01) # Adjust y-axis limits
    ax.grid(True, axis='y', zorder=0)
    ax.legend()
    
    fig.tight_layout() # Adjust layout to make room for rotated labels
    fig.savefig(save_path / "6_final_performance_comparison.png")
    logger.info(f"Saved plot: {save_path.name}/6_final_performance_comparison.png")
    plt.close(fig)