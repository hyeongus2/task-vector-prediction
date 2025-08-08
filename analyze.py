# analyze.py

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.config_utils import parse_args, load_config, merge_config
from utils.paths import get_save_path
from utils.tau_utils import load_tau, tau_magnitude, tau_cosine_similarity
from utils.tau_fitting import fit_exp_vector

def natural_sort(file_list):
    """Sort filenames by numeric step index extracted from their name."""
    return sorted(file_list, key=lambda f: int(re.findall(r"\d+", f)[0]))

def plot_tau_magnitude(indices: np.ndarray, magnitudes: list, mag_star: float, label: str, tau_dir: str):
    plt.figure()
    plt.plot(indices, magnitudes, marker='o')
    plt.axhline(y=mag_star, color='r', linestyle='--', label="||tau*||")
    plt.title(f"Tau L2 Magnitude over {label}s")
    plt.xlabel(label)
    plt.ylabel("||tau||")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_magnitude_{label.lower()}.png"))

def plot_tau_cosine_similarity(indices: np.ndarray, cos_sims: list, label: str, tau_dir: str):
    plt.figure()
    plt.plot(indices, cos_sims, marker='o', color='orange')
    plt.title(f"Cosine Similarity to tau* over {label}s")
    plt.xlabel(label)
    plt.ylabel("cos(tau_t, tau*)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_cosine_similarity_{label.lower()}.png"))


def evaluate_fitting_all_steps(A_fit, B_fit, all_t, all_taus, label, tau_dir, tau_star=None):
    """
    Evaluate how well the fitted A, B match each tau_t.
    Saves plots of:
    - Cosine similarity
    - L2 distance
    - Magnitude difference
    - tau_pred vs tau_t comparison
    - tau_star vs tau_t comparison
    """
    A = A_fit.to(all_taus.device)
    B = B_fit.to(all_taus.device)

    t_tensor = torch.tensor(all_t, dtype=torch.float32, device=all_taus.device).view(-1, 1)  # [T, 1]
    tau_hat = A * (1 - torch.exp(-B * t_tensor))  # [T, d]

    cos_sims = [tau_cosine_similarity(tau_hat[i], all_taus[i]) for i in range(len(all_taus))]
    l2_dists = [torch.norm(tau_hat[i] - all_taus[i]).item() for i in range(len(all_taus))]
    mags_hat = [tau_magnitude(tau_hat[i]) for i in range(len(tau_hat))]
    mags_true = [tau_magnitude(all_taus[i]) for i in range(len(all_taus))]

    # Plot cosine similarity between tau_hat and tau_t
    plt.figure()
    plt.plot(all_t, cos_sims, marker='o', color='green', label="cos(tau_hat_t, tau_t)")
    plt.title(f"cos(tau_hat_t, tau_t) over {label}s")
    plt.xlabel(label)
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"fit_cosine_{label.lower()}.png"))

    # Plot L2 distance between tau_hat and tau_t
    plt.figure()
    plt.plot(all_t, l2_dists, marker='o', color='blue', label="L2(tau_hat_t - tau_t)")
    plt.title(f"L2 Distance ||tau_hat_t - tau_t|| over {label}s")
    plt.xlabel(label)
    plt.ylabel("L2 Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"fit_l2_distance_{label.lower()}.png"))

    # Plot magnitude comparison: tau_hat vs tau_t
    plt.figure()
    plt.plot(all_t, mags_true, marker='o', label='||tau_t||', color='black')
    plt.plot(all_t, mags_hat, marker='x', label='||tau_hat_t||', color='red')
    plt.title(f"Tau Magnitude over {label}s")
    plt.xlabel(label)
    plt.ylabel("||tau||")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"fit_magnitude_{label.lower()}.png"))

    # ---------- tau_pred vs tau_t and tau_star ----------
    tau_pred = A  # predicted final tau (A)
    tau_pred = tau_pred.to(all_taus.device)

    cos_t_pred = [tau_cosine_similarity(all_taus[i], tau_pred) for i in range(len(all_taus))]
    l2_t_pred = [torch.norm(all_taus[i] - tau_pred).item() for i in range(len(all_taus))]

    if tau_star is not None:
        tau_star = tau_star.to(all_taus.device)
        cos_t_star = [tau_cosine_similarity(all_taus[i], tau_star) for i in range(len(all_taus))]
        l2_t_star = [torch.norm(all_taus[i] - tau_star).item() for i in range(len(all_taus))]
    else:
        cos_t_star = None
        l2_t_star = None

    # Plot: Cosine(tau_t, tau_pred) and Cosine(tau_t, tau_star)
    plt.figure()
    plt.plot(all_t, cos_t_pred, marker='s', color='purple', label='cos(tau_t, tau_pred)')
    if cos_t_star is not None:
        plt.plot(all_t, cos_t_star, marker='o', color='green', linestyle='--', label='cos(tau_t, tau_star)')
    plt.title(f"Cosine Similarity to tau_pred and tau* over {label}s")
    plt.xlabel(label)
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_cosine_pred_star_{label.lower()}.png"))

    # Plot: L2(tau_t - tau_pred) and L2(tau_t - tau_star)
    plt.figure()
    plt.plot(all_t, l2_t_pred, marker='^', color='darkred', label='L2(tau_t - tau_pred)')
    if l2_t_star is not None:
        plt.plot(all_t, l2_t_star, marker='x', color='blue', linestyle='--', label='L2(tau_t - tau_star)')
    plt.title(f"L2 Distance to tau_pred and tau* over {label}s")
    plt.xlabel(label)
    plt.ylabel("L2 Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_l2_pred_star_{label.lower()}.png"))

    print(f"[Fit Evaluation] Mean cos(tau_hat_t, tau_t): {np.mean(cos_sims):.4f}")


def analyze_tau_sequence(save_path: str, mode: str, predict_indices=None):
    """
    Analyze a series of tau vectors from step-wise or epoch-wise checkpoints.
    Saves plots of magnitude and cosine similarity with tau*,
    and compares fitted tau_pred with tau*.
    
    Args:
        save_path (str): base directory containing tau files
        mode (str): "step" or "epoch"
        predict_indices (int | list[int] | None): which indices to use for prediction
    """
    assert mode in {"step", "epoch"}

    # Auto-config by mode
    if mode == "step":
        tau_dir = os.path.join(save_path, "tau_early")
        prefix = "tau_step_"
        label = "Step"
    else:
        tau_dir = os.path.join(save_path, "tau_epoch")
        prefix = "tau_epoch_"
        label = "Epoch"

    if not os.path.isdir(tau_dir):
        print(f"[Skip] {tau_dir} does not exist.")
        return

    tau_files = [f for f in os.listdir(tau_dir) if f.startswith(prefix) and f.endswith(".pt")]
    tau_files = natural_sort(tau_files)
    if not tau_files:
        print(f"[Skip] No files found with prefix '{prefix}' in {tau_dir}")
        return

    star_path = os.path.join(save_path, "tau_star.pt")
    if not os.path.exists(star_path):
        print(f"[Skip] tau_star.pt not found at {star_path}")
        return

    tau_star, _ = load_tau(star_path)
    mag_star = tau_magnitude(tau_star)

    taus, indices = [], []
    for f in tau_files:
        idx = int(re.findall(r"\d+", f)[0])
        tau, _ = load_tau(os.path.join(tau_dir, f))
        taus.append(tau)
        indices.append(idx)

    taus = torch.stack(taus)        # [T, d]
    indices = np.array(indices)     # [T]

    # ----- Plotting -----
    magnitudes = [tau_magnitude(tau) for tau in taus]
    cos_sims = [tau_cosine_similarity(tau, tau_star) for tau in taus]

    plot_tau_magnitude(indices, magnitudes, mag_star, label, tau_dir)
    plot_tau_cosine_similarity(indices, cos_sims, label, tau_dir)

    # ===== Predict tau* from early steps =====
    print("\n[Compare] Predicting tau* via exponential_fit...")

    # --- Handle predict_indices ---
    if predict_indices is None:
        # Default: use first 3 taus
        sel_indices = indices[:3]
        sel_taus = taus[:3]
    elif isinstance(predict_indices, int):
        sel_indices = indices[:predict_indices]
        sel_taus = taus[:predict_indices]
    elif isinstance(predict_indices, (list, tuple)):
        sel_indices, sel_taus = [], []
        for i, t in enumerate(indices):
            if t in predict_indices:
                sel_indices.append(t)
                sel_taus.append(taus[i])
        sel_indices = np.array(sel_indices)
        sel_taus = torch.stack(sel_taus)
    else:
        raise ValueError("predict_indices must be int, list[int], or None")

    if len(sel_indices) < 2:
        print(f"[Skip] Not enough data points to fit (got {len(sel_indices)})")
        return

    # --- Fit exponential curve ---
    A_fit, B_fit = fit_exp_vector(sel_indices, sel_taus, num_iters=1000, lr=0.1)
    tau_pred_exp = A_fit

    # --- Compare with tau_star and tau_final ---
    tau_final = taus[-1]

    def compare(tau_pred, method_name):
        tau_pred_tensor = torch.tensor(tau_pred)

        # 1. Compare with tau_star
        sim_star = tau_cosine_similarity(tau_pred_tensor, tau_star)
        dist_star = torch.norm(tau_pred_tensor - tau_star).item()
        mag_pred = tau_magnitude(tau_pred_tensor)
        mag_star_val = tau_magnitude(tau_star)

        # 2. Compare with tau_final
        sim_final = tau_cosine_similarity(tau_pred_tensor, tau_final)
        dist_final = torch.norm(tau_pred_tensor - tau_final).item()
        mag_final_val = tau_magnitude(tau_final)

        print(f"\n[{method_name}]")
        print(f"  --- tau_star ---")
        print(f"    cos(tau_pred, tau*): {sim_star:.4f}")
        print(f"    L2 distance        : {dist_star:.4f}")
        print(f"    ||tau_pred||       : {mag_pred:.4f}")
        print(f"    ||tau*||           : {mag_star_val:.4f}")
        print(f"  --- tau_final ---")
        print(f"    cos(tau_pred, tau_final): {sim_final:.4f}")
        print(f"    L2 distance            : {dist_final:.4f}")
        print(f"    ||tau_final||          : {mag_final_val:.4f}")

    compare(tau_pred_exp, "exponential_fit")

    # Evaluate Fitting
    evaluate_fitting_all_steps(A_fit, B_fit, indices, taus, label, tau_dir, tau_star)

    print(f"\n[Done] Saved plots and tau_pred comparisons to: {tau_dir}")

def main():
    config_path, override_dict = parse_args()
    config = load_config(config_path)
    config = merge_config(config, override_dict)
    save_path = config["save"].get("path", get_save_path(config_path=config_path, overrides=override_dict))
    config["save"]["path"] = save_path

    # You can hardcode predict_indices here for testing
    # e.g., predict_indices = [30, 90, 150] or 5 or None
    predict_indices = [2, 3, 4, 5]

    if os.path.isdir(os.path.join(save_path, "tau_epoch")):
        analyze_tau_sequence(save_path, mode="epoch", predict_indices=predict_indices)
    # if os.path.isdir(os.path.join(save_path, "tau_early")):
    #     analyze_tau_sequence(save_path, mode="step", predict_indices=predict_indices)

if __name__ == "__main__":
    main()
