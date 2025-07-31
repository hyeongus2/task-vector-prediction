# analyze.py

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.config_utils import parse_args, load_config, merge_config
from utils.paths import get_save_path
from utils.tau_utils import load_tau, tau_magnitude, tau_cosine_similarity
from utils.tau_fitting import fit_tau_curve_fit, fit_tau_torch

def natural_sort(file_list):
    """
    Sort filenames by numeric step index extracted from their name.
    E.g., tau_step_2.pt before tau_step_10.pt
    """
    return sorted(file_list, key=lambda f: int(re.findall(r"\d+", f)[0]))


def analyze_tau_sequence(save_path: str, mode: str):
    """
    Analyze a series of tau vectors from step-wise or epoch-wise checkpoints.
    Saves plots of magnitude and cosine similarity with tau*.

    Args:
        save_path (str): Root experiment directory containing tau_* folders and tau_star.pt
        mode (str): "step" or "epoch"
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

    taus = []
    indices = []

    for f in tau_files:
        idx = int(re.findall(r"\d+", f)[0])
        tau, _ = load_tau(os.path.join(tau_dir, f))
        taus.append(tau)
        indices.append(idx)

    taus = torch.stack(taus)    # (k, d)
    indices = np.array(indices)
    

    # ----- Plot tau magnitude over time -----
    magnitudes = [tau_magnitude(tau) for tau in taus]

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


    # ----- Plot cosine similarity to tau_star -----
    cos_sims = [tau_cosine_similarity(tau, tau_star) for tau in taus]

    plt.figure()
    plt.plot(indices, cos_sims, marker='o', color='orange')
    plt.title(f"Cosine Similarity to tau* over {label}s")
    plt.xlabel(label)
    plt.ylabel("cos(tau_t, tau*)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_cosine_similarity_{label.lower()}.png"))


    # ===== Fit tau_pred and Compare =====
    print("\n[Compare] Predicting tau* via curve_fit and torch...")

    tau_pred_curve = fit_tau_curve_fit(indices, taus.numpy())   # [d]
    tau_pred_torch = fit_tau_torch(indices, taus.numpy())       # [d]

    # Compare cosine similarity and L2 distance
    def compare(tau_pred, method_name):
        tau_pred_tensor = torch.tensor(tau_pred)
        sim = tau_cosine_similarity(tau_pred_tensor, tau_star)
        dist = torch.norm(tau_pred_tensor - tau_star).item()
        mag_pred = tau_magnitude(tau_pred_tensor)
        mag_star = tau_magnitude(tau_star)

        print(f"\n[{method_name}]")
        print(f"  cos(tau_pred, tau*): {sim:.4f}")
        print(f"  L2 distance        : {dist:.4f}")
        print(f"  ||tau_pred||       : {mag_pred:.4f}")
        print(f"  ||tau*||           : {mag_star:.4f}")

    compare(tau_pred_curve, "curve_fit")
    compare(tau_pred_torch, "torch_fit")

    print(f"\n[Done] Saved plots and tau_pred comparisons to: {tau_dir}")


def main():
    # Load config
    config_path, override_dict = parse_args()
    config = load_config(config_path)
    config = merge_config(config, override_dict)
    save_path = config["save"].get("path", get_save_path(config_path=config_path, overrides=override_dict))
    config["save"]["path"] = save_path

    # Analyze each available tau mode
    if os.path.isdir(os.path.join(save_path, "tau_early")):
        analyze_tau_sequence(save_path, mode="step")

    if os.path.isdir(os.path.join(save_path, "tau_epoch")):
        analyze_tau_sequence(save_path, mode="epoch")


if __name__ == "__main__":
    main()
