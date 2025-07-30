# analyze.py

import os
import re
import matplotlib.pyplot as plt
from utils.config_utils import parse_args, load_config, merge_config
from utils.paths import get_save_path
from utils.tau_utils import load_tau, tau_magnitude, tau_cosine_similarity

def natural_sort(file_list):
    """
    Sort filenames by numeric step index extracted from their name.
    E.g., tau_step_2.pt before tau_step_10.pt
    """
    return sorted(file_list, key=lambda f: int(re.findall(r"\d+", f)[0]))


def analyze_tau_sequence(tau_dir: str, prefix: str, label: str):
    """
    Analyze a series of tau vectors from step-wise or epoch-wise checkpoints.
    Saves plots of magnitude and cosine similarity over time.

    Args:
        tau_dir (str): Directory containing tau_*.pt files.
        prefix (str): Filename prefix like "tau_step_" or "tau_epoch_".
        label (str): Label for plot titles (e.g., "Step", "Epoch")
    """
    tau_files = [f for f in os.listdir(tau_dir) if f.startswith(prefix) and f.endswith(".pt")]
    tau_files = natural_sort(tau_files)

    if not tau_files:
        print(f"[Skip] No files found with prefix '{prefix}' in {tau_dir}")
        return

    magnitudes = []
    cosine_similarities = []
    indices = []
    prev_tau = None

    for f in tau_files:
        idx = int(re.findall(r"\d+", f)[0])
        tau, _ = load_tau(os.path.join(tau_dir, f))
        indices.append(idx)

        magnitudes.append(tau_magnitude(tau))

        if prev_tau is not None:
            cosine_similarities.append(tau_cosine_similarity(prev_tau, tau))
        else:
            cosine_similarities.append(1.0)

        prev_tau = tau

    # Plot magnitude
    plt.figure()
    plt.plot(indices, magnitudes, marker='o')
    plt.title(f"Tau L2 Magnitude over {label}s")
    plt.xlabel(label)
    plt.ylabel("||tau||")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_magnitude_{label.lower()}.png"))

    # Plot cosine similarity
    plt.figure()
    plt.plot(indices, cosine_similarities, marker='o', color='orange')
    plt.title(f"Cosine Similarity between Consecutive Taus ({label}s)")
    plt.xlabel(label)
    plt.ylabel("cos(tau_t, tau_{t+1})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tau_dir, f"tau_cosine_similarity_{label.lower()}.png"))

    print(f"[Done] Saved plots for {label}-wise taus to: {tau_dir}")


def analyze_tau_star(save_path: str):
    """
    Analyze the final tau_star.pt file.

    Args:
        save_path (str): Path to directory containing tau_star.pt
    """
    star_path = os.path.join(save_path, "tau_star.pt")
    if not os.path.exists(star_path):
        print("[Skip] tau_star.pt not found.")
        return

    tau, _ = load_tau(star_path)
    mag = tau_magnitude(tau)

    with open(os.path.join(save_path, "tau_star_magnitude.txt"), "w") as f:
        f.write(f"||tau_star|| (L2 norm): {mag:.6f}\n")

    print(f"[Done] tau_star magnitude: {mag:.6f} (saved to tau_star_magnitude.txt)")


def main():
    # Load config
    config_path, override_dict = parse_args()
    config = load_config(config_path)
    config = merge_config(config, override_dict)
    save_path = config["save"].get("path", get_save_path(config_path=config_path, overrides=override_dict))
    config["save"]["path"] = save_path

    # Analyze each available tau mode
    if os.path.isdir(os.path.join(save_path, "tau_early")):
        analyze_tau_sequence(os.path.join(save_path, "tau_early"), prefix="tau_step_", label="Step")

    if os.path.isdir(os.path.join(save_path, "tau_epoch")):
        analyze_tau_sequence(os.path.join(save_path, "tau_epoch"), prefix="tau_epoch_", label="Epoch")

    analyze_tau_star(save_path)


if __name__ == "__main__":
    main()
