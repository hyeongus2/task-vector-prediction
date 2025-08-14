# analyzers/utils.py
# Utility helpers for analysis: sorting files, subset selection,
# cache path building, robust torch.load, plotting helpers, and PNG cleanup.

import os
import re
import json
import glob
import hashlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# ---------- File / list helpers ----------

def natural_sort(file_list: list[str]) -> list[str]:
    """Sort a list of filenames by the first numeric token (e.g., step/epoch number)."""
    m = re.compile(r"(\d+)")
    def key(f: str) -> int:
        s = m.search(f)
        return int(s.group(1)) if s else -1
    return sorted(file_list, key=key)


def select_subset(indices_np: np.ndarray, taus_t: torch.Tensor, predict_indices=None) -> tuple[np.ndarray, torch.Tensor]:
    """
    Slice a subset (indices, taus) based on predict_indices:
      - None  -> first 3
      - int   -> first N
      - list/tuple/ndarray -> explicit indices
    """
    if predict_indices is None:
        return indices_np[:3], taus_t[:3]
    if isinstance(predict_indices, int):
        return indices_np[:predict_indices], taus_t[:predict_indices]
    if isinstance(predict_indices, (list, tuple, np.ndarray)):
        mask = np.isin(indices_np, np.array(predict_indices))
        sel_idx = indices_np[mask]
        mask_t = torch.from_numpy(mask).to(taus_t.device).bool()
        sel_taus = taus_t[mask_t]
        return sel_idx, sel_taus
    raise ValueError("predict_indices must be int, list/tuple[int], ndarray, or None")


def _indices_tag(predict_indices) -> str:
    """Human-readable tag for predict_indices; short-hash when the list is long."""
    if predict_indices is None:
        return "first3"
    if isinstance(predict_indices, int):
        return f"first{predict_indices}"
    if isinstance(predict_indices, (list, tuple, np.ndarray)):
        try:
            vals = sorted(int(v) for v in predict_indices)
        except Exception:
            vals = list(predict_indices)
        if len(vals) <= 8:
            return "idx-" + "_".join(map(str, vals))
        s = json.dumps(vals, ensure_ascii=False)
        h = hashlib.sha1(s.encode()).hexdigest()[:8]
        return f"idx-{vals[0]}_{vals[1]}_..._{vals[-1]}-{h}"
    return "custom"


def _lr_tag(lr: float) -> str:
    """Convert LR to a filesystem-friendly string (e.g., 0.1 -> 0p1)."""
    return f"{lr:.6g}".replace('.', 'p').replace('-', 'm')


def build_cache_path(save_path: str, mode: str, predict_indices, fit_num_iters: int, fit_lr: float) -> str:
    """
    Build a unique, human-readable filename for (mode, indices, iters, lr).
    Example: tau_pred_epoch_first5_it1000_lr0p1_deadbeef.pt
    """
    tag = _indices_tag(predict_indices)
    lr_s = _lr_tag(fit_lr)
    key = {"mode": mode, "indices": tag, "num_iters": fit_num_iters, "lr": fit_lr}
    short = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    fname = f"tau_pred_{mode}_{tag}_it{fit_num_iters}_lr{lr_s}_{short}.pt"
    return os.path.join(save_path, fname)


def safe_torch_load(path: str, map_location=None):
    """
    Robust torch.load that tolerates PyTorch 2.6+ `weights_only=True` default.
    Falls back to a safe load if the first attempt fails.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not accept weights_only
        return torch.load(path, map_location=map_location)


# ---------- Plotting ----------

def line_plot(
    x,
    ys: list[list[float]],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    hlines: Optional[list[tuple[float, str, str, str]]] = None,  # (yval, label, color, linestyle)
):
    """Generic line plot with optional horizontal guide lines."""
    plt.figure()
    markers = ['o', 'x', 's', '^', 'd', 'v']
    for i, (y, lab) in enumerate(zip(ys, labels)):
        plt.plot(x, y, marker=markers[i % len(markers)], label=lab)
    if hlines:
        for yval, lab, color, ls in hlines:
            plt.axhline(y=yval, linestyle=ls, color=color, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()


# ---------- Cleanup ----------

def remove_pngs_recursive(root_dir: str) -> int:
    """
    Delete all *.png files under `root_dir` recursively.
    Returns the number of deleted files.
    """
    count = 0
    for base, _, _ in os.walk(root_dir):
        for p in glob.glob(os.path.join(base, "*.png")):
            try:
                os.remove(p)
                count += 1
            except Exception:
                pass
    return count
