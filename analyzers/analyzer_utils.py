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
from typing import Optional, Union

# ---------- File / list helpers ----------

def natural_sort(file_list: list[str]) -> list[str]:
    """Sort a list of filenames by the first numeric token (e.g., step/epoch number)."""
    m = re.compile(r"(\d+)")
    def key(f: str) -> int:
        s = m.search(f)
        return int(s.group(1)) if s else -1
    return sorted(file_list, key=key)


def select_subset(
    indices_in: Union[torch.Tensor, np.ndarray, list[int], tuple],
    taus_t: torch.Tensor,
    predict_indices: Optional[Union[int, torch.Tensor, np.ndarray, list[int], tuple]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select a subset of rows from `taus_t` based on the provided index *values*.

    Returns:
      - sel_idx_t: 1D LongTensor on CPU with the selected index values (NOT positions)
      - sel_taus : Tensor on the same device as `taus_t`, containing the selected rows

    Behavior:
      - predict_indices is None        -> take the first 3 entries
      - predict_indices is int (N)     -> take the first N entries
      - predict_indices is list/tuple/ndarray/tensor -> keep rows whose index value
        (from `indices_in`) is in that set; preserves the original order of `indices_in`.

    Raises:
        ValueError: on shape mismatches, unsupported types, empty selection, or invalid counts.
        
    Notes:
      - 'indices_in' is normalized to a CPU LongTensor of values.
      - We build a boolean mask on CPU and move only the mask to `taus_t.device`
        to index `taus_t`, keeping large tensors on their original device.
    """
    if taus_t.shape[0] == 0:
        return torch.empty(0, dtype=torch.long), torch.empty_like(taus_t)

    # Normalize incoming index values to a 1D CPU LongTensor
    indices_t = torch.as_tensor(indices_in, dtype=torch.long, device="cpu").view(-1)

    # Simple slice cases
    if predict_indices is None:
        pos = slice(0, 3)
        sel_idx_t = indices_t[pos]     # CPU long
        sel_taus  = taus_t[pos]        # same device as taus_t
        return sel_idx_t, sel_taus

    if isinstance(predict_indices, int):
        pos = slice(0, predict_indices)
        sel_idx_t = indices_t[pos]
        sel_taus  = taus_t[pos]
        return sel_idx_t, sel_taus
    
    if not isinstance(predict_indices, (list, tuple, np.ndarray, torch.Tensor)):
        raise TypeError(
            "predict_indices must be one of: int | list[int] | tuple | np.ndarray | torch.Tensor | None"
        )

    # Explicit membership case
    sel_vals = torch.as_tensor(predict_indices, dtype=torch.long, device="cpu").view(-1)

    # Build CPU mask: torch.isin returns a bool tensor already (no .bool() needed)
    mask_cpu = torch.isin(indices_t, sel_vals)        # dtype: torch.bool on CPU
    sel_idx_t = indices_t[mask_cpu]                   # CPU long (selected values)

    if sel_idx_t.numel() == 0:
        return sel_idx_t, torch.empty(0, *taus_t.shape[1:], device=taus_t.device, dtype=taus_t.dtype)

    # Index taus_t with a mask on the same device as taus_t
    sel_taus = taus_t[mask_cpu.to(device=taus_t.device)]
    return sel_idx_t, sel_taus


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
    fig = plt.figure()
    try:
        markers = ['o', 'x', 's', '^', 'd', 'v', 'p', '*', '+'] # 마커 추가
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
        
        # Ensure the output directory exists
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        plt.savefig(outpath)
    finally:
        plt.close(fig)


# ---------- Cleanup ----------

def remove_pngs_recursive(root_dir: str) -> int:
    """
    Delete all *.png files under `root_dir` recursively.
    Returns the number of deleted files.
    """
    if not os.path.isdir(root_dir):
        return 0
        
    count = 0
    # Search for all .png files in root_dir and its subdirectories
    for p in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
        try:
            os.remove(p)
            count += 1
        except OSError:
            # e.g., permission error, file not found (if deleted by another process)
            pass
    return count
