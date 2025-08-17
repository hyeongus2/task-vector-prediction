# analyzers/tau_analysis.py
# High-level analysis pipeline:
# - Load tau series (tau_t) and tau_star
# - Fit A,B for tau_hat(t) = A * (1 - exp(-B t)), where A = tau_pred (t→∞)
# - Compute tau_hat_t, make plots, and optionally evaluate a model by adding tau_pred to backbone.

import os
import re
import torch
import numpy as np
from typing import Optional

from utils.tau_utils import load_tau, tau_magnitude, tau_cosine_similarity, safe_torch_load
from .tau_fitting import fit_exp_vector, fit_exp2_vector
from analyzers.analyzer_utils import (
    natural_sort, select_subset, build_cache_path, remove_pngs_recursive
)
from analyzers.plots import *
from analyzers.eval_apply import eval_with_backbone_tau_using_trainer



def _cfg_for_mode(mode: str) -> tuple[str, str, str]:
    assert mode in {"step", "epoch"}
    return {"step": ("tau_early", "tau_step_", "Step"),
            "epoch": ("tau_epoch", "tau_epoch_", "Epoch")}[mode]


def _load_tau_series(
    save_path: str, subdir: str, prefix: str, logger
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """Load all tau_t tensors and their indices from save_path/subdir that match prefix."""
    tau_dir = os.path.join(save_path, subdir)
    if not os.path.isdir(tau_dir):
        logger.warning(f"[Skip] {tau_dir} does not exist.")
        return None, None, None

    files = [f for f in os.listdir(tau_dir) if f.startswith(prefix) and f.endswith(".pt")]
    files = natural_sort(files)
    if not files:
        logger.warning(f"[Skip] No files found with prefix '{prefix}' in {tau_dir}")
        return None, None, None

    taus, indices = [], []
    pat = re.compile(r"(\d+)")
    for f in files:
        m = pat.search(f)
        if not m:
            continue
        idx = int(m.group(1))
        t, _ = load_tau(os.path.join(tau_dir, f))
        taus.append(t)
        indices.append(idx)

    if not taus:
        logger.warning(f"[Skip] No tau tensors parsed in {tau_dir}")
        return None, None, None

    return torch.stack(taus), torch.tensor(indices), tau_dir


def _load_tau_star(save_path: str, logger):
    """Load tau_star (tau at best validation checkpoint)."""
    star_path = os.path.join(save_path, "tau_star.pt")
    if not os.path.exists(star_path):
        logger.warning(f"[Skip] tau_star.pt not found at {star_path}")
        return None
    tau_star, _ = load_tau(star_path)
    return tau_star


def _fit_or_load_tau_pred(save_path: str, mode: str,
                          indices: torch.Tensor, taus: torch.Tensor,
                          predict_indices,
                          num_iters: int, lr: float, device: torch.device,
                          fit_type: str, logger):
    """
    Load cached tau_pred/B_fit if exists; otherwise fit with chosen model.
    fit_type: 'exp' (single exponential) | 'exp2' (two-term exponential if available)
    """
    cache_path = build_cache_path(save_path, mode, predict_indices, num_iters, lr)
    logger.info(f"[Cache] tau_pred path = {cache_path}")

    B_fit = None; extra = {}
    tau_pred = None; obj = None

    if os.path.exists(cache_path):
        logger.info(f"[Cache] Loading tau_pred from {cache_path}")
        obj = safe_torch_load(cache_path, map_location=device)
        if isinstance(obj, dict):
            tau_pred = obj.get('tau_pred')
            B_fit    = obj.get('B_fit')
            extra    = obj.get('extra', {})
            if tau_pred is not None: tau_pred = tau_pred.to(device)
            if B_fit is not None:    B_fit    = B_fit.to(device)
        elif isinstance(obj, torch.Tensor):
            tau_pred = obj.to(device)

    if tau_pred is None or B_fit is None:
        logger.info(f"\n[Fit] Predicting tau∞ via {fit_type} fit ...")
        sel_idx, sel_tau = select_subset(indices, taus, predict_indices)
        if len(sel_idx) < 2:
            logger.warning(f"[Skip] Not enough data points to fit (got {len(sel_idx)})")
            return None, None, extra

        if fit_type == "exp2":
            # Assume fit_exp2_vector returns (A1,B1,A2,B2); tau∞ = A1 + A2
            A1, B1, A2, B2 = fit_exp2_vector(sel_idx, sel_tau, num_iters=num_iters, lr=lr)
            tau_pred = (A1 + A2)
            # For plotting quality, store an averaged B
            B_fit = (B1 + B2) / 2.0
            extra = {"A1": A1.detach().cpu(), "B1": B1.detach().cpu(),
                     "A2": A2.detach().cpu(), "B2": B2.detach().cpu()}
        else:
            tau_pred, B_fit = fit_exp_vector(sel_idx, sel_tau, num_iters=num_iters, lr=lr)

        torch.save({
            "tau_pred": tau_pred.detach().cpu(),
            "B_fit":    B_fit.detach().cpu() if B_fit is not None else None,
            "indices_used": sel_idx.detach().cpu(),
            "fit_params": {"num_iters": num_iters, "lr": lr, "fit_type": fit_type},
            "extra": extra
        }, cache_path)
        logger.info(f"[Cache] Saved tau_pred & B_fit to {cache_path}")
    else:
        logger.info("[Cache] tau_pred (and possibly B_fit) loaded.")

    return tau_pred.to(device), B_fit.to(device), {k: v.to(device) for k, v in extra.items()}


def run_one_mode(config: dict, mode: str, predict_indices, fit_num_iters: int, fit_lr: float,
                 fit_type: str, cleanup_pngs: bool, logger):
    """Run full analysis for a single mode: load, fit/load cache, plot, and optional eval."""
    use_wandb = config["logging"].get("use_wandb", False)
    save_path = config["save"]["path"]
    subdir, prefix, label = _cfg_for_mode(mode)

    # Optional cleanup: remove stale *.png to avoid confusion
    tau_dir = os.path.join(save_path, subdir)
    if cleanup_pngs and os.path.isdir(tau_dir):
        removed = remove_pngs_recursive(tau_dir)
        logger.info(f"[Cleanup] Removed {removed} *.png under {tau_dir}")

    # Load tau_t series and tau*
    taus, indices, tau_dir = _load_tau_series(save_path, subdir, prefix, logger)
    if taus is None:
        return
    device = taus.device

    tau_star = _load_tau_star(save_path, logger)
    if tau_star is None:
        return

    logger.info(
        f"[Info] Earliest ||tau||={float(tau_magnitude(taus[0])):.6f}, "
        f"||tau_final||={float(tau_magnitude(taus[-1])):.6f}, "
        f"||tau*||={float(tau_magnitude(tau_star)):.6f}"
    )

    # Load or fit tau_pred (=A) and B
    tau_pred, B_fit, _ = _fit_or_load_tau_pred(
        save_path, mode, indices, taus, predict_indices, fit_num_iters, fit_lr, device, fit_type, logger
    )
    if tau_pred is None:
        return

    # Predict tau_hat(t) for all t (if B is available)
    if B_fit is None:
        logger.warning("[Eval] B_fit is None — skipping tau_hat_t computation")
        tau_hat_list = [taus[i] for i in range(len(indices))]  # placeholder to keep plotting code simple
    else:
        t_tensor = torch.tensor(indices, dtype=torch.float32, device=device).view(-1, 1)
        tau_hat = tau_pred * (1 - torch.exp(-B_fit * t_tensor))
        tau_hat_list = [tau_hat[i] for i in range(len(indices))]

    # Log summary vs tau* and tau_final
    tau_final = taus[-1]
    logger.info(
        "\n[Compare] tau_pred (A, t→∞) vs tau* and tau_final\n"
        f"  --- tau* ---\n"
        f"    cos(tau_pred, tau*): {float(tau_cosine_similarity(tau_pred, tau_star)):.4f}\n"
        f"    L2 distance        : {float(torch.norm(tau_pred - tau_star).item()):.4f}\n"
        f"    ||tau_pred||       : {float(tau_magnitude(tau_pred)):.4f}\n"
        f"    ||tau*||           : {float(tau_magnitude(tau_star)):.4f}\n"
        f"  --- tau_final ---\n"
        f"    cos(tau_pred, tau_final): {float(tau_cosine_similarity(tau_pred, tau_final)):.4f}\n"
        f"    L2 distance            : {float(torch.norm(tau_pred - tau_final).item()):.4f}\n"
        f"    ||tau_final||          : {float(tau_magnitude(tau_final)):.4f}"
    )

    # Create plots (6 figures)
    paths = []
    paths.append(plot_combo_magnitude(indices, taus, tau_hat_list, tau_star, tau_pred, label, tau_dir))
    paths.append(plot_cos_obs_refs(indices, taus, tau_star, tau_pred, tau_final, label, tau_dir))
    paths.append(plot_cos_hat_refs(indices, tau_hat_list, taus, tau_star, tau_pred, tau_final, label, tau_dir))
    paths.append(plot_l2_obs_refs(indices, taus, tau_star, tau_pred, tau_final, label, tau_dir))
    paths.append(plot_l2_hat_refs(indices, tau_hat_list, taus, tau_star, tau_pred, tau_final, label, tau_dir))
    paths.append(plot_refs_pairwise(tau_star, tau_pred, tau_final, label, tau_dir))
    logger.info(f"[Done] Saved plots to: {tau_dir}")

    # Optional: upload figures to W&B
    if use_wandb:
        try:
            import wandb
            imgs = [wandb.Image(p, caption=os.path.basename(p)) for p in paths if os.path.exists(p)]
            if imgs:
                wandb.log({f"analyze/{mode}": imgs})
                logger.info(f"[W&B] Uploaded {len(imgs)} plots for mode={mode}")
        except Exception as e:
            logger.warning(f"[W&B] Upload failed: {e}")

    # Copy final head, add tau_pred to backbone, sweep alpha
    az = config.get("analyze", {})
    alpha_grid        = az.get("alpha_grid", [0.25, 0.5, 0.75, 1.0])   # whatever you want
    head_ckpts        = az.get("head_ckpts")                    # optional: list[str]
    bn_recalc_batches = int(az.get("bn_recalc_batches", 100))

    logger.info("[Eval] Running eval_with_backbone_tau_using_trainer ...")
    eval_with_backbone_tau_using_trainer(
        config=config,
        tau_pred=tau_pred,
        logger=logger,
        alphas=alpha_grid,
        head_ckpts=head_ckpts,
        bn_recalc_batches=bn_recalc_batches
    )


def run_analysis(config: dict, modes: list[str], logger) -> None:
    """
    Public entry for analysis:
      - Reads options from config["analyze"]
      - Loops over modes and runs per-mode pipeline
    """
    az = config.get("analyze", {})
    predict_indices    = az.get("indices", az.get("predict_indices", 5))
    fit_num_iters      = az.get("num_iters", az.get("fit_num_iters", 1000))
    fit_lr             = az.get("lr", az.get("fit_lr", 0.1))
    fit_type           = az.get("fit_type", "exp").lower()  # 'exp' | 'exp2'
    cleanup_pngs       = az.get("cleanup_pngs", False)

    logger.info(f"[Analyze] predict_indices={predict_indices}, fit_num_iters={fit_num_iters}, fit_lr={fit_lr}, fit_type={fit_type}")
    for mode in modes:
        logger.info(f"\n===== Analyzing mode: {mode} =====")
        run_one_mode(config, mode, predict_indices, fit_num_iters, fit_lr, fit_type, cleanup_pngs, logger)
