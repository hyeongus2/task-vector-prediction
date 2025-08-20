# analyzers/eval_apply.py
# Evaluate adding a predicted task vector (tau_pred) to the backbone only,
# reusing BaseTrainer.eval() and applying backbone updates via reconstruct_model + saved meta.
#
# Changes vs previous:
# - Deep-copy base state_dict to avoid reference entanglement
# - Strict asserts on tau length vs meta
# - Optional BN recalibration is applied consistently (baseline and alphas) only if enabled
# - DeltaCheck: verifies (theta_new - theta0) matches alpha * tau (L2 ratio, cosine, max abs error)
# - Extra validations and clearer logging

import os
import re
import copy
import torch
import torch.nn as nn
from typing import Iterable, Optional
from torch.utils.data import DataLoader
from torchvision import transforms

from models import build_model, get_input_size
from models.model_utils import get_head, reset_head
from data import get_datasets
from trainers.base_trainer import BaseTrainer
from utils.tau_utils import safe_torch_load, reconstruct_model

# ---------------------------
# Data loader aligned with training
# ---------------------------

def build_eval_loader(config: dict, model: nn.Module, logger) -> DataLoader:
    """Build a test DataLoader with transforms consistent with training."""
    _, test_dataset = get_datasets(config)
    model_type = config["model"]["type"]
    input_size = get_input_size(model, config)

    if model_type == "image":
        in_channels = input_size[0]
        resize = input_size[1:]

        tfm = [transforms.Resize(resize)]
        sample = test_dataset[0][0]
        if hasattr(sample, "mode") and sample.mode == "L" and in_channels == 3:
            tfm.append(transforms.Grayscale(3))
        tfm.append(transforms.ToTensor())
        test_dataset.transform = transforms.Compose(tfm)

    elif model_type == "text":
        logger.info("Text dataset assumed pre-tokenized by get_datasets.")
    elif model_type == "tabular":
        logger.info("Tabular dataset assumed pre-scaled by get_datasets.")
    elif model_type == "synthetic":
        logger.info("Synthetic dataset: no extra transform applied.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

# ---------------------------
# Checkpoint & head helpers
# ---------------------------

def _extract_state_dict(obj):
    """Extract a state_dict from common checkpoint layouts; also supports raw state_dict."""
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "weights", "ema_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    elif isinstance(obj, nn.Module):
        return obj.state_dict()
    raise RuntimeError("Unsupported checkpoint format for state_dict extraction")

def _score_filename_priority(fname: str) -> int:
    """Heuristic priority for head checkpoints: 'best' > 'last' > 'epochNNN' > 'stepNNN' > others."""
    f = fname.lower()
    if "best" in f:
        return 1_000_000
    if "last" in f:
        return 900_000
    m = re.search(r"epoch[_\-]?(\d+)", f)
    if m:
        return 100_000 + int(m.group(1))
    m = re.search(r"step[_\-]?(\d+)", f)
    if m:
        return 10_000 + int(m.group(1))
    return 0

def find_head_checkpoints(save_path: str) -> list[str]:
    """
    Recursively search for *.pt / *.pth under save_path (exclude tau_*.pt).
    Prefer files with 'best' or 'last' in the name; else fall back to epoch/step files.
    Sorted by (priority desc, mtime desc).
    """
    cand_best_last, cand_epoch_step = [], []

    for root, _, files in os.walk(save_path):
        for f in files:
            if not (f.endswith(".pt") or f.endswith(".pth")):
                continue
            if f.startswith("tau_"):
                continue
            full = os.path.join(root, f)
            pri = _score_filename_priority(f)
            if ("best" in f.lower()) or ("last" in f.lower()):
                cand_best_last.append((pri, os.path.getmtime(full), full))
            else:
                if re.search(r"(epoch|step)[_\-]?\d+", f.lower()):
                    cand_epoch_step.append((pri, os.path.getmtime(full), full))

    target = cand_best_last if cand_best_last else cand_epoch_step
    target.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, __, p in target]

def load_head_module_from_ckpt(config: dict, ckpt_path: str, device: str, logger) -> nn.Module:
    """
    Build a temporary model of the same architecture, load checkpoint weights into it
    (strict=False to tolerate different head shapes), then extract its head module.
    """
    tmp = build_model(config).to(device)
    obj = safe_torch_load(ckpt_path, map_location=device)
    sd = _extract_state_dict(obj)

    incompatible = tmp.load_state_dict(sd, strict=False)
    try:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
    except Exception:
        missing, unexpected = incompatible

    if missing or unexpected:
        logger.info(f"[Head] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")

    head = copy.deepcopy(get_head(tmp)).to(device)
    return head

# ---------------------------
# Delta check helpers
# ---------------------------

def _flatten_by_meta(sd: dict[str, torch.Tensor], meta) -> torch.Tensor:
    """Flatten parameters from state_dict in the order defined by meta."""
    parts: list[torch.Tensor] = []
    for name, shape, start, end in meta:
        t = sd[name]
        parts.append(t.reshape(-1).to(torch.float32).cpu())
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.float32)

def _delta_vs_tau_report(base_sd: dict[str, torch.Tensor],
                         new_sd: dict[str, torch.Tensor],
                         tau_vec: torch.Tensor,
                         alpha: float,
                         meta,
                         logger) -> None:
    """Compare (theta_new - theta0) vs alpha * tau over the backbone slice defined by meta."""
    v0 = _flatten_by_meta(base_sd, meta)
    v1 = _flatten_by_meta(new_sd, meta)
    if v0.numel() == 0:
        logger.warning("[DeltaCheck] Empty meta slice; nothing to compare.")
        return
    delta = v1 - v0
    target = (alpha * tau_vec.to(torch.float32).cpu())

    eps = 1e-12
    l2_ratio = float(delta.norm() / (target.norm() + eps))
    cos = float((delta @ target) / ((delta.norm() + eps) * (target.norm() + eps)))
    max_abs = float((delta - target).abs().max())
    logger.info(f"[DeltaCheck] alpha={alpha:.6f} | l2_ratio={l2_ratio:.6f}, cos={cos:.6f}, max_abs_err={max_abs:.3e}")

# ---------------------------
# BN recalibration
# ---------------------------

@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader: DataLoader, device: str, num_batches: int = 100):
    """Recompute BatchNorm running stats with a few forward passes. No-op if no BN layers or num_batches<=0."""
    if not num_batches or num_batches <= 0:
        return
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)) for m in model.modules())
    if not has_bn:
        return
    was_training = model.training
    model.train()
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        model(x)
        seen += 1
        if seen >= max(1, num_batches):
            break
    model.train(was_training)

# ---------------------------
# Public API (reconstruct_model path)
# ---------------------------

def eval_with_backbone_tau_using_trainer(
    config: dict,
    tau_pred: torch.Tensor,          # flattened predicted tau (ideally backbone-only)
    logger,
    alphas: Iterable[float] = (0.5, 1.0),
    head_ckpts: Optional[list[str]] = None,
    bn_recalc_batches: int = 0,      # set >0 to enable BN recalibration; applied consistently to baseline and alphas
):
    """
    Evaluate by reconstructing backbone weights via reconstruct_model(pretrained_state, alpha*tau_pred, meta)
    and swapping in a finetuned head module extracted from each candidate checkpoint.
    Logs every trial and emits DeltaCheck diagnostics.
    """
    use_wandb = config["logging"].get("use_wandb", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build theta0 (pretrained) model and test loader
    model = build_model(config).to(device)
    test_loader = build_eval_loader(config, model, logger)

    # Instantiate BaseTrainer to reuse eval(); no training is performed
    trainer = BaseTrainer(config, model, None, test_loader, logger)

    # Load meta once (shared for all trials)
    meta = load_meta_for_reconstruct(config["save"]["path"], logger)

    # Resolve head checkpoints if not provided
    if not head_ckpts:
        head_ckpts = find_head_checkpoints(config["save"]["path"])
        if not head_ckpts:
            logger.warning("[Eval] No head checkpoints found; aborting.")
            return

    # Cache pretrained state_dict snapshot (deep copy!)
    base_sd = copy.deepcopy(trainer.model.state_dict())

    # Assert tau length matches meta
    try:
        last_end = max(end for _, _, _, end in meta)
    except Exception as e:
        raise RuntimeError(f"[Eval] Invalid meta structure: {e}")

    assert tau_pred.numel() == last_end, f"[Eval] tau_pred length {tau_pred.numel()} != meta size {last_end}"

    # Norms for quick scale sense-check
    tau_l2 = float(tau_pred.to(torch.float32).norm())
    tau_inf = float(tau_pred.to(torch.float32).abs().max())
    logger.info(f"[Tau] ||tau_pred||_2={tau_l2:.4e}, ||tau_pred||_inf={tau_inf:.4e}, len={tau_pred.numel()}")

    best = {"acc": None, "loss": float("inf"), "alpha": 0.0, "head": None}
    alphas = list(alphas)

    grid_rows = []
    global_step = 0

    for h_idx, ckpt in enumerate(head_ckpts):
        try:
            # Build a head module from this checkpoint
            head_module = load_head_module_from_ckpt(config, ckpt, device, logger)

            # Baseline (alpha=0): reset to theta0, set head, optional BN recalib, eval
            trainer.model.load_state_dict(base_sd, strict=True)
            reset_head(trainer.model, head_module)
            recalibrate_bn(trainer.model, trainer.test_loader, device, num_batches=bn_recalc_batches)
            base_acc, base_loss = trainer.eval(epoch=None)
            logger.info(f"[Eval] HEAD='{os.path.basename(ckpt)}' | alpha=0.0 → acc={base_acc}, loss={base_loss:.4f}")

            if use_wandb:
                try:
                    logger.log_wandb_scalar({
                        "eval/grid/acc": -1.0 if base_acc is None else base_acc,
                        "eval/grid/loss": base_loss,
                        "eval/grid/alpha": 0.0,
                        "eval/grid/head_idx": float(h_idx),
                    }, step=global_step)
                except Exception:
                    pass
            grid_rows.append({
                "head": os.path.basename(ckpt),
                "alpha": 0.0,
                "acc": float(-1.0 if base_acc is None else base_acc),
                "loss": float(base_loss),
            })

            better = False
            if base_acc is not None:
                if best["acc"] is None or base_acc > best["acc"]:
                    better = True
            else:
                if base_loss < best["loss"]:
                    better = True
            if better:
                best.update({"acc": base_acc, "loss": base_loss, "alpha": 0.0, "head": ckpt})

            global_step += 1

            # Sweep alphas
            for a in alphas:
                trainer.model.load_state_dict(base_sd, strict=True)

                # Reconstruct backbone state with alpha * tau_pred
                new_sd = reconstruct_model(base_sd, a * tau_pred, meta)

                # DeltaCheck to verify addition is correct
                _delta_vs_tau_report(base_sd, new_sd, tau_pred, a, meta, logger)

                # Load reconstructed state (strict=False to tolerate head mismatches)
                trainer.model.load_state_dict(new_sd, strict=False)

                # Restore this finetuned head
                reset_head(trainer.model, head_module)

                # Optional BN recalibration (applied consistently)
                recalibrate_bn(trainer.model, trainer.test_loader, device, num_batches=bn_recalc_batches)

                # Eval
                acc, loss = trainer.eval(epoch=None)
                logger.info(f"[Eval] HEAD='{os.path.basename(ckpt)}' | alpha={a:.4f} → acc={acc}, loss={loss:.4f}")

                if use_wandb:
                    try:
                        logger.log_wandb_scalar({
                            "eval/grid/acc": -1.0 if acc is None else acc,
                            "eval/grid/loss": loss,
                            "eval/grid/alpha": float(a),
                            "eval/grid/head_idx": float(h_idx),
                        }, step=global_step)
                    except Exception:
                        pass
                grid_rows.append({
                    "head": os.path.basename(ckpt),
                    "alpha": float(a),
                    "acc": float(-1.0 if acc is None else acc),
                    "loss": float(loss),
                })

                better = False
                if acc is not None:
                    if best["acc"] is None or acc > best["acc"]:
                        better = True
                else:
                    if loss < best["loss"]:
                        better = True
                if better:
                    best.update({"acc": acc, "loss": loss, "alpha": a, "head": ckpt})

                global_step += 1

        except Exception as e:
            logger.warning(f"[Eval] Failed on head '{ckpt}': {e}")

    # Final report
    if best["head"] is not None:
        logger.info(
            f"[Eval] BEST: head='{os.path.basename(best['head'])}', alpha={best['alpha']:.4f} "
            f"→ acc={best['acc']}, loss={best['loss']:.4f}"
        )
        try:
            if use_wandb:
                logger.log_wandb_scalar(
                    {"eval/best_acc": -1.0 if best["acc"] is None else best["acc"],
                     "eval/best_loss": best["loss"],
                     "eval/best_alpha": best["alpha"]}
                )
        except Exception:
            pass
    else:
        logger.warning("[Eval] No valid (head, alpha) produced a result.")

    # Log the full grid as a W&B Table (one shot)
    if use_wandb and grid_rows:
        try:
            import wandb
            table = wandb.Table(columns=["head", "alpha", "acc", "loss"])
            for r in grid_rows:
                table.add_data(r["head"], r["alpha"], r["acc"], r["loss"])
            wandb.log({"eval/grid_table": table})
        except Exception as e:
            logger.warning(f"[W&B] Table log failed: {e}")

# ---------------------------
# Meta loader (from tau files)
# ---------------------------

def load_meta_for_reconstruct(save_path: str, logger):
    """
    Load `meta` from a saved tau file produced during training.
    Preference: tau_star.pt → any tau_epoch_*.pt → any tau_step_*.pt
    """
    star = os.path.join(save_path, "tau_star.pt")
    if os.path.exists(star):
        obj = safe_torch_load(star, map_location="cpu")
        if isinstance(obj, dict) and "meta" in obj:
            logger.info("[Meta] Loaded meta from tau_star.pt")
            return obj["meta"]

    epoch_dir = os.path.join(save_path, "tau_epoch")
    if os.path.isdir(epoch_dir):
        for f in sorted(os.listdir(epoch_dir)):
            if f.endswith(".pt"):
                obj = safe_torch_load(os.path.join(epoch_dir, f), map_location="cpu")
                if isinstance(obj, dict) and "meta" in obj:
                    logger.info(f"[Meta] Loaded meta from {os.path.join(epoch_dir, f)}")
                    return obj["meta"]

    step_dir = os.path.join(save_path, "tau_early")
    if os.path.isdir(step_dir):
        for f in sorted(os.listdir(step_dir)):
            if f.endswith(".pt"):
                obj = safe_torch_load(os.path.join(step_dir, f), map_location="cpu")
                if isinstance(obj, dict) and "meta" in obj:
                    logger.info(f"[Meta] Loaded meta from {os.path.join(step_dir, f)}")
                    return obj["meta"]

    raise FileNotFoundError("Could not find any tau_*.pt from which to load meta.")
