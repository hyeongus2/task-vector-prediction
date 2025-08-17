# analyzers/eval_apply.py
# Evaluate adding a predicted task vector (tau_pred) to the backbone only,
# while reusing BaseTrainer.eval() to avoid duplicating evaluation code.
#
# Pipeline:
#   1) Build pretrained model (theta0) from config
#   2) Prepare train/test DataLoaders exactly like training
#   3) Create BaseTrainer(model, loaders, logger)  ← we'll only call trainer.eval()
#   4) Overwrite classifier head from one or more finetuned checkpoints
#   5) For each alpha: add alpha * tau_pred to backbone (skip head), optional BN recalibration
#   6) Call trainer.eval() and pick the best (head, alpha) by accuracy
#
# Notes:
# - We never call trainer.train(); optimizer exists but is unused here.
# - If your criterion is MSE (regression), accuracy will be None; we pick by lower loss.

import os
import re
import torch
import torch.nn as nn
from typing import Iterable, Optional
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import transforms

from models import build_model, get_input_size
from models.model_utils import is_head_param
from data import get_datasets
from trainers.base_trainer import BaseTrainer
from utils.tau_utils import safe_torch_load

# ---------------------------
# Data loaders (aligned with train)
# ---------------------------

def build_eval_loaders(config: dict, model: nn.Module) -> tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders with preprocessing consistent with training."""
    train_dataset, test_dataset = get_datasets(config)
    model_type = config["model"]["type"]
    input_size = get_input_size(model, config)

    if model_type == "image":
        c, h, w = input_size
        tfm = [transforms.Resize((h, w))]
        # If dataset sample is grayscale but the model expects 3 channels, expand to 3ch
        sample = train_dataset[0][0]
        if hasattr(sample, "mode") and sample.mode == "L" and c == 3:
            tfm.append(transforms.Grayscale(3))
        tfm.append(transforms.ToTensor())
        transform = transforms.Compose(tfm)
        train_dataset.transform = transform
        test_dataset.transform = transform

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader


# ---------------------------
# Checkpoint helpers
# ---------------------------

def _extract_state_dict(obj):
    """Extract a state_dict from common checkpoint layouts."""
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
    """
    Heuristic priority for head checkpoints:
      'best' > 'last' > 'epochNNN' > 'stepNNN' > others (higher is better)
    """
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
    Recursively search for *.pt / *.pth under save_path.
    Sort by (priority desc, mtime desc).
    """
    cand = []
    for root, _, files in os.walk(save_path):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth"):
                full = os.path.join(root, f)
                pri = _score_filename_priority(f)
                mtime = os.path.getmtime(full)
                cand.append((pri, mtime, full))
    cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, __, p in cand]


def overwrite_head_from_ckpt(model: nn.Module, ckpt_path: str, device: str, logger) -> int:
    """
    Copy only head parameters from a finetuned checkpoint into the current model.
    Returns the number of tensors actually copied.
    """
    obj = safe_torch_load(ckpt_path, map_location=device)
    sd_src = _extract_state_dict(obj)
    sd_tgt = model.state_dict()
    copied = 0
    for k, v in sd_src.items():
        if is_head_param(k) and k in sd_tgt and sd_tgt[k].shape == v.shape:
            sd_tgt[k] = v.to(sd_tgt[k].device)
            copied += 1
    model.load_state_dict(sd_tgt, strict=False)
    logger.info(f"[Head] Copied {copied} head tensors from {ckpt_path}")
    return copied


# ---------------------------
# Tau application (backbone only)
# ---------------------------

def add_tau_to_backbone_only(model: nn.Module, tau_full: torch.Tensor):
    """
    Add tau (flattened full vector) to backbone parameters only (skip head).
    Assumes same flatten order as parameters_to_vector(model.parameters()).
    """
    flat = parameters_to_vector(model.parameters()).detach()
    assert tau_full.numel() == flat.numel(), f"tau_pred length mismatch: {tau_full.numel()} vs {flat.numel()}"
    new_flat = flat.clone()

    idx = 0
    for name, p in model.named_parameters():
        n = p.numel()
        sl = slice(idx, idx + n)
        if not is_head_param(name):
            new_flat[sl] = new_flat[sl] + tau_full[sl].to(new_flat.device)
        idx += n

    vector_to_parameters(new_flat, model.parameters())


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader: DataLoader, device: str, num_batches: int = 100):
    """
    Recompute BatchNorm running stats with a few forward passes.
    No-op if there are no BN layers.
    """
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)) for m in model.modules())
    if not has_bn:
        return
    was_training = model.training
    model.train()  # BN updates only in train mode
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        model(x)
        seen += 1
        if seen >= max(1, num_batches):
            break
    model.train(was_training)


# ---------------------------
# Public API (reuses BaseTrainer.eval)
# ---------------------------

def eval_with_backbone_tau_using_trainer(
    config: dict,
    tau_pred: torch.Tensor,
    logger,
    alphas: Iterable[float] = (1.0,),
    head_ckpts: Optional[list[str]] = None,
    bn_recalc_batches: int = 100,
):
    """
    Full evaluation using BaseTrainer.eval():
      - Build pretrained model and train/test loaders
      - Instantiate BaseTrainer(model, loaders, logger)
      - For each head_ckpt (auto-discovered if None) and each alpha:
          * Reset model to pretrained state
          * Overwrite head from ckpt
          * Add alpha * tau_pred to backbone
          * (Optional) Recompute BN stats with a few batches
          * Call trainer.eval() → (acc, loss)
      - Report/broadcast the best (head, alpha) by accuracy if classification, else by loss
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build theta0 (pretrained) model and loaders
    model = build_model(config).to(device)
    train_loader, test_loader = build_eval_loaders(config, model)

    # 2) BaseTrainer (we will only call eval(); no training here)
    trainer = BaseTrainer(config, model, train_loader, test_loader, logger)

    # 3) Resolve head checkpoints
    if not head_ckpts:
        save_path = config["save"]["path"]
        head_ckpts = find_head_checkpoints(save_path)
        if not head_ckpts:
            logger.warning("[Eval] No head checkpoints found under save_path; aborting.")
            return

    # 4) Cache base (pretrained) state_dict to reset before every trial
    base_sd = trainer.model.state_dict()

    # 5) Quick sanity check on tau length
    base_flat_len = parameters_to_vector(trainer.model.parameters()).numel()
    if tau_pred.numel() != base_flat_len:
        logger.warning(
            f"[Eval] tau_pred length {tau_pred.numel()} != model params length {base_flat_len}. "
            "Evaluation will likely fail."
        )

    # 6) Sweep (head, alpha)
    best = {"acc": -1.0, "loss": float("inf"), "alpha": 0.0, "head": None}
    alphas = list(alphas)

    for ckpt in head_ckpts:
        try:
            # Reset model → theta0
            trainer.model.load_state_dict(base_sd, strict=True)

            # Overwrite only head from finetuned checkpoint
            copied = overwrite_head_from_ckpt(trainer.model, ckpt, device, logger)
            if copied == 0:
                logger.warning(f"[Eval] No matching head tensors in {ckpt}; skip.")
                continue

            # Baseline: alpha=0 (final head only)
            base_acc, base_loss = trainer.eval(epoch=None)
            logger.info(f"[Eval] HEAD='{os.path.basename(ckpt)}' | alpha=0.0 → acc={base_acc}, loss={base_loss:.4f}")
            # Pick winner depending on task type
            if base_acc is not None:
                # classification: higher acc is better
                if base_acc > best["acc"]:
                    best.update({"acc": base_acc, "loss": base_loss, "alpha": 0.0, "head": ckpt})
            else:
                # regression: lower loss is better
                if base_loss < best["loss"]:
                    best.update({"acc": -1.0, "loss": base_loss, "alpha": 0.0, "head": ckpt})

            # Sweep alphas
            for a in alphas:
                trainer.model.load_state_dict(base_sd, strict=True)
                overwrite_head_from_ckpt(trainer.model, ckpt, device, logger)
                add_tau_to_backbone_only(trainer.model, a * tau_pred.to(device))
                recalibrate_bn(trainer.model, trainer.test_loader, device, num_batches=bn_recalc_batches)

                acc, loss = trainer.eval(epoch=None)
                logger.info(
                    f"[Eval] HEAD='{os.path.basename(ckpt)}' | alpha={a:.4f} → acc={acc}, loss={loss:.4f}"
                )

                if acc is not None:
                    if acc > best["acc"]:
                        best.update({"acc": acc, "loss": loss, "alpha": a, "head": ckpt})
                else:
                    if loss < best["loss"]:
                        best.update({"acc": -1.0, "loss": loss, "alpha": a, "head": ckpt})

        except Exception as e:
            logger.warning(f"[Eval] Failed on head '{ckpt}': {e}")

    # 7) Final report (+ optional W&B scalars)
    if best["head"] is not None:
        logger.info(
            f"[Eval] BEST: head='{os.path.basename(best['head'])}', alpha={best['alpha']:.4f} "
            f"→ acc={best['acc']}, loss={best['loss']:.4f}"
        )
        try:
            logger.log_wandb_scalar(
                {"eval/best_acc": best["acc"] if best["acc"] is not None else -1.0,
                 "eval/best_loss": best["loss"],
                 "eval/best_alpha": best["alpha"]}
            )
        except Exception:
            pass
    else:
        logger.warning("[Eval] No valid (head, alpha) produced a result.")
