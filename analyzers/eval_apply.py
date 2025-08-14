# analyzers/eval_apply.py
# Evaluation pipeline to test whether adding tau_pred to the backbone (with the
# final head weights) improves validation metrics. Includes BN recalibration.

import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from models import build_model
from models.utils import is_head_param  # single source of truth
from data import get_datasets


def get_input_size(model: nn.Module, config: dict) -> tuple:
    """
    Infers input size per model type. Mirrors logic from train.py to ensure
    we apply the same preprocessing for evaluation.
    """
    model_type = config['model']['type']
    if model_type == 'image':
        if hasattr(model, 'default_cfg') and 'input_size' in model.default_cfg:
            return model.default_cfg['input_size']
        first_layer = next(model.children())
        if isinstance(first_layer, nn.Conv2d):
            return (3, 224, 224)
        elif isinstance(first_layer, nn.Linear):
            return (1, 1, first_layer.in_features)
        return (3, 224, 224)
    elif model_type == 'text':
        return (512,)
    elif model_type == 'tabular':
        first_layer = next(model.children())
        if isinstance(first_layer, nn.Linear):
            return (first_layer.in_features,)
        elif isinstance(first_layer, nn.Sequential):
            first_layer = next(first_layer.children())
            if isinstance(first_layer, nn.Linear):
                return (first_layer.in_features,)
        else:
            raise ValueError("Unsupported first layer type for tabular model")
    elif model_type == 'synthetic':
        return (1,)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def build_eval_loader(config: dict, model: nn.Module) -> DataLoader:
    """Builds a test/eval DataLoader, applying the same basic transforms as train.py."""
    train_dataset, test_dataset = get_datasets(config)
    model_type = config["model"]["type"]
    input_size = get_input_size(model, config)

    if model_type == "image":
        in_channels = input_size[0]
        resize = input_size[1:]
        tfm = [transforms.Resize(resize)]
        if hasattr(train_dataset[0][0], 'mode') and train_dataset[0][0].mode == 'L' and in_channels == 3:
            tfm.append(transforms.Grayscale(3))
        tfm.append(transforms.ToTensor())
        transform = transforms.Compose(tfm)
        train_dataset.transform = transform
        test_dataset.transform = transform

    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = torch.cuda.is_available()
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


def _load_state_dict_maybe_nested(obj):
    """
    Extract a state_dict from common checkpoint structures:
      - {'state_dict': ...}, {'model_state_dict': ...}, etc.
      - Or a plain mapping of tensor parameters.
    """
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "weights", "ema_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    elif isinstance(obj, nn.Module):
        return obj.state_dict()
    raise RuntimeError("Unsupported checkpoint format for state_dict extraction")


def overwrite_head_from_ckpt(model: nn.Module, ckpt_path: str, device: str, logger) -> None:
    """Copy only head parameters from a 'final' finetuned checkpoint into the current model."""
    try:
        src = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        src = torch.load(ckpt_path, map_location=device)
    sd_src = _load_state_dict_maybe_nested(src)
    sd_tgt = model.state_dict()
    cnt = 0
    for k, v in sd_src.items():
        if is_head_param(k) and k in sd_tgt and sd_tgt[k].shape == v.shape:
            sd_tgt[k] = v.to(sd_tgt[k].device)
            cnt += 1
    model.load_state_dict(sd_tgt, strict=False)
    logger.info(f"[Head] Copied {cnt} head tensors from {ckpt_path}")


def add_tau_to_backbone_only(model: nn.Module, tau_full: torch.Tensor):
    """
    Add tau (flattened full vector) to the backbone parameters only,
    leaving the head parameters unchanged.
    """
    flat = parameters_to_vector(model.parameters()).detach()
    assert tau_full.numel() == flat.numel(), f"tau_pred size mismatch: {tau_full.numel()} vs {flat.numel()}"
    new_flat = flat.clone()

    idx = 0
    for name, p in model.named_parameters():
        n = p.numel()
        sl = slice(idx, idx + n)
        # Skip head params
        if not is_head_param(name):
            new_flat[sl] = new_flat[sl] + tau_full[sl].to(new_flat.device)
        idx += n
    vector_to_parameters(new_flat, model.parameters())


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader: DataLoader, device: str, num_batches: int = 100):
    """
    Recompute batch-norm statistics with a short pass of data.
    Useful for ResNet-style models after modifying the backbone.
    """
    has_bn = any(isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) for m in model.modules())
    if not has_bn:
        return
    was_training = model.training
    model.train()
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        model(x)
        seen += 1
        if seen >= num_batches:
            break
    model.train(was_training)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """Return (top-1 accuracy, average CE loss) on the provided DataLoader."""
    model.eval()
    crit = nn.CrossEntropyLoss().to(device)
    n, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return (correct / max(1, n)), (total_loss / max(1, n))


def eval_with_final_head_and_backbone_tau(config: dict,
                                          tau_pred: torch.Tensor,
                                          logger,
                                          theta0_ckpt: str,
                                          final_head_ckpt: str,
                                          alphas: list[float] = [1.0]):
    """
    Evaluation routine:
      1) Build model and eval DataLoader
      2) Load theta0 weights
      3) Overwrite head with final-head weights
      4) Sweep alpha and apply alpha * tau_pred to backbone
      5) Report best alpha by accuracy
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build base model and loader
    model = build_model(config).to(device)
    loader = build_eval_loader(config, model)

    # Load theta0 into model
    try:
        base_obj = torch.load(theta0_ckpt, map_location=device, weights_only=False)
    except TypeError:
        base_obj = torch.load(theta0_ckpt, map_location=device)
    base_sd = _load_state_dict_maybe_nested(base_obj)
    model.load_state_dict(base_sd, strict=True)

    # Overwrite only head with the final finetuned head
    overwrite_head_from_ckpt(model, final_head_ckpt, device, logger)

    # Baseline (alpha=0)
    base_acc, base_loss = evaluate_model(model, loader, device)
    logger.info(f"[Eval] Baseline (alpha=0.0, final head only): acc={base_acc:.4f}, loss={base_loss:.4f}")

    # Alpha sweep
    best = {"alpha": 0.0, "acc": base_acc, "loss": base_loss}
    for a in alphas:
        # Reset from theta0 for a fair comparison each time
        model.load_state_dict(base_sd, strict=True)
        overwrite_head_from_ckpt(model, final_head_ckpt, device, logger)
        add_tau_to_backbone_only(model, a * tau_pred.to(device))
        recalibrate_bn(model, loader, device, num_batches=100)  # no-op if no BN

        acc, loss = evaluate_model(model, loader, device)
        logger.info(f"[Eval] alpha={a:.3f} → acc={acc:.4f}, loss={loss:.4f}")
        if acc > best["acc"]:
            best = {"alpha": a, "acc": acc, "loss": loss}

    logger.info(f"[Eval] BEST alpha={best['alpha']:.3f} → acc={best['acc']:.4f}, loss={best['loss']:.4f}")
