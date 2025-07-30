# utils/paths.py

import os

def get_save_path(config_path: str, overrides: dict = {}, base_dir: str = "checkpoints") -> str:
    """
    Generate experiment save directory based on config filename and optional overrides.

    Args:
        config_path (str): Path to config file, e.g., "configs/vittiny_cifar10_sgd.yaml"
        overrides (dict): Dict of override args, e.g., {"epochs": 10, "lr": 1e-3}
        base_dir (str): Root save directory (default: "checkpoints")

    Returns:
        str: Final experiment directory, e.g., "checkpoints/vitbase_cifar10_sgd_epochs10_lr0.001"
    """
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(config_path))[0]

    if overrides:
        # Sort keys for deterministic name
        suffix = "_" + "_".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        exp_name = base_name + suffix
    else:
        exp_name = base_name

    return os.path.join(base_dir, exp_name)
