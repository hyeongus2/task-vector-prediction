# analyze.py
# Thin entry point that loads config, initializes the logger, determines modes,
# and delegates to analyzers.tau_analysis.run_analysis.

import os

from utils.config_utils import parse_args, load_config, merge_config
from utils.paths import get_save_path
from utils.logger import init_logger
from analyzers.tau_analysis import run_analysis


def main():
    # 1) Load config and CLI overrides
    config_path, override_dict = parse_args()
    config = load_config(config_path)
    config = merge_config(config, override_dict)

    # 2) Resolve save path and keep it in config for consistency/logging
    save_path = config["save"].get("path", get_save_path(config_path=config_path, overrides=override_dict))
    config["save"]["path"] = save_path

    # 3) Initialize logger (wandb + file + console)
    logger = init_logger(config)

    # 4) Determine which modes to analyze
    az_cfg = config.get("analyze", {})
    modes = az_cfg.get("modes")
    if modes is None:
        modes = []
        if os.path.isdir(os.path.join(save_path, "tau_epoch")):
            modes.append("epoch")
        if os.path.isdir(os.path.join(save_path, "tau_early")):
            modes.append("step")
    else:
        if isinstance(modes, str):
            modes = [modes]
        modes = [m for m in modes if m in {"epoch", "step"}]

    if not modes:
        logger.warning("[Skip] No analyzable directories found (tau_epoch/tau_early). Exiting.")
        logger.finish_wandb()
        return

    logger.info(f"[Analyze] save_path={save_path} modes={modes}")

    # 5) Run analysis
    run_analysis(config, modes, logger)

    # 6) Finish
    logger.finish_wandb()


if __name__ == "__main__":
    main()
