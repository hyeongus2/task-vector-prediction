# analyze.py
# Thin entry point that loads config, initializes the logger, determines modes,
# and delegates to analyzers.tau_analysis.run_analysis.

import os
from typing import Any, Optional

from utils.config_utils import parse_args, load_config, merge_config
from utils.paths import get_save_path
from utils.logger import init_logger
from analyzers.tau_analysis import run_analysis

VALID_MODES = {"epoch", "step"}
TAU_EPOCH_DIR = "tau_epoch"
TAU_EARLY_DIR = "tau_early"

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

    try:
        # 4) Determine which modes to analyze
        az_config: dict[str, Any] = config.get("analyze", {})
        modes: Optional[list[str]] = az_config.get("modes")

        if modes is None:
            # Auto-detect modes if not specified in config
            modes = []
            if os.path.isdir(os.path.join(save_path, TAU_EPOCH_DIR)):
                modes.append("epoch")
            if os.path.isdir(os.path.join(save_path, TAU_EARLY_DIR)):
                modes.append("step")
        else:
            # Ensure modes from config are valid
            if isinstance(modes, str):
                modes = [modes]
            # Filter the list to only include valid, known modes
            modes = [m for m in modes if m in VALID_MODES]

        if not modes:
            logger.warning(f"[Skip] No analyzable directories found in {save_path}. Searched for: {TAU_EPOCH_DIR}, {TAU_EARLY_DIR}. Exiting.")
            return

        logger.info(f"[Analyze] save_path='{save_path}' modes={modes}")

        # 5) Run analysis
        run_analysis(config, modes, logger)

    finally:
        # 6) Finish
        if logger:
            logger.finish_wandb()


if __name__ == "__main__":
    main()
