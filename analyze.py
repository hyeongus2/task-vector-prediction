# analyze.py
import argparse
from pathlib import Path
import yaml
import logging
from src.tvp.analyzer import analyze
from src.tvp.utils import set_seed

# It's good practice to have a logger, even in the entrypoint script.
logger = logging.getLogger(__name__)

# Try to import IPEX, warn if not available
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    logger.warning("Intel Extension for PyTorch (IPEX) not found. XPU device will not be available.")
    pass


def main():
    """
    Entrypoint for the analysis script.
    This function handles command-line argument parsing, loads the experiment
    config, and then calls the main analysis engine.
    """
    parser = argparse.ArgumentParser(description="Run an analysis of a completed training run.")
    
    # Required arguments
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to the experiment output directory.")
    
    # Arguments for the analysis itself
    parser.add_argument(
        "--prediction_space", 
        type=str, 
        choices=['adapter', 'operational'], 
        default='adapter', 
        help="The space in which to predict the trajectory ('adapter' or 'operational'). Defaults to 'adapter'."
    )
    parser.add_argument("--k", type=int, default=3, help="Number of exponential terms (k) for the trajectory model.")
    parser.add_argument("--N", type=int, default=6, help="Number of early data points (N) to use for fitting.")
    # Optional overrides for analysis hyperparameters (helps quick experiments without editing config)
    parser.add_argument("--num_trials", type=int, default=None, help="Override config.analysis.num_trials")
    parser.add_argument("--num_alternating_steps", type=int, default=None, help="Override config.analysis.num_alternating_steps")
    parser.add_argument("--num_r_steps_per_alternation", type=int, default=None, help="Override config.analysis.num_r_steps_per_alternation")
    parser.add_argument("--num_final_candidates", type=int, default=None, help="Override config.analysis.num_final_candidates")
    parser.add_argument("--lr_r", type=float, default=None, help="Override config.analysis.lr_r")
    
    args = parser.parse_args()

    if args.N <= args.k:
        logger.error(f"Number of data points (N={args.N}) must be greater than k={args.k} for fitting."); return

    try:
        # Load the configuration file from the experiment directory
        config_path = Path(args.exp_dir) / "effective_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {config_path}")
        logger.error("Please ensure the --exp_dir argument points to a valid experiment output directory.")
        return

    set_seed(config.get('seed', 42))

    # Apply CLI overrides to config.analysis if provided
    config.setdefault('analysis', {})
    if args.num_trials is not None:
        config['analysis']['num_trials'] = args.num_trials
    if args.num_alternating_steps is not None:
        config['analysis']['num_alternating_steps'] = args.num_alternating_steps
    if args.num_r_steps_per_alternation is not None:
        config['analysis']['num_r_steps_per_alternation'] = args.num_r_steps_per_alternation
    if args.num_final_candidates is not None:
        config['analysis']['num_final_candidates'] = args.num_final_candidates
    if args.lr_r is not None:
        config['analysis']['lr_r'] = args.lr_r

    # Call the main analysis engine, passing both args and config
    analyze(args=args, config=config)


if __name__ == "__main__":
    main()