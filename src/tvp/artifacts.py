"""Validation shared by training resume and trajectory sampling."""
from bisect import bisect_left
from pathlib import Path


def resolve_run_directory(group_dir: Path, run_id: str, resume: bool) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    output_dir = group_dir / run_id
    if resume:
        required = ("checkpoint.pt", "theta0.pt", "text_features.pt", "effective_config.yaml")
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot resume {output_dir}; missing artifacts: {', '.join(missing)}")
    elif output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    return output_dir


def select_fitting_indices(steps, count: int, save_interval: int, multiplier: int):
    """Select distinct recorded steps at or after requested positive targets."""
    if count <= 0 or save_interval <= 0 or multiplier <= 0:
        raise ValueError("count, save_interval and multiplier must be positive")
    if not steps or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Recorded steps must be nonempty and strictly increasing")
    targets = [save_interval * multiplier * i for i in range(1, count + 1)]
    indices = [bisect_left(steps, target) for target in targets]
    if indices[-1] >= len(steps):
        raise ValueError(f"Need recorded steps through {targets[-1]}, but last available step is {steps[-1]}; reduce N or collect more checkpoints")
    if len(set(indices)) != len(indices):
        raise ValueError("Checkpoint gaps select duplicate fitting points; reduce N or adjust the save interval")
    return indices
