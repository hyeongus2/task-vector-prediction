# src/tvp/utils.py
import torch
import numpy as np
import random
import logging
import wandb
from pathlib import Path
from typing import Union, Any, Dict, Optional

# Create a logger for this module
logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, log_filename: str, enabled: bool = True):
    """
    Sets up the root logger to output to both a file and the console.

    Args:
        log_dir (Path): The directory where the log file will be saved.
        log_filename (str): The name of the log file (e.g., "train.log").
        enabled (bool): If False, disables logging by adding a NullHandler.
    """
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    if enabled:
        root_logger.setLevel(logging.INFO)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_filename
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        logger.info(f"Logging enabled. Logs will be saved to {log_file_path}")
    else:
        root_logger.addHandler(logging.NullHandler())


def get_device() -> torch.device:
    """
    Automatically detects and returns the available accelerator device (CUDA, XPU).

    Returns:
        torch.device: The detected device object ('cuda', 'xpu', or 'cpu').
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA (NVIDIA GPU): {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        logger.info("Using XPU (Intel GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def set_seed(seed: int = 42):
    """
    Sets the seed for random, numpy, and torch to ensure experiment reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def save_torch(obj: Any, path: Union[str, Path]):
    """
    Saves a PyTorch-related object to the specified path.

    Args:
        obj (Any): The Python object to save.
        path (Union[str, Path]): The destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    logger.debug(f"Saved object to {path}")


def load_torch(path: Union[str, Path]) -> Any:
    """
    Loads a PyTorch-related object from the specified path to the CPU.

    Args:
        path (Union[str, Path]): The source file path.

    Returns:
        Any: The loaded Python object.
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found at {path}")
        raise FileNotFoundError(f"File not found at {path}")
    
    obj = torch.load(path, map_location='cpu')
    logger.debug(f"Loaded object from {path}")
    return obj


def convert_to_operational_tau(adapter_tau: Dict) -> Dict:
    """
    Converts a LoRA adapter-space tau (A, B matrices) into an 
    operational-space tau (effective delta_W) by computing B@A.
    If not a LoRA tau, returns it as is.

    Args:
        adapter_tau (Dict): The task vector in adapter space (from a saved file).

    Returns:
        Dict: The task vector in operational space.
    """    
    is_lora = any('lora_A' in k for k in adapter_tau.keys())
    if not is_lora:
        return adapter_tau
    

    logger.debug("LoRA tau detected. Converting to operational tau (B@A)...")
    operational_tau = {}
    lora_b_keys = [k for k in adapter_tau if 'lora_B' in k]
    
    with torch.no_grad():
        for b_key in lora_b_keys:
            a_key = b_key.replace('lora_B', 'lora_A')
            # This logic infers the original weight matrix's name from the adapter key name.
            # e.g., '...q_proj.lora_B.weight' -> '...q_proj.weight'
            original_key = b_key.replace('lora_B.weight', 'base_layer.weight')
            original_key = original_key.replace('base_model', 'vision_model.base_model')
            
            lora_b = adapter_tau[b_key]
            lora_a = adapter_tau[a_key]
            delta_w = lora_b @ lora_a
            operational_tau[original_key] = delta_w
            
    return operational_tau


def calculate_task_vector(model: torch.nn.Module, theta0_state_dict: Dict) -> Dict:
    """
    Calculates the adapter-space task vector (tau) for the trainable 
    parameters of a model. For LoRA, this is the state of the adapter itself.

    Args:
        model (torch.nn.Module): The current trained model (theta_t), potentially a PeftModel.
        theta0_state_dict (Dict): The initial state_dict of the model (theta0 on CPU).

    Returns:
        Dict: The adapter-space task vector (tau_t), ready for saving.
    """
    trainable_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    
    # For PEFT models, the task vector is simply the state of the trainable adapter.
    if any("lora" in key for key in trainable_keys):
        from peft import get_peft_model_state_dict
        # Pass the top-level model to let peft find the adapter state dict automatically.
        # This is more robust than hardcoding a submodule like 'model.vision_model'.
        return get_peft_model_state_dict(model.vision_model, unwrap_compiled=True)

    # For full finetuning, we calculate the difference.
    theta_t_state_dict = model.state_dict()
    task_vector = {}
    for key in trainable_keys:
        if key in theta0_state_dict:
            task_vector[key] = theta_t_state_dict[key].cpu() - theta0_state_dict[key]
        else:
            logger.warning(f"Key '{key}' not found in theta0_state_dict. Skipping.")
    return task_vector


def unflatten_to_state_dict(flat_tensor: torch.Tensor, reference_dict: Dict) -> Dict:
    """
    Converts a flat 1D tensor back into a state dictionary structure,
    matching the keys and shapes of a reference dictionary.

    Args:
        flat_tensor (torch.Tensor): The 1D tensor to be un-flattened.
        reference_dict (Dict): A state dictionary with the target structure.

    Returns:
        Dict: A new state dictionary with the data from the flat tensor.
    """
    new_dict = {}
    current_pos = 0
    for key, tensor in reference_dict.items():
        num_elements = tensor.numel()
        # Use .view() which is standard for this kind of reshaping.
        param_values = flat_tensor[current_pos : current_pos + num_elements].view(tensor.shape)
        new_dict[key] = param_values
        current_pos += num_elements
    
    if current_pos != flat_tensor.numel():
        raise ValueError("The number of elements in the flat tensor does not match the reference dictionary.")
        
    return new_dict


def reconstruct_theta(theta0_dict: Dict, operational_tau_dict: Dict) -> Dict:
    """
    DEPRECATED for LoRA. This function performs a simple addition of two state_dicts.
    It is kept for potential use with full finetuning experiments.
    For LoRA models, use PEFT's merge_and_unload() method instead.
    """
    final_dict = {k: v.clone() for k, v in theta0_dict.items()}
    for key, delta_w in operational_tau_dict.items():
        if key in final_dict:
            final_dict[key].add_(delta_w)
        else:
            # This case can happen with LoRA if not handled properly, but we will avoid it.
            logger.error(f"Key '{key}' from tau_dict not found in theta0_dict. Adding it directly.")
            raise KeyError(f"Key '{key}' from tau_dict not found in theta0_dict.")
    return final_dict


def log_tau_elements_to_wandb(
    tau_dict_actual: Optional[Dict],
    tau_dict_predicted: Optional[Dict],
    global_step: int,
    tracked_indices: Optional[torch.Tensor],
    config: Dict
) -> Optional[torch.Tensor]:
    """
    Logs trajectories, norms, and histograms of actual and/or predicted tau vectors to Wandb.
    When both are provided, it logs element trajectories on the same overlayed chart.

    Args:
        tau_dict_actual (Optional[Dict]): The actual, observed task vector (tau_t).
                                           Can be a zero-filled dict for the t=0 step.
        tau_dict_predicted (Optional[Dict]): The predicted task vector (tau_hat_t).
                                              Can be None.
        global_step (int): The current training/analysis step (x-axis for the plot).
        tracked_indices (Optional[torch.Tensor]): Indices of elements to track.
                                                      If None, new indices will be sampled on the first valid call.
        config (Dict): The experiment configuration dictionary.

    Returns:
        Optional[torch.Tensor]: The (potentially newly created) tracked indices, which should be
                                passed back into this function on the next call.
    """
    if not config['logging']['wandb']:
        return None

    log_dict = {}

    # --- Prepare Data ---
    flat_actual_tau = None
    if tau_dict_actual is not None:
        op_actual = convert_to_operational_tau(tau_dict_actual)
        if op_actual: 
            flat_actual_tau = torch.cat([p.flatten() for p in op_actual.values()])

    flat_predicted_tau = None
    if tau_dict_predicted is not None:
        op_predicted = convert_to_operational_tau(tau_dict_predicted)
        if op_predicted:
            flat_predicted_tau = torch.cat([p.flatten() for p in op_predicted.values()])

    # --- Sample Indices ---
    tau_for_sampling = flat_actual_tau if flat_actual_tau is not None else flat_predicted_tau
    if tracked_indices is None and (tau_for_sampling is not None and tau_for_sampling.numel() > 0):
        # Use different num_elements for analysis vs. training
        if tau_dict_predicted is not None:
            num_elements = config['analysis'].get('num_analysis_elements', 50)
        else:
            num_elements = config['analysis'].get('num_monitoring_elements', 20)
        
        logger.info(f"Sampling {num_elements} indices from operational tau for trajectory logging.")
        tracked_indices = torch.linspace(0, tau_for_sampling.numel() - 1, num_elements, dtype=torch.long)

    # --- Log Metrics ---
    if tracked_indices is not None:
        num_digits = len(str(len(tracked_indices) - 1))
        for i, idx in enumerate(tracked_indices):
            log_key_group = f"element_trajectories/element_{i:0{num_digits}d}"
            element_log = {}
            if flat_actual_tau is not None: element_log["actual"] = flat_actual_tau[idx].item()
            if flat_predicted_tau is not None: element_log["predicted"] = flat_predicted_tau[idx].item()
            if element_log: log_dict[log_key_group] = element_log
    
    # Increase the number of bins for better resolution
    num_histogram_bins = 512

    if flat_actual_tau is not None:
        log_dict["tau_norm/actual_L2"] = torch.linalg.vector_norm(flat_actual_tau).item()
        log_dict["tau_hist/actual"] = wandb.Histogram(
            flat_actual_tau.cpu().tolist(), num_bins=num_histogram_bins
        )

    if flat_predicted_tau is not None:
        log_dict["tau_norm/predicted_L2"] = torch.linalg.vector_norm(flat_predicted_tau).item()
        log_dict["tau_hist/predicted"] = wandb.Histogram(
            flat_predicted_tau.cpu().tolist(), num_bins=num_histogram_bins
        )

    # --- Send to Wandb ---
    if log_dict:
        wandb.log(log_dict, step=global_step)
        
    return tracked_indices