# src/tvp/analyzer.py
import argparse
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from peft import set_peft_model_state_dict

# Import from our local modules
from . import utils, data_loader, model as model_loader, plotting, predictor
from .trainer import evaluate

# Create a logger for this module
logger = logging.getLogger(__name__)


# --- Hierarchical Configuration Dictionary ---
# Define baseline parameters for each (optimizer, space) combination.
FITTING_CONFIGS = {
    'adamw': {
        'adapter': {
            'r_init_low': 1e-3, 'r_init_high': 1e-1,
            'r_reg_lambda_base': 1e-4,
            'A_reg_lambda_base': 1e-4,
            'score_A_norm_weight_base': 10.0
        },
        'operational': {
            'r_init_low': 1e-3, 'r_init_high': 1e-1,
            'r_reg_lambda_base': 1e-7,
            'A_reg_lambda_base': 1e-1, # Much stronger baseline for operational
            'score_A_norm_weight_base': 1.0 # Less reliance on w due to strong pre-regulation
        }
    },
    'mmt': {
        'adapter': {
            'r_init_low': 3e-4, 'r_init_high': 3e-2,
            'r_reg_lambda_base': 1e-4,
            'A_reg_lambda_base': 5e-3, # Less regularization needed than AdamW
            'score_A_norm_weight_base': 5.0
        },
        'operational': {
            'r_init_low': 3e-4, 'r_init_high': 3e-2,
            'r_reg_lambda_base': 1e-7,
            'A_reg_lambda_base': 1e-1,
            'score_A_norm_weight_base': 1.0
        }
    },
    'sgd': { # Example for SGD without momentum (slower)
        'adapter': {
            'r_init_low': 5e-5, 'r_init_high': 5e-3,
            'r_reg_lambda_base': 1e-4,
            'A_reg_lambda_base': 5e-4,
            'score_A_norm_weight_base': 5.0
        },
        'operational': {
            'r_init_low': 5e-5, 'r_init_high': 5e-3,
            'r_reg_lambda_base': 1e-7,
            'A_reg_lambda_base': 5e-2,
            'score_A_norm_weight_base': 1.0
        }
    }
}

def get_fitting_config(optimizer: str, space: str, k: int, N: int) -> dict:
    """
    Dynamically generates the fitting configuration using a hierarchical approach.
    """
    # Step 1: Look up the base configuration for the given optimizer and space
    base_config = FITTING_CONFIGS[optimizer][space]
    r_init_low = base_config['r_init_low']
    r_init_high = base_config['r_init_high']
    r_reg_lambda_base = base_config['r_reg_lambda_base']
    A_reg_lambda_base = base_config['A_reg_lambda_base']
    score_A_norm_weight_base = base_config['score_A_norm_weight_base']

    # Step 2: Calculate dynamic multipliers for k and N
    k_multiplier = 3.0 ** (k - 3.0)
    N_multiplier = 6.0 / N
    
    # Step 3: Apply multipliers to the base values
    final_r_reg_lambda = r_reg_lambda_base * k_multiplier * N_multiplier

    # A_reg_lambda increases with k and N risk
    final_A_reg_lambda = A_reg_lambda_base * k_multiplier * N_multiplier
    
    # score_A_norm_weight increases with k and N risk (as model variety grows)
    final_score_A_norm_weight = score_A_norm_weight_base * k_multiplier * N_multiplier
    
    # But if space is operational, A_norm has less variance, so w's effect is less critical.
    # The lower base value for operational space already handles this.

    return {
        'num_trials': 20 if space == 'adapter' else 5,
        'num_final_candidates': 3,
        'num_alternating_steps': 30,
        'num_r_steps_per_alternation': 5,
        'lr_r': 1e-3,
        'r_reg_lambda': final_r_reg_lambda,
        'A_reg_lambda': final_A_reg_lambda,
        'r_init_low': r_init_low,
        'r_init_high': r_init_high,
        'score_A_norm_weight': final_score_A_norm_weight
    }


def analyze(args: argparse.Namespace, config: dict):
    """
    Main analysis engine. This function is called by the entrypoint script (analyze.py).
    It performs trajectory fitting, model evaluation, visualization, and logging.
    """
    utils.set_seed(config.get('seed', 42))

    # --- 1: Unpack arguments and prepare environment ---
    exp_dir, k, N = Path(args.exp_dir), args.k, args.N
    space = args.prediction_space
    
    # Setup logging and directories
    log_filename = f"analyze_{space}_k{k}_N{N}.log"
    utils.setup_logging(exp_dir / "logs", log_filename=log_filename, enabled=config['logging']['enabled'])
    logger.info(f"--- Starting Full Analysis Engine for: {exp_dir.name} | Space: {space}, k={k}, N={N} ---")
    
    plots_dir = exp_dir / "plots" / f"{space}_k{k}_N{N}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots will be saved to {plots_dir}")

    # --- 2: Load Artifacts & Prepare Data ---
    logger.info("Loading artifacts (theta0, text_features, tau vectors)...")
    theta0_dict = utils.load_torch(exp_dir / "theta0.pt")
    text_features = utils.load_torch(exp_dir / "text_features.pt")
    tau_star_path = exp_dir / "task_vectors" / "tau_star.pt"
    all_tau_files = sorted([p for p in exp_dir.glob("task_vectors/tau_*.pt") if "star" not in p.stem], key=lambda p: int(p.stem.split('_')[1]))

    if not all_tau_files or not tau_star_path.exists():
        logger.error("Not all required artifact files found. Exiting."); return
    
    # Load original data (without t=0) first
    all_steps_without_zero = torch.tensor([int(p.stem.split('_')[1]) for p in all_tau_files], dtype=torch.float32)
    all_adapter_taus_without_zero = [utils.load_torch(p) for p in all_tau_files]
    tau_star_adapter = utils.load_torch(tau_star_path)

    # Create a zero-filled dictionary as a placeholder for t=0
    # 1. Get a template from the first real tau vector
    tau_template = all_adapter_taus_without_zero[0]
    # 2. Create a dictionary with the same keys, but with zero tensors of the same shape
    zero_tau_dict = {key: torch.zeros_like(value) for key, value in tau_template.items()}

    # 3. Create the final lists that include the t=0 data point
    all_adapter_taus = [zero_tau_dict] + all_adapter_taus_without_zero
    all_steps = torch.cat([torch.tensor([0.0]), all_steps_without_zero])

    # ALWAYS prepare operational-space data for geometric analysis and visualization
    logger.info("Converting all taus to operational space for geometric analysis...")
    all_operational_taus = [utils.convert_to_operational_tau(t) for t in all_adapter_taus]
    tau_star_operational = utils.convert_to_operational_tau(tau_star_adapter)
    all_flat_operational_taus = torch.stack([torch.cat([p.flatten() for p in tau.values()]) for tau in all_operational_taus])
    tau_star_flat_operational = torch.cat([p.flatten() for p in tau_star_operational.values()])
    
    # --- 3: HYBRID MODEL SELECTION (Fitting, Candidate Selection) ---
    logger.info(f"Preparing data for trajectory fitting in '{space}' space...")
    if space == 'adapter':
        y_data_for_fitting = torch.stack([torch.cat([p.flatten() for p in tau.values()]) for tau in all_adapter_taus])
        reference_dict_for_unflatten = all_adapter_taus[1]
    else: # operational
        y_data_for_fitting = all_flat_operational_taus
        reference_dict_for_unflatten = all_operational_taus[1]

    d = y_data_for_fitting.shape[1]

    # --- Select N data points with optimizer-dependent spacing ---

    # 1. Determine the optimizer type to decide the sampling strategy
    optimizer = config['finetuning']['optimizer'].lower()
    if optimizer == 'sgd' and config['finetuning']['momentum'] > 0.0:
        optimizer = 'mmt' # Treat SGD with momentum as a separate case

    # 2. Define a spacing multiplier based on the optimizer's learning speed
    # The base interval between saved checkpoints is assumed to be 50 steps.
    STEP_INTERVAL = 50 
    
    if optimizer == 'sgd':
        # SGD is the slowest, so sample points very far apart to see the long-term trend.
        STEP_MULTIPLIER = 6
    elif optimizer == 'mmt':
        # SGD with momentum is moderately slow, sample with a medium gap.
        STEP_MULTIPLIER = 3
    else: # AdamW and others are fast
        # AdamW learns fast, so use the initial dense points (standard interval).
        STEP_MULTIPLIER = 1

    # 3. Create the target steps to be sampled (N points, excluding t=0)
    # The actual interval for sampling. e.g., 50 (AdamW), 150 (MMT), 250 (SGD)
    sampling_interval = STEP_INTERVAL * STEP_MULTIPLIER
    
    # e.g., if interval=150, N=6 -> [150, 300, 450, 600, 750, 900]
    target_steps = torch.arange(1, N + 1) * sampling_interval

    # 4. Find the indices in the full trajectory that are closest to our target steps
    # This robustly finds the available data points nearest to our desired sample points.
    indices = torch.searchsorted(all_steps, target_steps)

    # 5. Select the final data using the found indices
    x_data = all_steps[indices]
    y_data = y_data_for_fitting[indices]

    logger.info(f"Optimizer '{optimizer}': Sampling {len(x_data)} points with a step multiplier of {STEP_MULTIPLIER}x.")
    logger.info(f"Fitting on steps: {[int(s) for s in x_data.tolist()]}")
    logger.info(f"Fitting k={k} model on N={N} data points using Multi-start Alternating Optimization.")

    # --- START OF FITTING SECTION ---

    # 1. Hyperparameters for the fitting strategy
    fitting_config = get_fitting_config(optimizer, space, k, N)

    num_trials = config['analysis'].get('num_trials', fitting_config['num_trials']) # Number of different random initializations for r
    num_final_candidates = config['analysis'].get('num_final_candidates', fitting_config['num_final_candidates']) # Number of top candidates to fully evaluate
    num_alternating_steps = config['analysis'].get('num_alternating_steps', fitting_config['num_alternating_steps']) # Number of times to alternate between A and r
    num_r_steps_per_alternation = config['analysis'].get('num_r_steps_per_alternation', fitting_config['num_r_steps_per_alternation']) # Number of gradient steps for r in each alternation
    lr_r = config['analysis'].get('lr_r', fitting_config['lr_r']) # Learning rate for r's optimizer
    r_reg_lambda = config['analysis'].get('r_reg_lambda', fitting_config['r_reg_lambda']) # Regularization strength to keep r values small
    A_reg_lambda = config['analysis'].get('A_reg_lambda', fitting_config['A_reg_lambda']) # Regularization strength to keep all elements of A small
    r_init_low = config['analysis'].get('r_init_low', fitting_config['r_init_low']) # Sensible range for r initialization
    r_init_high = config['analysis'].get('r_init_high', fitting_config['r_init_high'])
    score_A_norm_weight = config['analysis'].get('score_A_norm_weight', fitting_config['score_A_norm_weight']) # Weight for the A_norm term in the combined score

    EPSILON = 1e-10 # Small constant for numerical stability
    def get_log_uniform_initial_r(k, low, high):
        """
        Generates initial log_r parameters for the model,
        such that the resulting r values are log-uniformly distributed between low and high.
        """
        import numpy as np
        
        # This is the most direct and correct way to implement the logic.
        # It performs the steps logically: sample in log space, convert back, then find inverse.
        r_values = torch.from_numpy(
            np.exp(np.random.uniform(np.log(low), np.log(high), k))
        ).float()
        
        # The inverse softplus with a stability guard.
        return torch.log(torch.exp(r_values) - 1 + EPSILON)
    
    def get_logspace_initial_r(k, low, high):
        """
        Generates initial log_r parameters for the model,
        with r values evenly spaced on a log scale between low and high.
        """
        import numpy as np

        # num_trials = 1
        
        # Generate k values evenly spaced on a log scale
        r_values = torch.from_numpy(
            np.exp(np.linspace(np.log(low), np.log(high), k))
        ).float()
        
        # The inverse softplus with a stability guard
        return torch.log(torch.exp(r_values) - 1 + EPSILON)

    trial_results = []
    for trial in range(num_trials):
        logger.info(f"--- Starting Trial {trial + 1}/{num_trials} ---")
        
        # Create a fresh model and optimizer for each trial
        pred_model = predictor.TrajectoryPredictor(k, d)
        
        # Initialize r with a new random set of values for this trial
        initial_log_r = get_log_uniform_initial_r(k, r_init_low, r_init_high)
        # initial_log_r = get_logspace_initial_r(k, r_init_low, r_init_high)
        pred_model.log_r.data = initial_log_r
        optimizer_r = torch.optim.AdamW([pred_model.log_r], lr=lr_r)
        total_loss = torch.tensor(float('inf'))

        for step in range(num_alternating_steps):
            # --- STEP 1: Fix r, Solve for A ---
            with torch.no_grad():
                rates = F.softplus(pred_model.log_r)
                F_matrix = 1 - torch.exp(-rates.unsqueeze(0) * x_data.unsqueeze(1))

                # Use Ridge Regression to solve for A
                # This prevents the Frobenius norm of A from becoming too large.
                FtF = F_matrix.T @ F_matrix
                FtY = F_matrix.T @ y_data
                identity = torch.eye(k, device=FtF.device)
                
                # Solves (F.T @ F + lambda_A * I) @ A = F.T @ y
                A_new = torch.linalg.solve(FtF + A_reg_lambda * identity, FtY)
                pred_model.A.data = A_new

            # --- STEP 2: Fix A, Update r ---
            for _ in range(num_r_steps_per_alternation):
                optimizer_r.zero_grad()
                predicted_y = pred_model(x_data)
                mse_loss = F.mse_loss(predicted_y, y_data)
                
                # Ridge Regularization to penalize large r values
                current_rates = F.softplus(pred_model.log_r)
                reg_loss = torch.sum(current_rates ** 2)
                
                total_loss = mse_loss + r_reg_lambda * reg_loss
                total_loss.backward()
                optimizer_r.step()

        # Final update of A for the last updated r
        # Ensure the final (A, r) pair is consistent
        logger.debug("Performing final update of A for the last r.")
        with torch.no_grad():
            rates = F.softplus(pred_model.log_r)
            F_matrix = 1 - torch.exp(-rates.unsqueeze(0) * x_data.unsqueeze(1))
            FtF = F_matrix.T @ F_matrix
            FtY = F_matrix.T @ y_data
            identity = torch.eye(k, device=FtF.device)
            A_new = torch.linalg.solve(FtF + A_reg_lambda * identity, FtY)
            pred_model.A.data = A_new
        
        # --- At the end of each trial, calculate score and log ---
        with torch.no_grad():
            final_trial_loss = total_loss.item()
            A_norm = torch.linalg.vector_norm(pred_model.A).item()
            
        trial_results.append({
            'trial_num': trial + 1,
            'loss': final_trial_loss,
            'A_norm': A_norm,
            'state_dict': {k: v.clone() for k, v in pred_model.state_dict().items()}
        })
        logger.info(f"Trial {trial + 1} finished with Loss: {final_trial_loss:.2e}, A_Norm: {A_norm:.2f}")

    # --- Select top candidates using Standardized Score ---
    if not trial_results:
        logger.error("Fitting failed. No trial results were recorded.")
        return

    # Standardize the loss and A_norm values
    losses = torch.tensor([r['loss'] for r in trial_results])
    a_norms = torch.tensor([r['A_norm'] for r in trial_results])

    z_losses = (losses - losses.mean()) / (losses.std() + EPSILON)
    z_a_norms = (a_norms - a_norms.mean()) / (a_norms.std() + EPSILON)

    # Calculate a combined score based on standardized values
    for i, r in enumerate(trial_results):
        r['score'] = z_losses[i].item() + score_A_norm_weight * z_a_norms[i].item()

    # Select the candidates with the best (lowest) scores
    trial_results.sort(key=lambda r: r['score'])
    top_candidates = trial_results[:num_final_candidates]

    logger.info(f"--- Top {num_final_candidates} Candidates Selected (using Standardized Score) ---")
    for candidate in top_candidates:
        logger.info(f"  - Trial {candidate['trial_num']}: Score={candidate['score']:.2f} (Loss: {candidate['loss']:.2e}, A_Norm: {candidate['A_norm']:.2f})")

    # --- END OF FITTING SECTION ---

    # --- 4: Evaluate top candidates to find the final best model ---
    device = utils.get_device()
    processor = model_loader.CLIPProcessor.from_pretrained(config['model_id'])
    _, val_loader, class_names = data_loader.get_dataloaders(config, processor)
    
    # Create a single, correctly structured model shell that will be reused.
    # create_model already returns a PeftModel if config specifies lora.
    # It is created on CPU to avoid holding unnecessary GPU memory.
    eval_model_shell, _ = model_loader.create_model(config, class_names, processor, device='cpu')
    is_lora = (config['finetuning']['method'] == 'lora')

    # --- Evaluate baseline model (theta_0) and the actual best model (theta_*) performance beforehand ---
    logger.info("Evaluating baseline model (theta_0)...")

    # 1. Reset shell to theta0 state
    eval_model_shell.load_state_dict(theta0_dict)
    eval_model_shell.to(device)
    loss_zero, acc_zero = evaluate(eval_model_shell, val_loader, text_features.to(device), device)
    eval_model_shell.to('cpu') # Move back to CPU to free up GPU memory
    logger.info(f"  -> Baseline (theta_0): Val Loss = {loss_zero:.4f}, Val Acc = {acc_zero:.4f}")

    logger.info("Evaluating ground truth best model (theta_*)...")
    if is_lora:
        # 2. Load the star adapter on top of theta0
        set_peft_model_state_dict(eval_model_shell.vision_model, tau_star_adapter)
    else: # Full finetuning
        # 2. For full finetuning, manually add the operational tau to theta0
        theta_star_dict = utils.reconstruct_theta(theta0_dict, tau_star_operational)
        eval_model_shell.load_state_dict(theta_star_dict)
    
    # 3. Evaluate the resulting model (PeftModel or standard model)
    eval_model_shell.to(device)
    loss_star, acc_star = evaluate(eval_model_shell, val_loader, text_features.to(device), device)
    eval_model_shell.to('cpu')
    logger.info(f"  -> Ground Truth Optimal (theta_*): Val Loss={loss_star:.4f}, Val Acc={acc_star:.4f}")

    logger.info(f"--- Evaluating Top {num_final_candidates} Candidates to Find Best Model ---")
    best_val_acc = -1.0
    best_model_final_info = {}
    for candidate in top_candidates:
        logger.info(f"Evaluating candidate from Trial {candidate['trial_num']}...")
        
        # Create a temporary model to load the candidate's state
        temp_model = predictor.TrajectoryPredictor(k, d)
        temp_model.load_state_dict(candidate['state_dict'])
        
        # Calculate tau_hat_infty for this specific candidate model
        with torch.no_grad():
            tau_hat_infty_flat = temp_model.A.sum(dim=0)
        
        # Load the task vector into the shell model for evaluation
        if space == 'adapter':
            if not is_lora: 
                logger.error(f"The space of evaluation for Trial {candidate['trial_num']} is 'adapter' but method is not 'lora'.")
                return
            
            tau_hat_infty_adapter = utils.unflatten_to_state_dict(tau_hat_infty_flat, reference_dict_for_unflatten)
            tau_hat_infty_operational = utils.convert_to_operational_tau(tau_hat_infty_adapter)
            # Reset to theta0, then load the adapter
            eval_model_shell.load_state_dict(theta0_dict)
            set_peft_model_state_dict(eval_model_shell.vision_model, tau_hat_infty_adapter)
        else: # operational
            tau_hat_infty_operational = utils.unflatten_to_state_dict(tau_hat_infty_flat, reference_dict_for_unflatten)
            # Reset to theta0, then manually reconstruct and load the full state_dict
            theta_hat_infty_dict = utils.reconstruct_theta(theta0_dict, tau_hat_infty_operational)
            eval_model_shell.load_state_dict(theta_hat_infty_dict)

        # Perform the evaluation
        eval_model_shell.to(device)
        loss, acc = evaluate(eval_model_shell, val_loader, text_features.to(device), device)
        eval_model_shell.to('cpu')
        
        # Store the loss and accuracy in the candidate's dictionary
        candidate['val_loss'] = loss
        candidate['val_acc'] = acc
        logger.info(f"  -> Candidate Trial {candidate['trial_num']}: Val Loss = {loss:.4f}, Val Acc = {acc:.4f}")
        
        # Check if this is the new best model
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_final_info = candidate
            logger.info(f"  ---> New best validation accuracy found from Trial {candidate['trial_num']}!")

    # --- FINALIZATION: Load the single best model for visualization and logging ---
    if not best_model_final_info:
        logger.error("Evaluation failed for all candidates. No final model selected.")
        return

    logger.info(f"--- Final Model Selection ---")
    logger.info(f"Selected Trial {best_model_final_info['trial_num']} with Val Loss {best_model_final_info['val_loss']:.4f}, Best Val Acc: {best_val_acc:.4f}")

    # Load the winning model's state into the main pred_model
    pred_model = predictor.TrajectoryPredictor(k, d)
    pred_model.load_state_dict(best_model_final_info['state_dict'])

    with torch.no_grad():
        predicted_flat_trajectory = pred_model(all_steps)
        tau_hat_infty_flat = pred_model.A.sum(dim=0)

    # Log the final fitted parameters from the best model
    logger.info(f"--- Best Fitted Trajectory Parameters (k={k}) ---")
    fitted_r = F.softplus(pred_model.log_r).detach().cpu()
    fitted_A = pred_model.A.detach().cpu()

    # Log the corresponding r_i and a_i statistics together
    logger.info("Statistics for each component (r_i, a_i):")
    for i in range(k):
        r_i = fitted_r[i].item()
        a_i = fitted_A[i]
        log_message = (
            f"  Component {i}:\n"
            f"\t- r_{i}      = {r_i:.6f}\n"
            f"\t- a_{i} L2 Norm = {torch.linalg.vector_norm(a_i).item():.2f}\n"
            f"\t- a_{i} Mean    = {a_i.mean().item():.6f}\n"
            f"\t- a_{i} Max     = {a_i.max().item():.6f}\n"
            f"\t- a_{i} Min     = {a_i.min().item():.6f}"
        )
        logger.info(log_message)

    # --- 5: VISUALIZATION ---
    logger.info("Calculating metrics for visualization...")
    with torch.no_grad():
        if space == 'adapter':
            tau_hat_infty = utils.unflatten_to_state_dict(tau_hat_infty_flat, reference_dict_for_unflatten)
            tau_hat_infty_operational = utils.convert_to_operational_tau(tau_hat_infty)
            predicted_adapter_taus = [utils.unflatten_to_state_dict(p, reference_dict_for_unflatten) for p in predicted_flat_trajectory]
            predicted_operational_taus = [utils.convert_to_operational_tau(t) for t in predicted_adapter_taus]
            predicted_flat_operational = torch.stack([torch.cat([p.flatten() for p in tau.values()]) for tau in predicted_operational_taus])
        else:
            tau_hat_infty_operational = utils.unflatten_to_state_dict(tau_hat_infty_flat, reference_dict_for_unflatten)
            predicted_flat_operational = predicted_flat_trajectory
        
        # Calculate all metrics needed for plotting
        metrics = {
            'cos_sims_t': F.cosine_similarity(predicted_flat_operational, all_flat_operational_taus, dim=1),
            'l2_dists_t': torch.linalg.vector_norm(predicted_flat_operational - all_flat_operational_taus, dim=1),
            'mag_tau_t': torch.linalg.vector_norm(all_flat_operational_taus, dim=1),
            'mag_tau_hat_t': torch.linalg.vector_norm(predicted_flat_operational, dim=1),
            'cos_sim_to_star_actual': F.cosine_similarity(all_flat_operational_taus, tau_star_flat_operational, dim=1),
            'cos_sim_to_star_pred': F.cosine_similarity(predicted_flat_operational, tau_star_flat_operational, dim=1),
            'l2_dist_to_star_actual': torch.linalg.vector_norm(all_flat_operational_taus - tau_star_flat_operational, dim=1),
            'l2_dist_to_star_pred': torch.linalg.vector_norm(predicted_flat_operational - tau_star_flat_operational, dim=1),
        }
        metrics['cos_sims_t'][0] = torch.nan # cos sim at t=0 is undefined
        metrics['l2_dists_t'][0] = torch.nan # l2 dist at t=0 is undefined
        tau_infty_flat = all_flat_operational_taus[-1]
        tau_hat_infty_flat_operational = torch.cat([p.flatten() for p in tau_hat_infty_operational.values()])
        metrics['final_cos_sim'] = F.cosine_similarity(tau_hat_infty_flat_operational, tau_infty_flat, dim=0).item()
        metrics['final_l2_dist'] = torch.linalg.vector_norm(tau_hat_infty_flat_operational - tau_infty_flat).item()

    logger.info("Generating plots...")
    title_suffix = f'({space}, k={k}, N={N})'
    
    plotting.plot_magnitude_comparison(all_steps, metrics['mag_tau_t'], metrics['mag_tau_hat_t'], torch.linalg.vector_norm(tau_star_flat_operational).item(), title_suffix, plots_dir)
    plotting.plot_cosine_similarity_between_trajectories(all_steps, metrics['cos_sims_t'], metrics['final_cos_sim'], title_suffix, plots_dir)
    plotting.plot_l2_distance_between_trajectories(all_steps, metrics['l2_dists_t'], metrics['final_l2_dist'], title_suffix, plots_dir)
    plotting.plot_alignment_to_star(all_steps, metrics['cos_sim_to_star_actual'], metrics['cos_sim_to_star_pred'], title_suffix, plots_dir)
    plotting.plot_distance_to_star(all_steps, metrics['l2_dist_to_star_actual'], metrics['l2_dist_to_star_pred'], title_suffix, plots_dir)
    plotting.plot_final_performance_comparison(top_candidates, best_val_acc, acc_star, acc_zero, title_suffix, plots_dir)

    # Log the hyperparameters used for this fitting run
    hyperparam_log_message = (
        f"--- Hyperparameters Used for Fitting ---\n"
        f"  - num_trials: {num_trials}\n"
        f"  - num_alternating_steps: {num_alternating_steps}\n"
        f"  - num_r_steps_per_alternation: {num_r_steps_per_alternation}\n"
        f"  - lr_r: {lr_r}\n"
        f"  - r_reg_lambda: {r_reg_lambda}\n"
        f"  - A_reg_lambda: {A_reg_lambda}\n"
        f"  - r_init_range: [{r_init_low}, {r_init_high}]\n"
        f"  - score_A_norm_weight: {score_A_norm_weight}"
    )
    logger.info(hyperparam_log_message)

    # --- 6: WANDB LOGGING ---
    if config['logging']['wandb']:
        logger.info("Initializing Wandb for analysis run...")
        analysis_config = {
        "source_run": exp_dir.name, "k": k, "N": N, "prediction_space": space,
        "num_trials": num_trials,
        "num_final_candidates": num_final_candidates,
        "num_alternating_steps": num_alternating_steps,
        "num_r_steps_per_alternation": num_r_steps_per_alternation,
        "lr_r": lr_r,
        "A_reg_lambda": A_reg_lambda,
        "r_reg_lambda": r_reg_lambda,
        "r_init_low": r_init_low, "r_init_high": r_init_high,
        "score_A_norm_weight": score_A_norm_weight        
        }
        run_name = f"analysis_{exp_dir.parent.name}/{exp_dir.name}_{space}_k{k}_N{N}"
        wandb.init(project="tvp-analysis", config=analysis_config, name=run_name)

        logger.info("Logging element trajectories to Wandb...")
        # Loop over all subsequent time steps including t=0
        wandb_tracked_indices = None
        for i, step in enumerate(tqdm(all_steps, desc="Logging trajectories")):
            predicted_tau_i_dict = utils.unflatten_to_state_dict(predicted_flat_trajectory[i], reference_dict_for_unflatten)

            wandb_tracked_indices = utils.log_tau_elements_to_wandb(
                tau_dict_actual=all_adapter_taus[i],
                tau_dict_predicted=predicted_tau_i_dict,
                global_step=int(step),
                tracked_indices=wandb_tracked_indices,
                config=config
            )
        
        logger.info("Logging plots to Wandb...")
        wandb.log({p.stem: wandb.Image(str(p)) for p in plots_dir.glob("*.png")})
        
        logger.info("Logging final selection results table to Wandb...")
        results_table = wandb.Table(columns=[
        "Trial #", "Type", "Fitting Loss", "A_Norm", "Score", "Val Acc"
        ])
        results_table.add_data(
            "N/A", "Baseline (theta_0)", None, None, None, acc_zero
        )
        results_table.add_data(
            "N/A", "Ground Truth (tau_*)", None, None, None, acc_star
        )

        for candidate in top_candidates:
            is_best = '(Selected)' if candidate['trial_num'] == best_model_final_info['trial_num'] else ''
            results_table.add_data(
                str(candidate['trial_num']),
                f"Candidate {is_best}",
                candidate.get('loss'),
                candidate.get('A_norm'),
                candidate.get('score'),
                candidate.get('val_acc')
            )

        wandb.log({"final_performance_table": results_table})        
        wandb.finish()

    logger.info(f"Analysis complete for prediction space: {space}.")
    logger.info("-----------------------------------------------------\n")