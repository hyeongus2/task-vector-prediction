# src/tvp/trainer.py
import logging
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

from . import utils, data_loader, model as model_loader

# Create a logger for this module
logger = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
             text_features: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on the given dataloader.

    Args:
        model (torch.nn.Module): The CLIPModel (or PeftModel) to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        text_features (torch.Tensor): The pre-computed text features for classification.
        device (torch.device): The device to run evaluation on.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            
            image_features = model.get_image_features(pixel_values=batch['pixel_values'])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = image_features @ text_features.T
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * len(labels)
            
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy


def train(config: dict, experiment_group_dir: Path, resume_id: Optional[str] = None):
    """
    Main training function that orchestrates the entire finetuning process.
    """
    # --- 1. Setup: Wandb & Directories ---
    run = None
    if config['logging']['wandb']:
        run = wandb.init(
            project="task-vector-prediction", 
            config=config, 
            name=experiment_group_dir.name,
            id=resume_id,
            resume="must" if resume_id else None
        )
    
    run_id = run.id if run else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = experiment_group_dir / run_id
    task_vectors_dir = output_dir / "task_vectors"

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "effective_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    utils.set_seed(config.get('seed', 42))
    
    utils.setup_logging(output_dir / "logs", log_filename="train.log", enabled=config['logging']['enabled'])
    logger.info(f"Output directory: {output_dir}")
    
    device = utils.get_device()

    # --- 2. Data Loading ---
    processor = model_loader.CLIPProcessor.from_pretrained(config['model_id'])
    train_loader, val_loader, class_names = data_loader.get_dataloaders(config, processor)

    # --- 3. Model Creation ---
    model, text_features = model_loader.create_model(config, class_names, processor, device)
    
    theta0_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    logger.info("Saving initial model state (theta0)...")
    utils.save_torch(theta0_state_dict, output_dir / "theta0.pt")
    logger.info("Saving text features (classifier weights)...")
    utils.save_torch(text_features, output_dir / "text_features.pt")

    # --- 4. Optimizer Setup ---
    finetune_config = config['finetuning']
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if finetune_config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=finetune_config['lr'], momentum=finetune_config['momentum'])
        logger.info(f"Optimizer: SGD, LR: {finetune_config['lr']}, Momentum: {finetune_config['momentum']}")
    else: # Default to adamw
        optimizer = torch.optim.AdamW(trainable_params, lr=finetune_config['lr'])
        logger.info(f"Optimizer: {finetune_config['optimizer'].upper()}, LR: {finetune_config['lr']}")

    # --- 5. Checkpoint Loading ---
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    if resume_id:
        checkpoint_path = output_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = utils.load_torch(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resumed from Epoch {start_epoch}, Global Step {global_step}")
        else:
            logger.warning(f"Resume ID provided, but checkpoint file not found. Starting from scratch.")

    # --- 6. Monitoring Setup ---
    wandb_tracked_indices = None
    if config['logging']['wandb'] and not resume_id:
        # Log t=0 state.
        # Create a proper zero-filled dictionary for t=0.
        
        # 1. Get the structure of the adapter by calculating it once on the initial model.
        #    This serves as a template.
        tau_template = utils.calculate_task_vector(model, theta0_state_dict)
        
        # 2. Create a zero-filled dict with the same structure (keys and shapes).
        zero_tau_dict = {key: torch.zeros_like(value) for key, value in tau_template.items()}
        
        # 3. Pass the valid zero-filled dict to the logger.
        wandb_tracked_indices = utils.log_tau_elements_to_wandb(
            tau_dict_actual=zero_tau_dict,
            tau_dict_predicted=None,
            global_step=0,
            tracked_indices=None,
            config=config
        )

    # --- 7. Training Loop ---
    logger.info("--- Starting Training ---")
    for epoch in range(start_epoch, finetune_config['epochs']):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{finetune_config['epochs']}")
        
        epoch_train_loss, epoch_train_correct, epoch_train_samples = 0, 0, 0
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            
            image_features = model.get_image_features(pixel_values=batch['pixel_values'])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            epoch_train_correct += (preds == labels).sum().item()
            epoch_train_samples += len(labels)

            if global_step % config['analysis']['save_tau_every_n_steps'] == 0:
                adapter_tau = utils.calculate_task_vector(model, theta0_state_dict)
                utils.save_torch(adapter_tau, task_vectors_dir / f"tau_{global_step}.pt")
                
                # Call the unified helper function for logging
                if config['logging']['wandb']:
                    wandb_tracked_indices = utils.log_tau_elements_to_wandb(
                        tau_dict_actual=adapter_tau,
                        tau_dict_predicted=None,
                        global_step=global_step,
                        tracked_indices=wandb_tracked_indices,
                        config=config
                    )

        # --- 8. End-of-Epoch ---
        avg_train_loss = epoch_train_loss / epoch_train_samples if epoch_train_samples > 0 else 0
        avg_train_acc = epoch_train_correct / epoch_train_samples if epoch_train_samples > 0 else 0
        val_loss, val_accuracy = evaluate(model, val_loader, text_features, device)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
        
        if config['logging']['wandb']:
            wandb.log({
                "train/epoch_loss": avg_train_loss, "train/epoch_accuracy": avg_train_acc,
                "validation/loss": val_loss, "validation/accuracy": val_accuracy,
                "epoch": epoch + 1
            }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss! Saving tau_star.")
            tau_star = utils.calculate_task_vector(model, theta0_state_dict)
            utils.save_torch(tau_star, task_vectors_dir / "tau_star.pt")

        checkpoint = { 'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss }
        utils.save_torch(checkpoint, output_dir / "checkpoint.pt")
        logger.info(f"Saved checkpoint for epoch {epoch+1}")

    if config['logging']['wandb']:
        wandb.finish()
    
    logger.info("--- Training Finished ---")
    logger.info("-----------------------------------------------------\n")