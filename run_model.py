#!/usr/bin/env python3
import os
import yaml
import logging
from datetime import datetime

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from core.datasetclass import LPBFDataset
from core.meshgraphnet import EncodeProcessDecode
from core.rollout import rollout

# -------------------------
# Helper functions
# -------------------------
def noise_schedule(epoch, total_epochs, initial_noise=0.1, final_noise=0.01):
    """Linear noise schedule from initial_noise to final_noise over total_epochs."""
    if epoch >= total_epochs:
        return final_noise
    return initial_noise + (final_noise - initial_noise) * (epoch / total_epochs)

def log_model_parameters(model, logger):
    total_params = 0
    total_trainable = 0

    logger.info("===== Model Parameters =====")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            total_trainable += num_params
        logger.info(f"{name}: {param.size()} | params={num_params} | requires_grad={param.requires_grad}")

    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {total_trainable}")
    logger.info("============================")

def save_training_state(model_dir, optimizer, scheduler, epoch):
    """Save optimizer, scheduler and epoch to model_dir."""
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(model_dir, "scheduler.pt"))
    with open(os.path.join(model_dir, "epoch.txt"), "w") as f:
        f.write(str(epoch))

def try_load_opt_sched_epoch(optimizer, scheduler, continue_model_path, device):
    """Try to load optimizer, scheduler, epoch from a checkpoint path. Return start_epoch (int)."""
    start_epoch = 0
    opt_path = os.path.join(continue_model_path, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    sched_path = os.path.join(continue_model_path, "scheduler.pt")
    if os.path.exists(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
    epoch_file = os.path.join(continue_model_path, "epoch.txt")
    if os.path.exists(epoch_file):
        try:
            with open(epoch_file, "r") as f:
                start_epoch = int(f.read().strip())
        except Exception:
            start_epoch = 0
    return start_epoch

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # config defaults (will be overridden if config file exists)
    config_path = "train_config.yml"
    config = None

    # Minimal required keys will be validated after loading
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file '{config_path}' not found. Please provide it.")

    # --- Read model/training config (expect these to exist in config) ---
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    # Required model params
    node_in_dim = model_cfg["node_in_dim"]
    edge_in_dim = model_cfg["edge_in_dim"]
    hidden_size = model_cfg["hidden_size"]
    process_steps = model_cfg["process_steps"]
    node_out_dim = model_cfg["node_out_dim"]
    attention = model_cfg.get("attention", False)
    time_window = model_cfg.get("time_window", 1)

    # Training params
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    num_epochs = int(training_cfg.get("num_epochs", 200))
    start_noise = float(training_cfg.get("start_noise_level", 0.1))
    end_noise = float(training_cfg.get("end_noise_level", 0.01))

    # Paths
    save_model_dir = paths_cfg.get("save_model_dir", "./trained_models")
    data_dir = paths_cfg.get("data_dir", "./data")
    continue_model_path = paths_cfg.get("continue_model_path", None)  # None or path string

    # Create base save dir if not exists
    os.makedirs(save_model_dir, exist_ok=True)

    # -------------------------
    # Setup device, model, optimizer, scheduler
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncodeProcessDecode(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_size=hidden_size,
        process_steps=process_steps,
        node_out_dim=node_out_dim,
        attention=attention,
        time_window=time_window,
        device=device
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Scheduler (cosine annealing as before)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # -------------------------
    # Prepare model_dir and optionally load previous checkpoint
    # -------------------------
    now = datetime.now()
    dt_string = now.strftime("%Y%m%dT%H%M%S")

    if continue_model_path:
        if not os.path.exists(continue_model_path):
            raise FileNotFoundError(f"continue_model_path specified but not found: {continue_model_path}")

        # New folder for continued training, so original checkpoint isn't overwritten
        continued_basename = os.path.basename(os.path.normpath(continue_model_path))
        model_dir = os.path.join(save_model_dir, f"continued_from_{continued_basename}_{dt_string}")
        os.makedirs(model_dir, exist_ok=True)

        # Copy config into the new model_dir for provenance
        with open(os.path.join(model_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

        # Setup logging to new model_dir
        log_file = os.path.join(model_dir, "log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("train")

        logger.info(f"Continuing training from checkpoint: {continue_model_path}")
        # Load model weights + normalizers using your provided method
        model.load_model(continue_model_path)

        # Try load optimizer, scheduler, and epoch (if present in the checkpoint dir)
        start_epoch = try_load_opt_sched_epoch(optimizer, scheduler, continue_model_path, device)
        logger.info(f"Resuming from epoch (last completed): {start_epoch}. Next epoch will start at {start_epoch}")

    else:
        # Fresh run: create new model_dir with timestamp
        model_dir = os.path.join(save_model_dir, dt_string)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

        # Setup logging
        log_file = os.path.join(model_dir, "log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("train")
        start_epoch = 0
        logger.info("Starting fresh training run")

    # Log model parameters
    log_model_parameters(model, logger)

    # -------------------------
    # Prepare datasets
    # -------------------------
    dataset = LPBFDataset(
        data_dir=data_dir,
        add_target=True,
        noise_level=noise_schedule(0, num_epochs, initial_noise=start_noise, final_noise=end_noise),
        time_window=time_window
    )

    rollout_dataset = LPBFDataset(
        data_dir=data_dir,
        add_target=False,
        noise_level=0.0,
        time_window=time_window
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_val_loss = float('inf')

    for train_epoch in range(start_epoch, num_epochs):
        # train_epoch will be the integer written into epoch.txt at end of epoch
        model.train()
        total_loss = 0.0

        # Update dataset noise according to schedule (per-epoch)
        dataset.noise_level = noise_schedule(train_epoch, num_epochs, initial_noise=start_noise, final_noise=end_noise)

        for traj_idx, data in enumerate(dataset):
            traj_total_loss = 0.0
            train_loader = DataLoader(data, batch_size=1, shuffle=True)
            loop = tqdm(train_loader, leave=False)
            for batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model.loss(batch)
                loss.backward()
                optimizer.step()
                traj_total_loss += loss.item()
                loop.set_description(f"Epoch {train_epoch + 1}, Traj {traj_idx + 1}")
                loop.set_postfix({"Loss": f"{loss.item():.4f}"})
            total_loss += traj_total_loss
            logger.info(f"Epoch {train_epoch + 1}, Trajectory {traj_idx + 1}: Train Loss: {traj_total_loss:.4f}")

        avg_loss = total_loss / max(1, len(dataset))
        logger.info(f"Epoch {train_epoch + 1}, Train Loss: {avg_loss:.6f}")

        # Step scheduler (after epoch)
        scheduler.step()

        # Evaluate with rollouts
        model.eval()
        total_rollout_loss = 0.0
        num_rollouts = 0
        with torch.no_grad():
            for trajectory in rollout_dataset:
                rollout_result = rollout(model, trajectory)
                # rollout_result expected to contain 'rmse_T' key as before
                rmse_T = rollout_result.get("rmse_T", None)
                if rmse_T is None:
                    # if rollout returns a different metric, try to handle gracefully
                    logger.warning("rollout did not return 'rmse_T' key. Skipping this rollout.")
                    continue
                total_rollout_loss += float(rmse_T)
                num_rollouts += 1
                logger.info(f"Rollout Nr.{num_rollouts} Loss: {rmse_T:.6f}")

        avg_rollout_loss = total_rollout_loss / max(1, num_rollouts)
        logger.info(f"Epoch {train_epoch + 1} Rollout Loss: {avg_rollout_loss:.6f}")

        # Save best model (and optimizer/scheduler/epoch files) in model_dir/best_model
        if avg_rollout_loss < best_val_loss:
            best_val_loss = avg_rollout_loss
            best_model_dir = os.path.join(model_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            # Use existing model saving method (assumes it writes model_weights.pth and normalizers)
            model.save_model(best_model_dir)
            # Save optimizer/scheduler/epoch for reproducibility
            save_training_state(best_model_dir, optimizer, scheduler, train_epoch)
            logger.info(f"Best model saved to {best_model_dir} (rollout loss improved to {best_val_loss:.6f})")

        # Save periodic checkpoint every 20 epochs (and always save current epoch state)
        if (train_epoch + 1) % 20 == 0:
            epoch_model_dir = os.path.join(model_dir, f"epoch_{train_epoch+1}")
            os.makedirs(epoch_model_dir, exist_ok=True)
            model.save_model(epoch_model_dir)
            save_training_state(epoch_model_dir, optimizer, scheduler, train_epoch)
            logger.info(f"Epoch checkpoint saved to {epoch_model_dir}")

        # Always update the "latest" working checkpoint with model + optimizer + scheduler + epoch
        current_working_dir = os.path.join(model_dir, "latest")
        os.makedirs(current_working_dir, exist_ok=True)
        model.save_model(current_working_dir)
        save_training_state(current_working_dir, optimizer, scheduler, train_epoch)

        # Also save into model_dir root so there's a snapshot for this epoch
        # (optional, but helpful)
        snapshot_dir = os.path.join(model_dir, f"snapshot_epoch_{train_epoch+1}")
        os.makedirs(snapshot_dir, exist_ok=True)
        model.save_model(snapshot_dir)
        save_training_state(snapshot_dir, optimizer, scheduler, train_epoch)

    logger.info("Training finished.")
