import numpy as np
from torch_geometric.data import Data
import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
from core.datasetclass import LPBFDataset
from core.model_graphnet import EncodeProcessDecode
import numpy as np
from tqdm import tqdm
from core.utils import * 
from run_rollout import rollout
import argparse
import yaml

device = "cuda"
    
def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN model with YAML config.")
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default="./train_config.yml")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    device = config["device"]
    data_dir = config["paths"]["data_dir"]
    output_dir = config["paths"]["output_dir"]
    model_dir = config["paths"].get("model_dir", None)

    run_dir = prepare_directories(output_dir)
    model_dir = model_dir or os.path.join(run_dir, 'model_checkpoint')
    logs_dir = os.path.join(run_dir, "logs")
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()

    time_window = int(config["training"]["time_window"])
    add_noise = float(config["training"]["add_noise"])  # Optional: could read from config["training"]["add_noise"]


    train_dataset = LPBFDataset(
        data_dir,
        add_targets=True,
        split_frames=True,
        add_noise=add_noise,
        time_window=time_window
    )

    val_dataset = LPBFDataset(
        data_dir,
        add_targets=False,
        split_frames=False,
        add_noise=None,
        time_window=time_window
    )

    model = EncodeProcessDecode(
        node_feature_size=config["model"]["node_feature_size"],
        mesh_edge_feature_size=config["model"]["mesh_edge_feature_size"],
        output_size=config["model"]["output_size"],
        latent_size=config["model"]["latent_size"],
        timestep=float(config["model"]["timestep"]),
        time_window=time_window,
        device=device,
        message_passing_steps=config["training"]["message_passing_steps"],
        attention=config["training"]["attention"]
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    num_epochs = config["training"]["num_epochs"]
    train_loss_per_epochs = []
    is_accumulate_normalizer_phase = True
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        val_total_loss = 0
        for traj_idx, trajectory in enumerate(train_dataset):  # assuming dataset.trajectories exists
            train_loader = DataLoader(trajectory, batch_size=1, shuffle=True)
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            trajectory_loss = 0
            for idx_traj, batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                predictions = model(batch)
                loss = model.loss(predictions, batch)
                # loss = model.fem_loss(predictions, batch)

                if not is_accumulate_normalizer_phase:
                    loss.backward()
                    optimizer.step()
                    trajectory_loss += loss.item()
                    loop.set_description(f"Epoch {epoch + 1} Traj {traj_idx + 1}/{len(train_dataset)}")
                    loop.set_postfix({"Loss": f"{loss.item():.4f}"})
            train_total_loss += trajectory_loss
            logger.info(f"Epoch {epoch+1}, Trajectory {traj_idx+1}: Loss = {train_total_loss:.4f}")

            # Rollout/Validation after each trajectory
        if not is_accumulate_normalizer_phase:
            val_total_loss = 0.0
            for traj_idx, trajectory in enumerate(val_dataset):
                output = rollout(model, trajectory, time_window)
                temp_rmse = torch.mean(output["temp_rmse"])
                percentage_temp_rmse = torch.mean(output["percentage_temp_rmse"])
                val_loss = percentage_temp_rmse
                val_total_loss += val_loss.item()

                logger.info(
                    f"Val Traj {traj_idx + 1}: percentage temperature error: {val_loss:.6e}, temp_rmse: {temp_rmse:.4f}"
                )
            avg_train_loss = train_total_loss / len(train_dataset)
            avg_val_loss = val_total_loss / len(val_dataset)

            logger.info(f"Epoch {epoch + 1} Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.6e}")
            print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.6e}")
            
            avg_train_loss = train_total_loss / len(train_dataset)
            train_loss_per_epochs.append(avg_train_loss)
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                model.save_model(model_dir)
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_state_dict.pth"))
                logger.info("Checkpoint saved (best model so far).")
            print(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_train_loss:.4f}")
        else:
            is_accumulate_normalizer_phase = False

