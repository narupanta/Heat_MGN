# load model and rollout
import torch
import yaml
import os
from datetime import datetime
from core.meshgraphnet import EncodeProcessDecode
from core.datasetclass import LPBFDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  
import numpy as np
from core.rollout import rollout
from core.utils import plot_paraview_pvd

if __name__ == "__main__":
    import logging
    import sys
    # -----------------------------
    # Setup logging
    log_file = "rollout_log.txt"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])
    log = logging.getLogger()
    # find config.yml in model directory
    load_model_dir = "./trained_models/continued_from_best_model_20251020T135254"
    data_dir = "./dataset/trainset"
    save_rollout_dir = "./rollouts/trainset"
    config_path = os.path.join(load_model_dir, 'config.yml')
    if not os.path.exists(config_path):
        print(f"Config file not found in {load_model_dir}")
        exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        # override model and training parameters if specified in config file
        node_in_dim = config["model"]["node_in_dim"]
        edge_in_dim = config["model"]["edge_in_dim"]
        hidden_size = config["model"]["hidden_size"]
        process_steps = config["model"]["process_steps"]
        node_out_dim = config["model"]["node_out_dim"]
        attention = config["model"]["attention"]
        time_window = config["model"]["time_window"]

        learning_rate = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"].get("weight_decay", 1e-5))
        num_epochs = config["training"]["num_epochs"]
        start_noise = config["training"]["start_noise_level"]
        end_noise = config["training"]["end_noise_level"]
        save_model_dir = config["paths"].get("save_model_dir", './trained_models')
        if data_dir :
            data_dir = data_dir
        else :
            data_dir = config["paths"]["data_dir"]

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
    model_path = os.path.join(load_model_dir, 'best_model')
    model.load_model(model_path)
    model.eval()
    
    dataset = LPBFDataset(data_dir = data_dir, add_target= False, noise_level=0.0, time_window = time_window)
    
    # loop through all samples in the dataset
    for idx in range(len(dataset)):
        sample_name = dataset.get_name(idx).rstrip(".npz")
        print(sample_name)
        # if not sample_name.endswith("rest") :
        #     continue  
        print(f"Running rollout for sample {sample_name} ({idx+1}/{len(dataset)})")
        data = dataset[idx]
        # input trajectory into rollout prediction
        trajectory_rollout = rollout(model, data)
        # save rollout predictions and error
        os.makedirs(os.path.join(save_rollout_dir, sample_name), exist_ok=True)
        np.savez_compressed(os.path.join(save_rollout_dir, sample_name, 'rollout.npz'),
                            pred=trajectory_rollout["pred"].detach().cpu().numpy(),
                            gt=trajectory_rollout["gt"].detach().cpu().numpy(),
                            heat_source = trajectory_rollout["heat_source"].detach().cpu().numpy(),
                            cells = trajectory_rollout["cells"].detach().cpu().numpy(),
                            mesh_pos = trajectory_rollout["mesh_pos"].detach().cpu().numpy(),
                            rmse_T = trajectory_rollout["rmse_T"].detach().cpu().numpy())
        print(f"Rollout predictions and error saved in {save_rollout_dir}")
        print(f"RMSE T: {trajectory_rollout["rmse_T"].detach().cpu().numpy().float():.6f}")
        plot_paraview_pvd(save_rollout_dir + "_paraview" + "/" + sample_name, sample_name, trajectory_rollout)
        

    