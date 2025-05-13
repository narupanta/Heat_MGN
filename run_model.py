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
# def load_dataset( , add_target = False, add_noise = False, split = False) :
#     file_path = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files/weld_fem_60mm.npz"
#     dataset = np.load(file_path)

#     data = Data()
#     return 

# def learner() :
    
device = "cuda"

if __name__ == "__main__" :
    
    data_dir = r"/home/y0113799/Hiwi/Heat_MGN/groundtruth"
    output_dir = r"/home/y0113799/Hiwi/Heat_MGN/output"
    run_dir = prepare_directories(output_dir)
    model_dir = os.path.join(run_dir, 'model_checkpoint')
    logs_dir = os.path.join(run_dir, "logs")
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()
    time_window = 10
    dataset = LPBFDataset(data_dir, add_targets= True, split_frames=True, add_noise = True, time_window = time_window)
    model = EncodeProcessDecode(node_feature_size = 3,
                                mesh_edge_feature_size = 5,
                                output_size = 1,
                                latent_size = 128,
                                timestep=1e-5,
                                time_window=time_window,
                                device=device,
                                message_passing_steps = 15)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
    num_epochs = 100
    train_loss_per_epochs = []
    is_accumulate_normalizer_phase = True
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        val_total_loss = 0
        for traj_idx, trajectory in enumerate(dataset):  # assuming dataset.trajectories exists
            train_loader = DataLoader(trajectory, batch_size=1, shuffle=True)
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

            for idx_traj, batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                predictions = model(batch)
                loss = model.loss(predictions, batch)

                if not is_accumulate_normalizer_phase:
                    loss.backward()
                    optimizer.step()
                    train_total_loss += loss.item()
                    loop.set_description(f"Epoch {epoch + 1} Traj {traj_idx + 1}/{len(dataset)}")
                    loop.set_postfix({"Loss": f"{loss.item():.4f}"})
                    logger.info(f"Epoch {epoch+1}, Trajectory {traj_idx+1}, Step {idx_traj+1}: Loss = {loss.item():.4f}")

            # Rollout/Validation after each trajectory
            if not is_accumulate_normalizer_phase:
                val_loss = torch.mean(rollout(model, trajectory, time_window)['mse'])
                val_total_loss += val_loss
                logger.info(f"Epoch {epoch + 1}, Trajectory {traj_idx + 1}: Rollout MSE = {val_loss:.4f}")

        if not is_accumulate_normalizer_phase:
            avg_train_loss = train_total_loss / len(dataset)
            train_loss_per_epochs.append(avg_train_loss)
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                model.save_model(model_dir)
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_state_dict.pth"))
            print(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_train_loss:.4f}")
        else:
            is_accumulate_normalizer_phase = False

