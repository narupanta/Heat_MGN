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
# def load_dataset( , add_target = False, add_noise = False, split = False) :
#     file_path = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files/weld_fem_60mm.npz"
#     dataset = np.load(file_path)

#     data = Data()
#     return 

# def learner() :
    
device = "cuda"

if __name__ == "__main__" :
    
    data_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files"
    output_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/output"
    run_dir = prepare_directories(output_dir)
    model_dir = os.path.join(run_dir, 'model_checkpoint')
    logs_dir = os.path.join(run_dir, "logs")
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()
    
    dataset = LPBFDataset(data_dir, add_targets= True, split_frames=True, add_noise = True)
    data = dataset[0]
    model = EncodeProcessDecode(node_feature_size = 4,
                                mesh_edge_feature_size = 5,
                                output_size = 1,
                                latent_size = 128,
                                message_passing_steps = 15)
    
    train_loader = DataLoader(data, batch_size = 1, shuffle = False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
    # Training loop
    num_epochs = 1000
    train_loss_per_epochs = []
    model.to(device)
    is_accumulate_normalizer_phase = True
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for idx_traj, batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss = model.loss(predictions, batch)

            # Backpropagation
            if is_accumulate_normalizer_phase is False : # use first epoch to accumulate normalizer
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()
                loop.set_description(f"Trajectory {idx_traj + 1}/{len(train_loader)}")
                loop.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Trajectory {idx_traj + 1}/{len(train_loader)}, MSE Loss {loss.item():.4f}")

        if is_accumulate_normalizer_phase is False :
            avg_train_loss = train_total_loss / len(train_loader)
            train_loss_per_epochs.append(avg_train_loss)
            model.save_model(model_dir)
            print(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_train_loss:.4f}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_train_loss:.4f}")
        else :
            is_accumulate_normalizer_phase = False