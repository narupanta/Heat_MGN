import json
import torch
from torch_geometric.data import Data, Dataset
import os
from .utils import *
import numpy as np
class LPBFDataset(Dataset):
    def __init__(self, data_dir, add_target, noise_level, time_window):
        """
        Generates synthetic dataset for material deformation use case.

        Args:
            num_graphs (int): Number of graphs in the dataset.
            num_nodes (int): Number of nodes per graph.
            num_features (int): Number of features per node.
            num_material_params (int): Number of material parameters.
        """
        super(LPBFDataset, self).__init__()
        self.data_dir = data_dir
        self.add_target = add_target
        self.noise_level = noise_level 
        self.time_window = time_window
        self.file_name_list = [filename for filename in sorted(os.listdir(data_dir)) if not os.path.isdir(os.path.join(data_dir, filename))]
    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        # Randomly generate node features
        file_name = self.file_name_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))

        decomposed_connectivity = tetrahedral_to_edges(torch.tensor(data['node_connectivity']))['two_way_connectivity']
        temperature = torch.tensor(data["T"], dtype=torch.float)
        heat_source = torch.tensor(data["q"], dtype=torch.float)
        mesh_pos = torch.tensor(data["mesh_pos"], dtype=torch.float)
        cells = torch.tensor(data['node_connectivity'])
        edge_index = torch.stack(decomposed_connectivity)
        if self.add_target :
            temperature_curr = temperature[1:-self.time_window]
            temperature_prev = temperature[:-self.time_window - 1]
            heat_source_curr = heat_source[1:-self.time_window]
            heat_source_prev = heat_source[:-self.time_window - 1]
            heat_source_next = torch.stack([heat_source[i + 1 : i + 1 + self.time_window] for i in range(temperature_curr.shape[0])])
            target_temperature = torch.stack([temperature[i + 1 : i + 1 + self.time_window] for i in range(temperature_curr.shape[0])])
            frames = []
            for idx in range(temperature_curr.shape[0]) :
                temperature_t = temperature_curr[idx]
                temperature_prev_t = temperature_prev[idx]
                target_temperature_t = target_temperature[idx]
                
                heat_source_t = heat_source_curr[idx]
                heat_source_prev_t = heat_source_prev[idx]
                heat_source_next_t = heat_source_next[idx]
                if self.noise_level > 0.0 :
                    temperature_noise_scale = (torch.max(temperature) - torch.min(temperature)) * self.noise_level
                    temperature_noise = torch.zeros_like(temperature_t) + temperature_noise_scale * torch.randn_like(temperature_t)
                    temperature_t += temperature_noise
                frame = Data(temperature_prev = temperature_prev_t,
                            temperature = temperature_t, 
                            target_temperature = target_temperature_t.T, 
                            heat_source_prev = heat_source_prev_t,
                            heat_source = heat_source_t,
                            heat_source_next = heat_source_next_t.T,
                            mesh_pos = mesh_pos,  
                            edge_index = edge_index,
                            cells = cells)
                frames.append(frame)
        else :
            frames = [Data(temperature = temperature[t],
                 heat_source = heat_source[t],
                 mesh_pos = mesh_pos,  
                 edge_index = edge_index,
                 cells = cells) for t in range(temperature.shape[0])]
        return frames

    def get_name(self, idx) :
        return self.file_name_list[idx]

if __name__ == "__main__" :
    data_dir = r"/mnt/c/Users/narun/Desktop/Project/Heat_MGN/dataset/trainset"
    dataset = LPBFDataset(data_dir, noise_level = 0.01, time_window = 10)
    data = dataset[1]
    check = 5