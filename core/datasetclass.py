import json
import torch
from torch_geometric.data import Data, Dataset
import os
from .utils import *
import numpy as np
class LPBFDataset(Dataset):
    def __init__(self, data_dir, add_targets, split_frames, add_noise):
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
        self.add_targets = add_targets
        self.add_noise = add_noise
        self.split_frames = split_frames
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
        node_type = torch.zeros((mesh_pos.shape[0]), dtype=torch.int64)
        # edge_index = torch.cat((decomposed_connectivity[0].reshape(1, -1), decomposed_connectivity[1].reshape(1, -1)), dim=0)
        senders, receivers = decomposed_connectivity[0], decomposed_connectivity[1]
        if self.add_targets :
            target_temperature = temperature[1:, :].detach().clone()
        if self.split_frames & self.add_targets :
            #list of data (frame)
            frames = []
            for idx in range(target_temperature.shape[0]) :
                temperature_t = temperature[idx].reshape(-1, 1)
                target_temperature_t = target_temperature[idx].reshape(-1, 1)
                heat_source_t = heat_source[idx].reshape(-1, 1)
                if self.add_noise :
                    temperature_noise_scale = 5
                    heat_source_noise_scale = 100
                    temperature_noise = torch.zeros_like(temperature_t) + temperature_noise_scale * torch.randn_like(temperature_t)
                    heat_source_noise = torch.zeros_like(heat_source_t) + heat_source_noise_scale * torch.randn_like(heat_source_t)
                    temperature_t += temperature_noise
                    heat_source_t += heat_source_noise
                frame = Data(temperature = temperature_t, 
                             target_temperature = target_temperature_t, 
                             heat_source = heat_source_t,
                             mesh_pos = mesh_pos,  
                             senders = senders, 
                             receivers = receivers, 
                             cells = cells,
                             node_type = node_type)
                frames.append(frame)
            return frames


        return Data(temperature = temperature, 
                    mesh_pos = mesh_pos, 
                    heat_source = heat_source, 
                    node_type = node_type,
                    cells = cells, 
                    senders = senders, 
                    receivers = receivers)
    def get_name(self, idx) :
        return self.file_name_list[idx]

if __name__ == "__main__" :
    data_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files"
    dataset = LPBFDataset(data_dir, add_targets=True, split_frames=True, add_noise = True)
    data = dataset[0]
    print(data[0].temperature)
    print(data[0].target_temperature)
    print(data[0])
    print(len(dataset))