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
import h5py
import meshio
import time

# def load_dataset( , add_target = False, add_noise = False, split = False) :
#     file_path = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files/weld_fem_60mm.npz"
#     dataset = np.load(file_path)

#     data = Data()
#     return 

# def learner() :
    
device = "cuda"
def run_step(graph) :
    with torch.no_grad():
        curr_temp = model.predict(graph)
        graph.temperature = curr_temp
        return graph
def rollout(data) :
    data = [frame.to(device) for frame in data]
    initial_state = data[0]
    curr_graph = initial_state
    timesteps = len(data)
    # q_list = [data[timestep].heat_source for timestep in range(timesteps)]
    pred_temperature_list = []
    gt_temperature_list = []
    progress = tqdm(range(timesteps), desc="Rollout")
    for timestep in progress :
        # print(timestep, "max temp:", torch.max(curr_graph.temperature))
        curr_graph.heat_source = data[timestep].heat_source
        curr_graph = run_step(curr_graph)
        pred_temperature_list.append(curr_graph.temperature)
        gt_temperature_list.append(data[timestep].target_temperature)

    return dict(mesh_pos = initial_state.mesh_pos,
                node_type = initial_state.node_type,
                cells = initial_state.cells,
                predict_temperature = pred_temperature_list,
                gt_temperature = gt_temperature_list,
                heat_source = [frame.heat_source for frame in data])


def plot_paraview_pvd(output_dir, filename, output, max_timesteps=50):
    os.makedirs(output_dir, exist_ok=True)
    
    mesh_pos = output["mesh_pos"].detach().cpu().numpy()
    cells = output["cells"].detach().cpu().numpy()
    total_timesteps = len(output["predict_temperature"])

    # Compute downsample step
    step = max(1, total_timesteps // max_timesteps)
    selected_indices = list(range(0, total_timesteps, step))[:max_timesteps]

    pvd_entries = []

    progress_bar = tqdm(selected_indices, total=len(selected_indices))
    for out_idx, timestep in enumerate(progress_bar):
        points = mesh_pos
        temp_data = output["predict_temperature"][timestep].detach().cpu().numpy()
        gt_temp_data = output["gt_temperature"][timestep].detach().cpu().numpy()
        q_data = output["heat_source"][timestep].detach().cpu().numpy()

        mesh = meshio.Mesh(
            points=points,
            cells=[("tetra", cells)],
            point_data={
                "pred_temperature": temp_data,
                "gt_temperature": gt_temp_data,
                "heat_source": q_data
            }
        )
        vtu_dir = os.path.join(output_dir, "vtu")
        os.makedirs(vtu_dir, exist_ok=True)
        vtu_file = f"temperature_ts_{out_idx}.vtu"
        meshio.write(os.path.join(vtu_dir, vtu_file), mesh)
        pvd_entries.append(f'    <DataSet timestep="{out_idx}" group="" part="0" file="./vtu/{vtu_file}"/>')

    # Write .pvd file
    pvd_content = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
        *pvd_entries,
        '  </Collection>',
        '</VTKFile>'
    ]

    with open(os.path.join(output_dir, f"{filename}.pvd"), "w") as f:
        f.write('\n'.join(pvd_content))

    print(f"PVD + VTU files written with downsampled time resolution (every {step} steps).")

if __name__ == "__main__" :
    test_on = "offset"
    data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/output/20250430T112913"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{test_on}"
    paraview_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{test_on}"
    # run_dir = prepare_directories(output_dir)
    # model_dir = os.path.join(run_dir, 'model_checkpoint')
    # logs_dir = os.path.join(run_dir, "logs")
    # logger_setup(os.path.join(logs_dir, "logs.txt"))
    # logger = logging.getLogger()
    model_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/trained_model/2025-04-30T23h28m03s/model_checkpoint"
    dataset = LPBFDataset(data_dir, add_targets= True, split_frames=True, add_noise = False)
    data = dataset[1]
    model = EncodeProcessDecode(node_feature_size = 3,
                                mesh_edge_feature_size = 5,
                                output_size = 1,
                                latent_size = 128,
                                message_passing_steps = 15)
    model.to(device)
    model.load_model(model_dir)
    model.eval()
    model = torch.compile(model)
    # Training loop
    output = rollout(data)
    plot_paraview_pvd(paraview_dir, test_on, output)