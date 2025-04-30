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


# def load_dataset( , add_target = False, add_noise = False, split = False) :
#     file_path = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files/weld_fem_60mm.npz"
#     dataset = np.load(file_path)

#     data = Data()
#     return 

# def learner() :
    
device = "cuda"
def run_step(graph) :
    with torch.no_grad():
        curr_temp = model.predict(graph.to(device))
        graph.temperature = curr_temp
        return graph
def rollout(data) :
    initial_state = data[0]
    mesh_pos = initial_state.mesh_pos
    node_type = initial_state.node_type
    cells = initial_state.cells
    curr_graph = initial_state
    inter_timesteps = len(data)
    extra_timesteps = len(data)
    pred_temperature_list = []
    gt_temperature_list = []
    for timestep in range(extra_timesteps) :
        print(timestep)
        if timestep < inter_timesteps :
            curr_graph.heat_source = data[timestep].heat_source
        else :
            curr_graph.heat_source = torch.zeros_like(initial_state.heat_source)
        curr_graph = run_step(curr_graph)
        pred_temperature_list.append(curr_graph.temperature)
        gt_temperature_list.append(data[timestep].target_temperature)

    return dict(mesh_pos = mesh_pos,
                node_type = node_type,
                cells = cells,
                predict_temperature = pred_temperature_list,
                gt_temperature = gt_temperature_list)


def plot_paraview(output_dir, output):
    mesh_pos = output["mesh_pos"].detach().cpu().numpy()
    cells = output["cells"].detach().cpu().numpy()
    pred_temperature_list = output["predict_temperature"]
    gt_temperature_list = output["gt_temperature"]
    progress_bar = tqdm(range(pred_temperature_list), total=len(temperature_list))
    for idx, t in progress_bar:
        points = mesh_pos  # Node positions at current timestep
        temp_data = pred_temperature_list[idx].detach().cpu().numpy()  # Temperature at current timestep
        gt_temp_data = gt_temperature_list[idx].detach().cpu().numpy() 
        # Create a Mesh object with only point data (no cells)
        mesh = meshio.Mesh(
            points=points,
            cells=[("tetra", cells)],  # No connectivity (point cloud)
            point_data={
                "pred_temperature": temp_data,
                "gt_temperature": gt_temp_data   # Add temperature as point data
            }
        )
        
        # Write the VTU file
        output_file = os.path.join(output_dir, f"temperature_ts_{idx}.vtu")
        meshio.write(output_file, mesh)
    
    print("VTU files (Point Cloud) generated successfully.")

def plot_paraview_pvd(output_dir, filename, output):
    os.makedirs(output_dir, exist_ok=True)
    
    mesh_pos = output["mesh_pos"].detach().cpu().numpy()
    cells = output["cells"].detach().cpu().numpy()
    pvd_entries = []
    pred_temperature_list = output["predict_temperature"]
    gt_temperature_list = output["gt_temperature"]
    progress_bar = tqdm(enumerate(pred_temperature_list), total=len(pred_temperature_list))
    for idx, t in progress_bar:
        points = mesh_pos  # Node positions at current timestep
        temp_data = pred_temperature_list[idx].detach().cpu().numpy()  # Temperature at current timestep
        gt_temp_data = gt_temperature_list[idx].detach().cpu().numpy() 
        # Create a Mesh object with only point data (no cells)
        mesh = meshio.Mesh(
            points=points,
            cells=[("tetra", cells)],  # No connectivity (point cloud)
            point_data={
                "pred_temperature": temp_data,
                "gt_temperature": gt_temp_data   # Add temperature as point data
            }
        )

        vtu_file = f"temperature_ts_{idx}.vtu"
        meshio.write(os.path.join(output_dir, vtu_file), mesh)
        pvd_entries.append(f'    <DataSet timestep="{idx}" group="" part="0" file="{vtu_file}"/>')

    # Create the .pvd file
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

    print("PVD + VTU time series written successfully.")

if __name__ == "__main__" :
    
    data_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/npz_files"
    output_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/output"
    paraview_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/paraview_files/single-track-mgn-case1"
    # run_dir = prepare_directories(output_dir)
    # model_dir = os.path.join(run_dir, 'model_checkpoint')
    # logs_dir = os.path.join(run_dir, "logs")
    # logger_setup(os.path.join(logs_dir, "logs.txt"))
    # logger = logging.getLogger()
    model_dir = r"/home/narupanta/Hiwi/weld-simulation-pinn/weld-simulation-pinn/output/2025-04-28T12h15m36s/model_checkpoint"
    dataset = LPBFDataset(data_dir, add_targets= True, split_frames=True, add_noise = False)
    data = dataset[2]
    model = EncodeProcessDecode(node_feature_size = 3,
                                mesh_edge_feature_size = 5,
                                output_size = 1,
                                latent_size = 128,
                                message_passing_steps = 15)
    model.to(device)
    model.load_model(model_dir)
    model.eval()
    # Training loop
    output = rollout(data)
    plot_paraview_pvd(paraview_dir, 'single-track-mgn-case1', output)