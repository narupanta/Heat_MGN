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
def run_step(model, graph) :
    with torch.no_grad():
        curr_temp = model.predict(graph.to(device))
        graph.temperature = curr_temp
        return graph
# def rollout(model, data) :
#     data = [frame for frame in data]
#     initial_state = data[0]
#     curr_graph = initial_state
#     timesteps = len(data)
#     # q_list = [data[timestep].heat_source for timestep in range(timesteps)]
#     pred_temperature_list = []
#     gt_temperature_list = []
#     rmse_list = []
#     progress = tqdm(range(timesteps), desc="Rollout")
#     for timestep in progress :
#         curr_graph.heat_source = data[timestep].heat_source
#         # curr_graph = run_step(curr_graph)
#         curr_graph = run_step(data[timestep])
#         rmse_error = torch.sqrt(torch.mean((curr_graph.temperature - data[timestep].target_temperature.to(device))**2))
#         pred_temperature_list.append(curr_graph.temperature)
#         gt_temperature_list.append(data[timestep].target_temperature)
#         rmse_list.append(rmse_error)
#         print("max temp:", torch.max(curr_graph.temperature), "\nerr:", rmse_error)
#     return dict(mesh_pos = initial_state.mesh_pos,
#                 node_type = initial_state.node_type,
#                 cells = initial_state.cells,
#                 predict_temperature = pred_temperature_list,
#                 gt_temperature = gt_temperature_list,
#                 rmse_list = rmse_list,
#                 heat_source = [frame.heat_source for frame in data])

def rollout(model, data, time_window):
    data = [frame for frame in data]
    initial_state = data[0]
    timesteps = len(data)

    rmse_list = []

    curr_graph = initial_state.clone()  # Start with first ground truth graph
    mse_tensor = torch.zeros((1,), device = model._device)
    pred_temperature_tensor = curr_graph.temperature.to(model._device)
    gt_temperature_tensor = curr_graph.temperature.to(model._device)
    progress = tqdm(range(0, timesteps - time_window, time_window), desc="Rollout")

    for t in progress:
        curr_graph.heat_source = data[t].heat_source  # Use correct heat source

        # === Predict next time_window temperatures ===
        with torch.no_grad():
            pred_temp = model.predict(curr_graph.to(device))  # (num_nodes, time_window)
            curr_graph.temperature = pred_temp[:, -1:].clone()  # use last timestep prediction for next input

        # === Ground truth from data[t+1] to data[t+time_window] ===
        gt_temp = data[t].target_temperature.to(device)
        window_mse = torch.mean((pred_temp - gt_temp) ** 2, dim=0)
        mse_tensor = torch.cat([mse_tensor, window_mse])  # (time_window,)
        pred_temperature_tensor = torch.cat([pred_temperature_tensor, pred_temp], dim=1)
        gt_temperature_tensor = torch.cat([gt_temperature_tensor, gt_temp], dim=1)

        print(f"[t={t}] max temp: {torch.max(pred_temp):.4f} | MSE: {torch.mean(window_mse):.4f}")

    return dict(
        mesh_pos=initial_state.mesh_pos,
        node_type=initial_state.node_type,
        cells=initial_state.cells,
        predict_temperature=pred_temperature_tensor.T,
        gt_temperature=gt_temperature_tensor.T,
        mse_tensor=mse_tensor,
        heat_source=[frame.heat_source for frame in data]
    )


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
        temp_data = output["predict_temperature"][timestep, :].detach().cpu().numpy()
        gt_temp_data = output["gt_temperature"][timestep, :].detach().cpu().numpy()
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

def plot_max_temperature_over_time(pred_temp, gt_temp, rmse_temp, time_step=1e-5, title="Temperature Over Time", save_path="temperature-time.png"):
    """
    Plots and saves the maximum temperature over time as a PNG image.
    
    Parameters:
    - temperature_sequence: Tensor of shape (T, ...) where T is the number of time steps
    - time_step: Time interval between steps (default 1.0)
    - title: Title of the plot
    - save_path: File path to save the PNG image
    """
    
    max_pred_temp = [torch.max(pred_temp_t).detach().cpu() for pred_temp_t in pred_temp] 
    max_gt_temp = [torch.max(gt_temp_t).detach().cpu() for gt_temp_t in gt_temp]
    rmse_temp = [rmse_temp_t.detach().cpu() for rmse_temp_t in rmse_temp]
    # Create time axis
    time_axis = torch.arange(len(gt_temp)) * time_step
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, max_pred_temp.numpy(), label="Max Pred Temp (K)", color='r')
    plt.plot(time_axis, max_gt_temp.numpy(), label="Max Gt Temp (K)", color='g')
    plt.plot(time_axis, rmse_temp.numpy(), label="RMSE Temp (K)", color='b')
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__" :
    test_on = "triple321"
    # data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/output/20250430T112913"
    # data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/output/20250429T151016"
    # data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/output/20250430T113333"
    # data_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/groundtruth/20250512T162621"
    data_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/groundtruth/{test_on}"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{test_on}"
    paraview_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{test_on}"
    # run_dir = prepare_directories(output_dir)
    # model_dir = os.path.join(run_dir, 'model_checkpoint')
    # logs_dir = os.path.join(run_dir, "logs")
    # logger_setup(os.path.join(logs_dir, "logs.txt"))
    # logger = logging.getLogger()
    time_window = 10
    model_dir = r"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/trained_model/2025-05-09T12h23m14s/model_checkpoint"
    dataset = LPBFDataset(data_dir, add_targets= True, split_frames=True, add_noise = False, time_window=time_window)
    data = dataset[1]
    model = EncodeProcessDecode(node_feature_size = 3,
                                mesh_edge_feature_size = 5,
                                output_size = 1,
                                latent_size = 128,
                                message_passing_steps = 15,
                                time_window=time_window,
                                timestep=1e-5,
                                device = device)
    model.to(device)
    model.load_model(model_dir)
    model.eval()
    model = torch.compile(model)
    # Training loop
    output = rollout(model, data, time_window)
    # save_dict_to_pkl(output, os.path.join(output_dir, f"{test_on}.pkl"))
    # plot_max_temperature_over_time(output["predict_temperature"], 
    #                                output["gt_temperature"], 
    #                                output["rmse_list"],
    #                                time_step=1e-5, 
    #                                title="Temperature Over Time", 
    #                                save_path = os.path.join(output_dir, f"{test_on}-temperature-time.png"))
    plot_paraview_pvd(paraview_dir, test_on, output)