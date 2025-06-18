import torch
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time
import datetime
import os
import ufl
import basix.ufl
import numpy as np
import dolfinx
from mpi4py import MPI

def tetrahedral_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = torch.cat(
            (faces[:, 0:2],
             faces[:, 1:3],
             faces[:, 2:4],
             torch.stack((faces[:, 3], faces[:, 0]), dim=1),
             torch.stack((faces[:, 3], faces[:, 1]), dim=1),
             torch.stack((faces[:, 2], faces[:, 1]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
    receivers, _ = torch.min(edges, dim=1)
    senders, _ = torch.max(edges, dim=1)

    packed_edges = torch.stack((senders, receivers), dim=1)
    unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
    senders, receivers = torch.unbind(unique_edges, dim=1)
    senders = senders.to(torch.int64)
    receivers = receivers.to(torch.int64)

    two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
    return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    
def plot_training_loss() :
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()

def logger_setup(log_path):
    # set log configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # console_output_handler = logging.StreamHandler(sys.stdout)
    # console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    # console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    # root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)
    return root_logger

def prepare_directories(output_dir):
    
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%dT%Hh%Mm%Ss")
    run_dir = os.path.join(output_dir, formatted_now)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # make all the necessary directories
    checkpoint_dir = os.path.join(run_dir, 'model_checkpoint')
    log_dir = os.path.join(run_dir, 'logs')

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)


    return run_dir

def save_overview(path, cube, size_min, dist_min, num_track, power, source_speed,
                  source_width, pen_depth, η, dt, T_total, T_ambient, h, ε, σ):
    with open(path, "w") as f:
        f.write("Simulation Summary\n")
        f.write("==================\n")
        f.write(f"Geometry: Cube ({cube.length:.1e} x {cube.width:.1e} x {cube.height:.1e}) m\n")
        f.write(f"Mesh: size_min = {size_min:.1e}, dist_min = {dist_min:.1e}, num_track = {num_track}\n")
        f.write("Material:\n")
        f.write("  Thermal conductivity k(T) = 25 W/(m·K)\n")
        f.write("  rho*cp(T) = temperature-dependent\n")
        f.write("Heat Source:\n")
        f.write(f"  Power = {power} W, speed = {source_speed} m/s, width = {source_width:.1e} m\n")
        f.write(f"  Penetration depth = {pen_depth:.1e} m, efficiency = {η}\n")
        f.write("Time:\n")
        f.write(f"  dt = {dt:.1e} s, total_time = {T_total:.4e} s\n")
        f.write("Boundary Conditions:\n")
        f.write(f"  Dirichlet bottom: T = {T_ambient + 200} K\n")
        f.write(f"  Ambient T = {T_ambient} K\n")
        f.write(f"  h = {h.value} W/(m²·K), ε = {ε.value}, σ = {σ.value} W/(m²·K⁴)\n")

def load_overview(path):
    settings = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=")
                key = key.strip().lower().replace(" ", "_")
                val = val.strip().split()[0]  # Take only the number
                try:
                    settings[key] = float(val)
                except ValueError:
                    settings[key] = val
    return settings

import pickle

def save_dict_to_pkl(data_dict, file_path):
    """
    Saves a dictionary to a .pkl file.
    
    Parameters:
    - data_dict: Dictionary to save
    - file_path: Destination .pkl file path (e.g., "data/output.pkl")
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)

def load_dict_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def create_mesh_from_arrays(node_position, node_connectivity, cell_type_name, gdim):
    """
    Creates a FEniCSx mesh from node positions and connectivity.

    Args:
        node_position (np.ndarray): A NumPy array of shape (num_nodes, gdim) 
                                    containing the coordinates of the nodes.
        node_connectivity (np.ndarray): A NumPy array of shape (num_cells, num_vertices_per_cell)
                                        containing the global indices of the nodes for each cell.
        cell_type_name (str): The name of the cell type (e.g., "triangle", "tetrahedron").
        gdim (int): The geometric dimension of the mesh (2 for 2D, 3 for 3D).

    Returns:
        dolfinx.mesh.Mesh: The created FEniCSx mesh object.
    """
    
    # Ensure arrays have correct data types
    node_position = np.asarray(node_position, dtype=np.float64, order='C')
    node_connectivity = np.asarray(node_connectivity, dtype=np.int64, order='C')

    # Define the UFL cell (geometric element)
    # The degree of the geometric element should usually be 1 for linear meshes
    # The shape parameter corresponds to the geometric dimension of the nodes.
    ufl_domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_type_name, 1, shape=(gdim,)))

    # Create the DOLFINx mesh
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, node_connectivity, node_position, ufl_domain)
    
    return mesh