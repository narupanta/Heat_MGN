import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import LinearProblem
from generate_geometry import create_msh_file, Cube
from datetime import datetime
import os
import time
# -----------------------------
# Transient 2D Heat Conduction with Enthalpy, Moving Heat Source, Convection & Radiation
# -----------------------------
import os
import time
import numpy as np
import ufl
import yaml
import argparse
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from generate_geometry import create_msh_file, Cube
from datetime import datetime
from sympy import symbols, lambdify, exp, parse_expr
import sympy
import ufl
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import torch
# --- Argument & Config ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

args = parse_args()
config = load_config(args.config)

# --- Geometry Parameters ---
geom_cfg = config["geometry"]
cube = Cube(geom_cfg["length"], geom_cfg["width"], geom_cfg["height"])
size_min = float(geom_cfg["size_min"])
dist_min = float(geom_cfg["dist_min"])
y_offset = float(geom_cfg["y_offset"])
num_track = int(geom_cfg["num_track"])
track_order = geom_cfg["track_order"]
assert len(track_order) == num_track and set(track_order) == set(range(1, num_track + 1))

# --- Paths ---
geometry_dir = config["paths"]["geometry_dir"]
output_dir = config["paths"]["output_dir"]

# --- Mesh ---
input_geometry_path = create_msh_file(geometry_dir, "cube", cube, size_min, dist_min, y_offset, num_track)
domain, cell_tags, facet_tags = gmshio.read_from_msh(input_geometry_path, MPI.COMM_WORLD, 0)
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
domain.topology.create_connectivity(fdim - 2, domain.topology.dim)

mesh_pos = domain.geometry.x
node_connectivity = domain.topology.connectivity(domain.topology.dim, fdim - 2).array.reshape(-1, 4)

# --- Function Space ---
V = fem.functionspace(domain, ("CG", 1))

# --- Simulation Parameters ---
sim_cfg = config["simulation"]
dt = float(sim_cfg["dt"])
power = float(sim_cfg["power"])
η = float(sim_cfg["eta"])
source_speed = float(sim_cfg["source_speed"])
source_width = float(sim_cfg["source_width"])
radius = source_width / 2
pen_depth = float(sim_cfg["penetration_depth"])
T_heat = cube.length * 0.5 / source_speed * num_track
T_total = T_heat * float(sim_cfg["total_factor"])

# --- Material Model ---
material_cfg = config["material"]
# rhocp = lambda T: 8351.910158 * ((446.337248 + 0.14180844 * (T-273) - 61.431671211432 * ufl.exp(-0.00031858431233904*((T-273)-525)**2) + 1054.9650568*ufl.exp(-0.00006287810196136*((T-273)-1545)**2)))
rhocp = lambda T: 8351.910158 * (446.337248 + 0.14180844 * (T-273) - 61.431671211432 * np.e ** (-0.00031858431233904*((T-273)-525)**2) + 1054.9650568*np.e **(-0.00006287810196136*((T-273)-1545)**2))
k = lambda T: 25   # W/(m·K)

# --- Boundary Conditions ---
T_ambient = config["boundary"]["ambient_temperature"]
zero = fem.Constant(domain, PETSc.ScalarType(T_ambient + 200))

def boundary_bottom(x): return np.isclose(x[2], 0)
bc = fem.dirichletbc(zero, fem.locate_dofs_geometrical(V, boundary_bottom), V)

# --- Neumann (optional) ---
h = fem.Constant(domain, PETSc.ScalarType(config["boundary"]["convection_coefficient"]))
ε = fem.Constant(domain, PETSc.ScalarType(config["boundary"]["emissivity"]))
σ = fem.Constant(domain, PETSc.ScalarType(config["boundary"]["stefan_boltzmann"]))
T_inf = fem.Constant(domain, PETSc.ScalarType(T_ambient))

# --- Heat Source Function ---
def moving_heat_source(x, y_center, t):
    f = 1
    q_max = 6 * η * f * power * np.sqrt(3) / (np.pi**1.5 * radius**3)
    return q_max * np.exp(-3 * (x[0] - source_speed * t - cube.length/4)**2 / radius**2
                          -3 * (x[1] - y_center)**2 / radius**2
                          -3 * (x[2] - cube.height)**2 / pen_depth**2)

# --- FEM setup ---
T_ = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
T_n = fem.Function(V, name="Temperature (K)")
T_n.x.array[:] = T_ambient

q = fem.Function(V, name="Heat Source")
q.x.array[:] = 0

a = (1/dt) * rhocp(T_n)*ufl.inner(T_, v)*ufl.dx + k(T_n)*ufl.inner(ufl.grad(T_), ufl.grad(v))*ufl.dx
L = (1/dt)*rhocp(T_n)*ufl.inner(T_n, v)*ufl.dx + ufl.inner(q, v)*ufl.dx

problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "ilu"})

# --- Output Setup ---
folder_name = datetime.now().strftime("%Y%m%dT%H%M%S")
result_dir = os.path.join(output_dir, folder_name)
os.makedirs(result_dir, exist_ok=True)

geometry_name = os.path.basename(input_geometry_path).replace(".msh", "")
xdmf = io.XDMFFile(MPI.COMM_WORLD, os.path.join(result_dir, f"{geometry_name}.xdmf"), "w")
xdmf.write_mesh(domain)

# --- Time Loop ---
t = 0.0
i = 0
t_list, T_list, q_list = [], [], []
start = time.time()

while t < T_total:
    t_list.append(t)
    T_list.append(T_n.x.array.copy())
    q_list.append(q.x.array.copy())

    if i % int(T_total/dt * 0.1) == 0:
        xdmf.write_function(T_n, t)
        xdmf.write_function(q, t)

    print(f"Time: {t:.6f} s")

    # Apply heat source
    if num_track == 1:
        y_center = cube.width / 2 + y_offset
        q_values = [moving_heat_source(x, y_center, t) if t <= T_heat else 0 for x in mesh_pos]
    else:
        track_duration = cube.length * 0.5 / source_speed
        total_duration = track_duration * num_track
        if t < total_duration:
            current_segment = int(t // track_duration)
            local_t = t - current_segment * track_duration
            track_id = track_order[current_segment]
            y_center = (cube.width / num_track) * (track_id - 0.5) + y_offset
            q_values = [moving_heat_source(x, y_center, local_t) for x in mesh_pos]
        else:
            q_values = [0.0 for _ in mesh_pos]

    q.x.array[:] = np.array(q_values, dtype=PETSc.ScalarType)
    T_sol = problem.solve()

    stiffness = assemble_matrix(problem.a)
    stiffness.assemble()
    ai, aj, av = stiffness.getValuesCSR()
    stiffness_tensor = torch.sparse_csr_tensor(ai, aj, av, dtype = torch.float)
    
    # stiffness = stiffness.convert("dense")
    # stiffness_mat = stiffness.getDenseArray()
    load = assemble_vector(problem.L)
    load.assemble()
    load_vec = load.getArray()
    load_vec_tensor = torch.tensor(load_vec, dtype = torch.float)
    res = torch.mean(stiffness_tensor @ torch.tensor(T_sol.x.array[:], dtype = torch.float) - load_vec_tensor)
    print(res)

    T_n.x.array[:] = T_sol.x.array[:]
    t += dt
    i += 1

# --- Save output ---
np.savez(os.path.join(result_dir, f"{geometry_name}.npz"), t=t_list, T=T_list,
         mesh_pos=mesh_pos, node_connectivity=node_connectivity, q=q_list)

xdmf.close()
print(f"Simulation done in {(time.time()-start):.2f} s")
