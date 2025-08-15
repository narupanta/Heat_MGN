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
cube = Cube(2e-3, 4e-3, 1e-3)

size_min = 0.04e-3
dist_min = 0.25e-3
y_offset = 0
num_track = 1
track_order = [1]  # Custom order (1-based index)
assert len(track_order) == num_track and set(track_order) == set(range(1, num_track + 1))
geometry_dir = "/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/geometry"
output_dir = "./groundtruth/"
# --- Geometry & Mesh ---
input_geometry_path = create_msh_file(geometry_dir, "cube", cube, size_min, dist_min, y_offset, num_track)
domain, cell_tags, facet_tags = gmshio.read_from_msh(input_geometry_path, MPI.COMM_WORLD, 0)
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
domain.topology.create_connectivity(fdim - 2, domain.topology.dim)

node_connectivity = domain.topology.connectivity(domain.topology.dim, fdim - 2).array.reshape(-1, 4)
mesh_pos = domain.geometry.x

# --- Function Space ---
V = fem.functionspace(domain, ("CG", 1))

# --- Time-Stepping Parameters ---s
# 1e-3/0.1
# --- Material Properties ---
# def k(T): return 234.887443886408 + 0.018367198 * (T-273) + 225.10088 * ufl.tanh(0.0118783 * ((T-273) - 1543.815))  # W/(m·K)
def rhocp(T): return 8351.910158 * ((446.337248 + 0.14180844 * (T-273) - 61.431671211432 * ufl.exp(
                    -0.00031858431233904*((T-273)-525)**2) + 1054.9650568*ufl.exp(-0.00006287810196136*((T-273)-1545)**2)))
def k(T): return 25   # W/(m·K)
# def rhocp(T): return 8351.910158 * (446.337248)

#CFL condition -> dt <= 1/2 * dx**2
# dt = 2e-5  # s

# --- Boundary Conditions ---
T_ambient = 298.0  # K (ambient temperature)
zero = fem.Constant(domain, PETSc.ScalarType(T_ambient + 200))

def boundary_bottom(x): return np.isclose(x[2], 0)
bc = fem.dirichletbc(zero, fem.locate_dofs_geometrical(V, boundary_bottom), V)

# --- Neumann Boundary (Convection & Radiation) ---
h = fem.Constant(domain, PETSc.ScalarType(25.0))  # W/(m²·K)
ε = fem.Constant(domain, PETSc.ScalarType(0.8))   # Emissivity
σ = fem.Constant(domain, PETSc.ScalarType(5.67e-8))  # Stefan–Boltzmann constant
T_inf = fem.Constant(domain, PETSc.ScalarType(T_ambient))

# Apply on top boundary
# def boundary_top(x): return np.isclose(x[2], 0.5e-3 + 0.05e-3 + 0.03e-3)
def boundary_top(x): return np.isclose(x[2], 1.03e-3)
top_facets = fem.locate_dofs_geometrical(V, boundary_top)

# --- Moving Heat Source ---
power = 150.0  # W
η = 1
source_speed = 0.1  # m/s
source_width = 0.9e-3  # m
radius = source_width/2
cf, cr = radius, radius
ff, fr = 2/(1 + cr/cf), 2/(1 + cf/cr)
pen_depth = 0.5e-3 # m
dt = 1e-5 # s
T_heat = cube.length * 0.5/source_speed * num_track
T_total = T_heat * 3 # s
def moving_heat_source(x, y_center, t):
    x_center = source_speed * t
    # c = cf * (0.5 + 0.5 * np.tanh(1000 * (x[0] - x_center))) + \
    #     cr * (0.5 + 0.5 * np.tanh(1000 * (x_center - x[0])))
    # f = ff * (0.5 + 0.5 * np.tanh(1000 * (x[0] - x_center))) + \
    #     fr * (0.5 + 0.5 * np.tanh(1000 * (x_center - x[0])))
    f = 1
    q_max = 6 * η * f * power * np.sqrt(3) / (np.pi**1.5 * radius * radius * radius)

    return q_max* np.exp(-3 * (x[0] - x_center - cube.length/4)**2/radius**2 + -3 * (x[1] - y_center)**2 / radius**2) \
                 * np.exp(-3 * (x[2] - cube.height)**2 / pen_depth**2)

q = fem.Function(V, name = "Heat Source")  # Heat source
q.x.array[:] = np.array([0.0 for _ in domain.geometry.x], dtype=PETSc.ScalarType)
# --- Weak Formulation ---
T_ = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
T_n = fem.Function(V, name = "Temperature (K)")
T_n.x.array[:] = T_ambient

# Bilinear form (LHS)
a = (1/dt) * rhocp(T_n)*ufl.inner(T_, v)*ufl.dx + k(T_n)*ufl.inner(ufl.grad(T_), ufl.grad(v))*ufl.dx

# Linear form (RHS)
L = (1/dt)*rhocp(T_n)*ufl.inner(T_n, v)*ufl.dx + ufl.inner(q, v)*ufl.dx

# --- Add Convection and Radiation Terms ---
conv_term = -h * (T_n - T_inf) * v * ufl.ds
top_marker = 1  # Adjust if facet_tags are available with different markers
rad_term = -ε * σ * (T_n**4 - T_inf**4) * v * ufl.ds
# L += conv_term + rad_term

# --- Linear Problem Setup ---
problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "ilu"})

# --- Output File ---

folder_name = datetime.now().strftime("%Y%m%dT%H%M%S")
result_dir = os.path.join(output_dir, folder_name)
os.makedirs(result_dir, exist_ok=True)

geometry_name = input_geometry_path.split("/")[-1].rstrip(".msh")
xdmf = io.XDMFFile(MPI.COMM_WORLD, os.path.join(result_dir, f"{geometry_name}.xdmf"), "w")
xdmf.write_mesh(domain)

# --- Time-Stepping Loop ---
t = 0.0
i = 0
t_list = []
T_list = []
q_list = []
start_time = time.time()
while t < T_total:
    t_list.append(t)
    T_n_arr = T_n.x.array[:]
    T_list.append(T_n_arr.copy())
    q_list.append(q.x.array[:].copy())
    if i%int((T_total/dt * 0.1)) == 0 :
        xdmf.write_function(T_n, t)
        xdmf.write_function(q, t)
    print(f"Time: {t:.6f} s")

    track_length = cube.length * 0.5
    track_duration = track_length / source_speed
    total_duration = track_duration * num_track
    track_spacing = cube.width / num_track

    if num_track == 1:
        # Single track at center
        y_center = cube.width / 2 + y_offset
        if t <= track_duration:
            q.x.array[:] = np.array([moving_heat_source(x, y_center, t) for x in domain.geometry.x], dtype=PETSc.ScalarType)
        else:
            q.x.array[:] = np.array([0.0 for _ in domain.geometry.x], dtype=PETSc.ScalarType)
    else:
        # Multiple tracks
        total_duration = track_duration * num_track
        if t < total_duration:
            current_segment = int(t // track_duration)
            local_t = t - current_segment * track_duration

            if current_segment < len(track_order):
                # Apply custom track ordering
                track_id = track_order[current_segment]
                y_center = track_spacing * (track_id - 0.5) + y_offset
                q.x.array[:] = np.array([moving_heat_source(x, y_center, local_t) for x in domain.geometry.x], dtype=PETSc.ScalarType)
            else:
                q.x.array[:] = np.array([0.0 for _ in domain.geometry.x], dtype=PETSc.ScalarType)
        else:
            q.x.array[:] = np.array([0.0 for _ in domain.geometry.x], dtype=PETSc.ScalarType)
    T_sol = problem.solve()
    # Update for next step
    T_n.x.array[:] = T_sol.x.array[:]
    t += dt
    i += 1
np.savez(os.path.join(result_dir, f"{geometry_name}.npz"), t = t_list, T = T_list, mesh_pos = mesh_pos, node_connectivity = node_connectivity, q = q_list)

xdmf.close()
end_time = time.time()
elapsed = end_time - start_time
print(f"Simulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")