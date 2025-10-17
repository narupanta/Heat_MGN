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

cube = Cube(2e-3, 2e-3, 0.5e-3)

size_min = 0.1e-3
dist_min = 0.25e-3
y_offset = 0
num_track = 4
track_order = [1, 2, 3, 4]  # Custom order (1-based index)
assert len(track_order) == num_track and set(track_order) == set(range(1, num_track + 1))

geometry_dir = "./geometry/"
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

# --- Material Properties ---
def rhocp(T):
    return 8351.910158 * (446.337248 + 0.14180844 * (T - 273) - 61.431671211432 * ufl.exp(
        -0.00031858431233904 * ((T - 273) - 525) ** 2) + 1054.9650568 * ufl.exp(-0.00006287810196136 * ((T - 273) - 1545) ** 2))

def k(T):
    return 25  # W/(m·K)

# --- Time-Stepping Parameters ---
dt = 5e-5  # s

# --- Boundary Conditions ---
T_ambient = 298.0  # K (ambient temperature)
zero = fem.Constant(domain, PETSc.ScalarType(T_ambient + 200))

def boundary_bottom(x):
    return np.isclose(x[2], 0)

bc = fem.dirichletbc(zero, fem.locate_dofs_geometrical(V, boundary_bottom), V)

# --- Neumann Boundary (Convection & Radiation) ---
h = fem.Constant(domain, PETSc.ScalarType(25.0))  # W/(m²·K)
ε = fem.Constant(domain, PETSc.ScalarType(0.8))   # Emissivity
σ = fem.Constant(domain, PETSc.ScalarType(5.67e-8))  # Stefan–Boltzmann constant
T_inf = fem.Constant(domain, PETSc.ScalarType(T_ambient))

def boundary_top(x):
    return np.isclose(x[2], cube.height)

top_facets = fem.locate_dofs_geometrical(V, boundary_top)

# --- Moving Heat Source Parameters ---
power = 150.0  # W
η = 1
source_speed = 0.1  # m/s
source_width = 0.9e-3  # m
radius_source = source_width / 2
pen_depth = 0.5e-3  # m
# Circular scan parameters
center_x = cube.length / 2
center_y = cube.width / 2
radius_scan = min(cube.length, cube.width) * 0.3  # 80% of half-size
num_rotations = 1  # number of full circles during simulation
omega = source_speed/radius_scan  # angular velocity

# --- Total simulation time ---
T_heat = num_rotations * 2 * np.pi * radius_scan/source_speed
T_total = T_heat * 1.1  # s

def moving_heat_source(x, x_center, y_center):
    q_max = 6 * η * power * np.sqrt(3) / (np.pi ** 1.5 * radius_source ** 3)
    return q_max * np.exp(-3 * ((x[0] - x_center) ** 2 + (x[1] - y_center) ** 2) / radius_source ** 2) \
           * np.exp(-3 * (x[2] - cube.height) ** 2 / pen_depth ** 2)

q = fem.Function(V, name="Heat Source")
q.x.array[:] = np.zeros_like(domain.geometry.x[:, 0], dtype=PETSc.ScalarType)

# --- Weak Formulation ---
T_ = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
T_n = fem.Function(V, name="Temperature (K)")
T_n.x.array[:] = T_ambient

a = (1/dt) * rhocp(T_n) * ufl.inner(T_, v) * ufl.dx + k(T_n) * ufl.inner(ufl.grad(T_), ufl.grad(v)) * ufl.dx
L = (1/dt) * rhocp(T_n) * ufl.inner(T_n, v) * ufl.dx + ufl.inner(q, v) * ufl.dx

problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "ilu"})

# --- Output File ---
folder_name = datetime.now().strftime("%Y%m%dT%H%M%S")
result_dir = os.path.join(output_dir, folder_name)
os.makedirs(result_dir, exist_ok=True)

geometry_name = input_geometry_path.split("/")[-1].rstrip(".msh")
xdmf = io.XDMFFile(MPI.COMM_WORLD, os.path.join(result_dir, f"{geometry_name}.xdmf"), "w")
xdmf.write_mesh(domain)

# --- Time-Stepping Loop with Circular Scan ---
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

    if i % int((T_total/dt * 0.1)) == 0:
        xdmf.write_function(T_n, t)
        xdmf.write_function(q, t)

    print(f"Time: {t:.6f} s")

    # Circular scan position
    theta = omega * t
    x_center = center_x + radius_scan * np.cos(theta)
    y_center = center_y + radius_scan * np.sin(theta)

    # Update heat source
    q.x.array[:] = np.array([moving_heat_source(x, x_center, y_center) for x in domain.geometry.x], dtype=PETSc.ScalarType)

    # Solve linear problem
    T_sol = problem.solve()
    T_n.x.array[:] = T_sol.x.array[:]

    t += dt
    i += 1

# Save results
np.savez(os.path.join(result_dir, f"{geometry_name}.npz"),
         t=t_list, T=T_list, mesh_pos=mesh_pos,
         node_connectivity=node_connectivity, q=q_list)

xdmf.close()
end_time = time.time()
elapsed = end_time - start_time
print(f"Simulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
