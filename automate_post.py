import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.ndimage import label, find_objects

def plot_temperature_fields_grid(mesh_pos, pred, gt, timesteps_to_plot, save_path=None):
    """
    Plots temperature fields in a grid with:
        - One colorbar per row
        - Shared min/max scale per row
        - Rows = [Predicted, Ground Truth, Percent Error]
        - Columns = Selected timesteps
    """
    # Set default font size
    plt.rcParams.update({'font.size': 14})

    x, y = mesh_pos[:, 0], mesh_pos[:, 1]
    triang = tri.Triangulation(x, y)
    n_cols = len(timesteps_to_plot)

    fig, axs = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12), constrained_layout=True)
    if n_cols == 1:
        axs = axs.reshape(3, 1)

    cmaps = ['viridis', 'viridis', 'seismic']
    row_titles = ['Predicted', 'Ground Truth', 'Percent Error']

    pred_vals = pred[timesteps_to_plot, :]
    gt_vals = gt[timesteps_to_plot, :]
    error_vals = (pred[timesteps_to_plot, :] - gt[timesteps_to_plot, :]) / pred[timesteps_to_plot, :] * 100

    global_ranges = [
        (pred_vals.min(), pred_vals.max()),
        (gt_vals.min(), gt_vals.max()),
        (-np.abs(error_vals).max(), np.abs(error_vals).max())
    ]

    mappables = [None, None, None]

    for col, step in enumerate(timesteps_to_plot):
        pred_step = pred_vals[col, :]
        gt_step = gt_vals[col, :]
        error_percent = error_vals[col, :]

        fields = [pred_step, gt_step, error_percent]

        for row in range(3):
            ax = axs[row, col]
            vmin, vmax = global_ranges[row]
            tpc = ax.tripcolor(triang, fields[row], cmap=cmaps[row], shading='flat', vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(f"Timestep {step}", fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if mappables[row] is None:
                mappables[row] = tpc

    for row in range(3):
        cbar = fig.colorbar(mappables[row], ax=axs[row, :], orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label(row_titles[row], fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    plt.suptitle("Temperature Field Comparison", fontsize=18)

    if save_path:
        with PdfPages(save_path) as pp:
            pp.savefig(fig, bbox_inches='tight')

    plt.show()

def analyze_melt_pool(mesh_pos, temperature, threshold=1618, grid_resolution=100, plot=True):
    """
    Analyze a 3D melt pool from temperature data.

    Parameters:
    - mesh_pos: (N, 3) ndarray of node positions (x, y, z)
    - temperature: (N,) ndarray of temperature values
    - threshold: temperature threshold for melt pool
    - grid_resolution: resolution in each dimension (int or tuple)
    - plot: whether to show a melt pool cross-section plot

    Returns:
    - length: melt pool length (X range)
    - width: melt pool width (Y range)
    - depth: melt pool depth (Z range)
    - melt_volume: 3D binary volume of melt pool
    """
    if isinstance(grid_resolution, int):
        grid_resolution = (grid_resolution,) * 3

    # Grid bounds
    x = np.linspace(mesh_pos[:, 0].min(), mesh_pos[:, 0].max(), grid_resolution[0])
    y = np.linspace(mesh_pos[:, 1].min(), mesh_pos[:, 1].max(), grid_resolution[1])
    z = np.linspace(mesh_pos[:, 2].min(), mesh_pos[:, 2].max(), grid_resolution[2])
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # Interpolate temperature to grid
    grid_temp = griddata(mesh_pos, temperature, (grid_x, grid_y, grid_z), method='linear')
    melt_mask = (grid_temp > threshold).astype(np.uint8)

    # Label connected melt pool regions
    labeled, num_features = label(melt_mask)
    if num_features == 0:
        return 0.0, 0.0, 0.0, melt_mask

    # Find the largest region (by voxel count)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # Ignore background
    largest_label = sizes.argmax()
    melt_mask = (labeled == largest_label)

    # Get bounding box of melt pool
    slices = find_objects(melt_mask)[largest_label - 1]
    x_slice, y_slice, z_slice = slices

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    length = (x_slice.stop - x_slice.start) * dx
    width  = (y_slice.stop - y_slice.start) * dy
    depth  = (z_slice.stop - z_slice.start) * dz

    # Optional: Plot mid-Z cross section
    if plot:
        mid_z = (z_slice.start + z_slice.stop) // 2
        plt.figure(figsize=(6, 5))
        plt.imshow(melt_mask[:, :, mid_z].T, origin='lower', cmap='hot',
                   extent=[x.min(), x.max(), y.min(), y.max()])
        plt.title(f"Melt Pool Cross Section (Z = {z[mid_z]:.2f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Melt Region")
        plt.axis('equal')
        plt.show()

    return length, width, depth

def plot_meltpool_predictions(meltpoolxyz_ts, total_steps, step_interval=100, save_path=None):
    """
    Plots prediction vs. ground truth for melt pool coordinates over time in 3 subplots (X, Y, Z).

    Parameters:
    - meltpoolxyz_ts: dict of time step -> {"pred": [x, y, z], "gt": [x, y, z]}
    - total_steps: int, total number of time steps
    - step_interval: int, interval between plotted time steps (default: 100)
    - save_path: str or None, path to save the plot (e.g., "meltpool_plot.png"); if None, only displays
    """
    steps = range(0, total_steps, step_interval)
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):  # X, Y, Z axes
        pred_values = [meltpoolxyz_ts[t]["pred"][i] for t in steps]
        gt_values   = [meltpoolxyz_ts[t]["gt"][i] for t in steps]

        axes[i].plot(steps, pred_values, label='Prediction', linestyle='--', marker='o')
        axes[i].plot(steps, gt_values, label='Ground Truth', linestyle='-', marker='x')
        axes[i].set_ylabel(axis_labels[i])
        axes[i].legend()
        axes[i].grid(True)

    axes[2].set_xlabel("Time Step")
    fig.suptitle("Melt Pool Dimension Prediction vs Ground Truth")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        with PdfPages(save_path) as pp:
            pp.savefig(fig, bbox_inches='tight')  # Save with proper spacing

    plt.show()

testcase_names = os.listdir("/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/testcases")
for testcase in testcase_names:
    rollout_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{testcase}"
    rollout = np.load(os.path.join(rollout_dir, "rollout.npz"))
    mesh_pos = rollout["mesh_pos"]
    predict_temperature = rollout["predict_temperature"]
    gt_temperature = rollout["gt_temperature"]
    heat_source = rollout["heat_source"]

    top_plane = mesh_pos[:, 2] == np.max(mesh_pos[:, 2])
    xy = mesh_pos[top_plane][:, :2]
    pred_T_xy = predict_temperature[:, top_plane]
    gt_T_xy = gt_temperature[:, top_plane]


    numstep_plot = 5
    timesteps = predict_temperature.shape[0]
    print(timesteps)
    interval = timesteps//numstep_plot
    plot_temperature_fields_grid(xy, pred_T_xy, gt_T_xy, [50 + interval * step for step in range(numstep_plot)], f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/figures/{testcase}.pdf")
    meltpoolxyz_ts = {} 
    total_steps = len(predict_temperature)
    for i in range(0, total_steps, 100) :
        meltpoolxyz_pred = analyze_melt_pool(mesh_pos, predict_temperature[i], plot = False)
        meltpoolxyz_gt = analyze_melt_pool(mesh_pos, gt_temperature[i], plot = False)
        meltpoolxyz_ts[i] = {"pred": meltpoolxyz_pred, "gt": meltpoolxyz_gt}
        print(i, "/", total_steps)
    plot_meltpool_predictions(meltpoolxyz_ts, total_steps, 100, f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/figures/{testcase}_meltpool.pdf")