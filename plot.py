import torch
import matplotlib.pyplot as plt
from core.utils import * 
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
    test_on = "offset"
    rollout_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{test_on}/{test_on}.pkl"
    output_dir = f"/mnt/c/Users/narun/OneDrive/Desktop/Project/Heat_MGN/rollout/{test_on}"
    png_pth = os.path.join(output_dir, f"{test_on}-temperature-time.png")
    output = load_dict_from_pkl(rollout_dir)
    print(output)
    plot_max_temperature_over_time(output["predict_temperature"], 
                                   output["gt_temperature"], 
                                   output["rmse_list"],
                                   time_step=1e-5, 
                                   title="Temperature Over Time", 
                                   save_path = png_pth)
    