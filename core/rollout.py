import torch
from copy import deepcopy
# from torch_geometric.data import Data
from tqdm import tqdm
# from copy import deepcopy
from torch_geometric.data import Data

def rollout(model, data):
    device = model.device
    rollout_preds = []
    load = []
    time_window = model.time_window
    curr_graph = deepcopy(data[1]) # deep copy ensures clean start

    with torch.no_grad():
        loop = tqdm(range(1, len(data), time_window), desc="Rollout")

        for t in loop:
            # Replace only time-dependent fields for this step
            # print(t)
            curr_graph.heat_source_prev = data[t-1].heat_source.clone()
            curr_graph.heat_source = data[t].heat_source.clone()
            heat_sources_next = []
            for w in range(time_window):
                idx = t + w + 1
                if idx < len(data):
                    heat_sources_next.append(data[idx].heat_source.clone())
                else:
                    # pad with zeros of the same shape
                    heat_sources_next.append(torch.zeros_like(data[0].heat_source))

            curr_graph.heat_source_next = torch.stack(heat_sources_next).T.clone()
            if t == 1:
                curr_graph.temperature_prev = data[0].temperature.clone()
                curr_graph.temperature = data[1].temperature.clone()
            # Predict next state
            pred_next = model.predict(curr_graph.to(device))
            if model.time_window > 1 :
                before_last_step = pred_next[:, -2].clone()
            else :
                before_last_step = curr_graph.temperature.clone()
            last_step = pred_next[:, -1].clone()
            # Update for next step
            curr_graph.temperature_prev = before_last_step
            curr_graph.temperature = last_step

            rollout_preds.append(pred_next)

        rollout_preds = torch.cat(rollout_preds, dim = -1)[:, :len(data) - 2].T
        rollout_gts = torch.stack([data[t].temperature for t in range(2, len(data))])
        heat_source = torch.stack([data[t].heat_source for t in range(2, len(data))])
        # Compute error
        check = torch.max(heat_source)
        error = (rollout_preds.to(device) - rollout_gts.to(device)) ** 2
        rmse_temperature = torch.sqrt(torch.mean(error))

        print(f"RMSE T: {rmse_temperature:.6f}")

    return {
        "pred": rollout_preds,
        "gt": rollout_gts,
        "heat_source": heat_source,
        "mesh_pos": data[0]["mesh_pos"],
        "cells": data[0]["cells"],
        "rmse_T": rmse_temperature
    }