import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .normalization import Normalizer
from scipy.spatial import Delaunay
from .normalization import Normalizer
import torch_scatter
from torch_geometric.nn import knn_graph
from torch_geometric.utils import softmax
import os
def edge_smoothness_loss(pred, edge_index):
    """Smoothness regularizer for spatial diffusion.
    Penalizes differences between connected node values.
    Args:
        pred: [num_nodes] or [num_nodes, feature_dim]
        edge_index: [2, num_edges]
    """
    i, j = edge_index
    diff = pred[i] - pred[j]
    return diff


def smoothness_on_delta(delta_pred, delta_gt, edge_index):
    """Smoothness regularizer on predicted ΔT = T_pred - T_prev."""
    grad_delta_pred = edge_smoothness_loss(delta_pred, edge_index)
    grad_delta_gt = edge_smoothness_loss(delta_gt, edge_index)
    diff = torch.mean((grad_delta_pred - grad_delta_gt)**2)
    return diff
def MLP(in_dim, out_dim, hidden_dims=(128, 128), activate_final=False, layer_norm=False):
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    if activate_final:
        layers.append(nn.ReLU())
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)

class EdgeNodeMessagePassing(MessagePassing):
    """Custom MP layer that:
      1) updates edge attributes using (x_i, x_j, e_ij)
      2) computes messages using updated edge attributes and x_j
      3) aggregates messages and updates node attributes

    This follows the Encode-Process-Decode pattern where the processing layer is a message-passing step.
    """

    def __init__(self, hidden_dim, attention):
        super().__init__(aggr='add')  # 'add' aggregation
        self.attention = attention
        # Node attention: compute attention scores for neighbors
        if self.attention :
            self.attn_lin = torch.nn.Linear(hidden_dim, hidden_dim)
        # edge update: (x_i, x_j, e_ij) -> e'_ij
        self.edge_mlp = MLP(hidden_dim * 2 + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
        # node update: (x_i, aggregated_message) -> x'_i
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
    def forward(self, node_feat, edge_index, edge_feat):
        
        # Node update
        new_node_features = self.propagate(edge_index, x= node_feat, edge_attr = edge_feat)        
        
        # Edge update
        row, col = edge_index
        new_edge_features = self.edge_mlp(torch.cat([node_feat[row], node_feat[col], edge_feat], dim=-1))
        
        # Add residuals
        new_node_features = new_node_features + node_feat
        new_edge_features = new_edge_features + edge_feat     
                
        return new_node_features, new_edge_features
    
    def message(self, x_i, x_j, edge_attr, index):
        # Compute attention score
        if self.attention :
            alpha = (self.attn_lin(x_i) * self.attn_lin(x_j)).sum(dim=-1)  # [E]
            alpha = F.leaky_relu(alpha)
            alpha = softmax(alpha, index=index)  # normalize per target node
            alpha = alpha.unsqueeze(-1)  # [E, 1]

            # Message is neighbor feature weighted by attention
            msg = alpha * self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        else :
            features = torch.cat([x_i, x_j, edge_attr], dim=-1)        
            msg = self.edge_mlp(features)
        return msg
    def update(self, aggr_out, x):
        # aggr_out has shape [num_nodes, out_channels]        
        tmp = torch.cat([aggr_out, x], dim=-1)                
       
        # Step 5: Return new node embeddings.        
        return self.node_mlp(tmp)
    
class Swish(torch.nn.Module):
    """Swish activation function."""
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
class EncodeProcessDecode(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_size=128,
                 process_steps=3,
                 node_out_dim=1, 
                 time_window=5,
                 attention=False,
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])

        # Decoder MLPs
        self.node_decoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size//2, 1),
            Swish(),
            nn.Conv1d(hidden_size//2, node_out_dim * time_window, 1)
        )
        self.device = device
        self.time_window = time_window
        self.node_out_dim = node_out_dim
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(time_window, device)
    def forward(self, graph):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(graph)
        e = self._build_edge_features(graph)
        x_h = self.node_encoder(self.node_features_normalizer(x))
        e_h = self.edge_encoder(self.edge_features_normalizer(e))

        # Process (multiple message-passing steps)
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, graph.edge_index, e_h)

        x_h = x_h.unsqueeze(-1)  
        decoded = self.node_decoder(x_h).squeeze(-1)
        dt = torch.arange(1, self.time_window + 1).to(device=self.device)
        delta = decoded.reshape(-1, self.time_window)

        return dt.unsqueeze(0) * delta
    def _build_node_features(self, graph) :
        u = graph.temperature
        u_dot = (graph.temperature - graph.temperature_prev)

        q = graph.heat_source 
        q_dot = (graph.heat_source - graph.heat_source_prev)
        q_next = graph.heat_source_next

        x = torch.cat([u.unsqueeze(-1), u_dot.unsqueeze(-1), q.unsqueeze(-1), q_dot.unsqueeze(-1), q_next], dim = -1)
        return x
    def _build_edge_features(self, graph) :
        senders, receivers = graph.edge_index[0], graph.edge_index[1]
        rel_position = graph.mesh_pos[senders, :] - graph.mesh_pos[receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)
        rel_temperature = graph.temperature[senders] - graph.temperature[receivers]
        edge_features = torch.cat([rel_position, distance, rel_temperature.unsqueeze(-1)], dim=-1)
        return edge_features
    def loss(self, graph):
        target = graph.target_temperature
        curr = graph.temperature
        target_delta = target - curr.unsqueeze(-1)
        target_delta_normalize = self.output_normalizer(target_delta)
        pred_delta = self.forward(graph)
        error = (pred_delta - target_delta_normalize)**2
        # smoothness_loss = torch.tensor(0, device = self.device)
        mse = torch.mean(error)
        return mse
    def predict(self, graph):
        pred_delta_normalized = self.forward(graph)
        pred_delta = self.output_normalizer.inverse(pred_delta_normalized)
        curr = graph.temperature
        pred = curr.unsqueeze(-1) + pred_delta
        return pred
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_features_normalizer, os.path.join(path, "edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
