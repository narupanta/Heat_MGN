import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, LazyLinear, Conv1d, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from .normalization import Normalizer
import os
import torch.nn.functional as F
import torch_scatter
import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import time
from .utils import create_mesh_from_arrays
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import numpy as np
from scipy.sparse import coo_matrix
class Swish(torch.nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)

class GraphNetBlock(torch.nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, latent_size, in_size1, in_size2, attention = None):
        super().__init__()
        self._latent_size = latent_size
        self.attention = attention
        # update_mesh_edge_net: e_m ij' = f1(xi, xj, e_m ij)
        self.mesh_edge_net = Sequential(Linear(in_size1,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))
        
        # update_node_features net (MLP): xi' = f2(xi, sum(e_m ij'))
        self.node_feature_net = Sequential(Linear(in_size2,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))    
        if self.attention :
            self.attention_layer = Sequential(LazyLinear(1),
                                            LeakyReLU(negative_slope=0.2)
                                            )
    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        senders = graph.senders
        receivers = graph.receivers     
        node_latents = graph.node_latents
        mesh_edge_latents = graph.mesh_edge_latents
        new_mesh_edge_latents = self.mesh_edge_net(torch.cat([node_latents[:, senders, :], node_latents[:, receivers, :], mesh_edge_latents], dim=-1))
        if self.attention :
            attention_input = self.attention_layer(new_mesh_edge_latents)
            attention = F.softmax(attention_input, dim=0)
            new_mesh_edge_latents = attention * new_mesh_edge_latents
        aggr = torch_scatter.scatter_add(new_mesh_edge_latents.float(), receivers, dim=1)
        new_node_latents = self.node_feature_net(torch.cat([node_latents, aggr], dim=-1))

        # add residual connections
        new_node_latents += node_latents
        new_mesh_edge_latents += mesh_edge_latents

        return Data(senders = senders,
                    receivers = receivers, 
                    node_latents = new_node_latents, 
                    mesh_edge_latents = new_mesh_edge_latents,)  
    
class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 node_feature_size,
                 mesh_edge_feature_size,
                 output_size,
                 latent_size,
                 timestep,
                 time_window,                 
                 message_passing_steps,
                 attention,
                 device,
                 name='EncodeProcessDecode'):
        super(EncodeProcessDecode, self).__init__()   
        self._node_feature_size = node_feature_size   
        self._mesh_edge_feature_size = mesh_edge_feature_size
        self._latent_size = latent_size
        self._output_size = output_size      
        self._message_passing_steps = message_passing_steps  
        self._attention = attention
        self._time_window = time_window
        self._timestep = timestep     
        self._output_normalizer = Normalizer(time_window, output_size, 'output_normalizer', device)
        self._node_features_normalizer = Normalizer(1, node_feature_size, 'node_features_normalizer', device)
        self._mesh_edge_normalizer = Normalizer(1, mesh_edge_feature_size, 'mesh_edge_normalizer', device)
        self._device = device
        # Encoding net (MLP) for node_features
        self.node_encode_net = Sequential(Linear(self._node_feature_size, self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))               
               
        # Encoding net (MLP) for edge_features
        self.mesh_edge_encode_net = Sequential(Linear(self._mesh_edge_feature_size, self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))            
        

        self.graphnet_blocks = torch.nn.ModuleList()
        for _ in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(self._latent_size, self._latent_size*3, self._latent_size*2, self._attention))
        # Decoding net (MLP) for node_features (output)
        # ND: "Node features Decoding"
        self.node_decode_net = Sequential(Conv1d(self._latent_size, 8, 1),
                                          Swish(),
                                          Conv1d(8, self._time_window, 1))
        # self.node_decode_net = Sequential(Linear(self._latent_size,self._latent_size),
        #                 ReLU(),
        #                 Linear(self._latent_size,self._output_size))
                        
    
    def encoder(self, graph) :
        node_features = self._build_node_latent_features(graph.node_type, graph.heat_source, graph.temperature)
        mesh_edge_features = self._build_mesh_edge_features(graph.mesh_pos, graph.temperature, graph.senders, graph.receivers)    
        
        node_latents = self.node_encode_net(self._node_features_normalizer(node_features))          
        mesh_edge_latents = self.mesh_edge_encode_net(self._mesh_edge_normalizer(mesh_edge_features))  
        
        return Data(senders = graph.senders, 
                    receivers = graph.receivers, 
                    node_latents = node_latents, 
                    mesh_edge_latents = mesh_edge_latents)
    def forward(self, graph):
        """Encodes and processes a graph, and returns node features."""                     
        # Encoding Layer  
        latent_graph = self.encoder(graph)  
        # Process Layer
        for graphnet_block in self.graphnet_blocks:
            latent_graph = graphnet_block(latent_graph)
        """Decodes node features from graph."""   
        # Decoding node features
        node_latents = latent_graph.node_latents.permute(0, 2, 1)
        decoded = self.node_decode_net(node_latents).permute(0, 2, 1)
        dt = torch.arange(1, self._time_window + 1).repeat_interleave(self._output_size).to(self._device)
        delta = (decoded * dt).reshape(-1, self._time_window, self._output_size).permute(1, 0, 2)
        # decoded_nodes = self.node_decode_net(latent_graph.node_latents)    
        return delta
    def get_output_normalizer(self):
        return self._output_normalizer
    def predict(self, graph) :
        self.eval()
        network_output = self.forward(graph)
        output_normalizer = self.get_output_normalizer()
        delta_temperature = output_normalizer.inverse(network_output)
        # delta_temperature = network_output
        cur_temp = graph.temperature
        next_temp = cur_temp + delta_temperature
        return next_temp
    def loss(self, output, graph) :
        initial_temp = graph.temperature                       # (num_nodes,)
        target_temp = graph.target_temperature                 # (num_nodes, time_window)
        delta_temp = target_temp - initial_temp  # (num_nodes, time_window)

        normalizer = self.get_output_normalizer()
        target_temp_normalized = normalizer(delta_temp)        # (num_nodes, time_window)

        node_type = graph.node_type                            # (num_nodes,)
        loss_mask = node_type == 0                             # (num_nodes,)

        error = (output - target_temp_normalized) ** 2               # (num_nodes,)
        loss = torch.max(error[:, loss_mask.squeeze(0), :], dim = 1).values      # scalar
        return torch.mean(loss)

    def physic_loss(self, network_output, graph) :
        normalizer = self.get_output_normalizer()
        # delta_temperature = network_output
        cur_temp = graph.temperature
        k = lambda T: 25
        rhocp = lambda T: 8351.910158 * (446.337248 + 0.14180844 * (T-273) - 61.431671211432 * np.e ** (-0.00031858431233904*((T-273)-525)**2) + 1054.9650568*np.e **(-0.00006287810196136*((T-273)-1545)**2))
        dt_window = torch.arange(1, self._time_window + 1).repeat_interleave(self._output_size).to(self._device) * self._timestep
        temp_laplacian = fem_laplacian_tetra(graph.mesh_pos.squeeze(0), graph.cells.squeeze(0), cur_temp.squeeze(0).squeeze(-1), self._device)
        physical = 1/rhocp(cur_temp) * (k(cur_temp) * temp_laplacian.unsqueeze(0).unsqueeze(-1) + graph.heat_source)
        delta_T_physic_onestep = normalizer(dt_window.reshape(-1, 1, 1) * physical)

        node_type = graph.node_type                            # (num_nodes,)
        loss_mask = node_type == 0                             # (num_nodes,)

        error = (network_output - delta_T_physic_onestep) ** 2               # (num_nodes,)
        loss = torch.mean(error[:, loss_mask.squeeze(0), :], dim = 1)      # scalar

        return torch.mean(loss)
    
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self._output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self._node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self._mesh_edge_normalizer, os.path.join(path, "mesh_edge_features_normalizer.pth"))
        
    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self._output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self._node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self._mesh_edge_normalizer = torch.load(os.path.join(path, "mesh_edge_features_normalizer.pth"))
    
    def _build_node_latent_features(self, node_type, heat_source_field, temp_field) :
        # node_type_onehot = F.one_hot(node_type).to(torch.float)
        node_latent_features = torch.cat(
            (heat_source_field, temp_field), 
            dim = -1
            )
        return node_latent_features
    def _build_mesh_edge_features(self, mesh_pos, temp_field, senders, receivers) :
        relative_mesh_pos = mesh_pos[:, senders, :] - mesh_pos[:, receivers, :]
        nodal_temperature_gradient = temp_field[:, senders, :] - temp_field[:, receivers, :]
        norm_rel_mesh_pos = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)
        # concatenate the mesh edges data together
        mesh_edge_features = torch.cat(
            (relative_mesh_pos, norm_rel_mesh_pos, nodal_temperature_gradient), 
            dim = -1
            )
        return mesh_edge_features

def tetra_volume(v0, v1, v2, v3):
    return torch.abs(torch.einsum('ij,ij->i', v1 - v0, torch.cross(v2 - v0, v3 - v0))) / 6.0

def grad_phi_matrix(verts, device):  # verts: (M, 4, 3)
    v0, v1, v2, v3 = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]

    # Matrix D = [v1 - v0, v2 - v0, v3 - v0] — shape (M, 3, 3)
    D = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=2)  # (M, 3, 3)
    Dinv = torch.linalg.inv(D)  # (M, 3, 3)

    # Gradients of barycentric basis functions
    grads = torch.zeros((verts.shape[0], 4, 3), dtype=torch.float32, device = device)
    grads[:, 1:] = Dinv.permute(0, 2, 1)
    grads[:, 0] = -grads[:, 1:].sum(dim=1)
    return grads  # shape (M, 4, 3)

def fem_laplacian_tetra(V, T, u, device):
    N = V.shape[0]
    M = T.shape[0]

    verts = V[T]  # shape (M, 4, 3)
    grads = grad_phi_matrix(verts, device)  # shape (M, 4, 3)
    vol = tetra_volume(*verts.permute(1, 0, 2))  # shape (M,)

    lap = torch.zeros(N, dtype=torch.float32, device = device)
    mass = torch.zeros(N, dtype=torch.float32, device = device)

    for a in range(4):
        for b in range(4):
            # Local stiffness: grad phi_a · grad phi_b * volume
            contrib = (grads[:, a] * grads[:, b]).sum(dim=1) * vol
            if a == b:
                mass.index_add_(0, T[:, a], vol / 4)  # lumped mass
            lap.index_add_(0, T[:, a], contrib * (u[T[:, b]] - u[T[:, a]]))

    lap /= (mass + 1e-8)
    return lap

