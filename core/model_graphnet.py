import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, LazyLinear
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from .normalization import Normalizer
import os
import torch.nn.functional as F
import torch_scatter
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import time
class GraphNetBlock(torch.nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, latent_size, in_size1, in_size2):
        super().__init__()
        self._latent_size = latent_size
        
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

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        senders = graph.senders
        receivers = graph.receivers     
        node_latents = graph.node_latents
        mesh_edge_latents = graph.mesh_edge_latents
        new_mesh_edge_latents = self.mesh_edge_net(torch.cat([node_latents[senders], node_latents[receivers], mesh_edge_latents], dim=-1))
        aggr = torch_scatter.scatter_add(new_mesh_edge_latents.float(), receivers, dim=0, dim_size=node_latents.shape[0])
        new_node_latents = self.node_feature_net(torch.cat([node_latents, aggr], dim=-1))
        # apply node function

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
                 message_passing_steps,
                 name='EncodeProcessDecode'):
        super(EncodeProcessDecode, self).__init__()   
        self._node_feature_size = node_feature_size   
        self._mesh_edge_feature_size = mesh_edge_feature_size
        self._latent_size = latent_size
        self._output_size = output_size      
        self._message_passing_steps = message_passing_steps       
        self._output_normalizer = Normalizer(size=output_size, name='output_normalizer')
        self._node_features_normalizer = Normalizer(size = node_feature_size, name='node_features_normalizer')
        self._mesh_edge_normalizer = Normalizer(size = mesh_edge_feature_size, name='mesh_edge_normalizer')
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
            self.graphnet_blocks.append(GraphNetBlock(self._latent_size, self._latent_size*3, self._latent_size*2))
        # Decoding net (MLP) for node_features (output)
        # ND: "Node features Decoding"
        self.node_decode_net = Sequential(Linear(self._latent_size,self._latent_size),
                        ReLU(),
                        Linear(self._latent_size,self._output_size))
                        
    
    def encoder(self, graph) :
        node_features = self._build_node_latent_features(graph.node_type, graph.temperature, graph.heat_source)
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
        decoded_nodes = self.node_decode_net(latent_graph.node_latents)    
        return decoded_nodes
    def get_output_normalizer(self):
        return self._output_normalizer
    def predict(self, graph) :
        start1 = time.perf_counter()
        self.eval()
        end1 = time.perf_counter()

        start2 = time.perf_counter()
        network_output = self.forward(graph)
        end2 = time.perf_counter()
        start3 = time.perf_counter()
        delta_temperature = self._output_normalizer.inverse(network_output)
        end3 = time.perf_counter()
        cur_temp = graph.temperature
        next_temp = cur_temp + delta_temperature
        print(
        f"eval: {end1 - start1:.6f}s, "
        f"forward: {end2 - start2:.6f}s, "
        f"inv_norm: {end3 - start3:.6f}s, "
        )
        return next_temp
    def loss(self, output, graph) :
        current_temp = graph.temperature
        target_temp = graph.target_temperature
        delta_temp = target_temp - current_temp
        target_temperature_normalizer = self.get_output_normalizer()
        target_temperature_normalized = target_temperature_normalizer(delta_temp)
        # target_normalized_stress = stress_normalizer(target_stress)

        node_type = graph.node_type

        #exclude loss from rigid body
        loss_mask = node_type == 0
        error = torch.sum((target_temperature_normalized - output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss
    def fem_physical_loss(self, network_output, graph) :
        return # to be developed
    def pde_physical_loss(self, network_output, graph) :
        return # to be developed
    
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
    
    def _build_node_latent_features(self, node_type, temp_field, heat_source_field) :
        node_type_onehot = F.one_hot(node_type).to(torch.float)
        node_latent_features = torch.cat(
            (temp_field, heat_source_field, node_type_onehot), 
            dim = -1
            )
        return node_latent_features
    def _build_mesh_edge_features(self, mesh_pos, temp_field, senders, receivers) :
        relative_mesh_pos = mesh_pos[senders] - mesh_pos[receivers]
        nodal_temperature_gradient = temp_field[senders] - temp_field[receivers]
        norm_rel_mesh_pos = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)
        # concatenate the mesh edges data together
        mesh_edge_features = torch.cat(
            (relative_mesh_pos, norm_rel_mesh_pos, nodal_temperature_gradient), 
            dim = -1
            )
        return mesh_edge_features

    def compute_tetra_transient_residual(
        node_coords, connectivity, T, T_prev, Q, dt,
        rho=1.0, cp=1.0, k=1.0
    ):
        """
        Compute residuals per element-node for transient heat equation (weak form) using PyTorch.

        All inputs must be torch tensors on the same device.
        T and T_prev: (N_nodes,) with requires_grad=True for T
        Returns: (N_elements, 4) residuals
        """
        N_elements = connectivity.shape[0]
        device = T.device
        element_residuals = []

        for e in range(N_elements):
            elem = connectivity[e]
            n = elem
            coords = node_coords[n]     # (4, 3)
            Te = T[n]                   # (4,)
            Tpe = T_prev[n]             # (4,)
            Qe = Q[n]                   # (4,)

            # Build shape function matrix
            X = torch.ones((4, 4), device=device)
            X[:, 1:] = coords

            detJ = torch.det(X)
            volume = torch.abs(detJ) / 6.0

            # Compute shape gradients
            grads = torch.zeros((4, 3), device=device)
            for i in range(4):
                mat = torch.cat([X[:i], X[i+1:]], dim=0)
                sign = -1 if i % 2 else 1
                grads[i, 0] = sign * torch.det(mat[:, [1, 2, 3]])
                grads[i, 1] = sign * torch.det(mat[:, [0, 2, 3]])
                grads[i, 2] = sign * torch.det(mat[:, [0, 1, 3]])
            grads /= detJ  # ∇N_i

            # Compute ∇T and ∂T/∂t
            grad_T = grads.T @ Te  # ∇T
            dT_dt = (Te - Tpe) / dt

            Ni = torch.ones(4, device=device) / 4  # shape fn at centroid
            Q_avg = torch.mean(Qe)

            # Residual per node in element
            res = torch.zeros(4, device=device)
            for i in range(4):
                term_time = rho * cp * Ni[i] * torch.mean(dT_dt)
                term_diff = grads[i] @ (k * grad_T)
                term_source = Ni[i] * Q_avg
                res[i] = (term_time + term_diff - term_source) * volume

            element_residuals.append(res)

        return torch.stack(element_residuals)

