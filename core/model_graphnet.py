import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, LazyLinear, Conv1d, LeakyReLU
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
from .utils import create_mesh_from_arrays
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import numpy as np
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
    def fem_loss(self, network_output, graph) :
        mesh_pos = graph.mesh_pos.detach().cpu()
        cells = graph.cells.detach().cpu()
        output_normalizer = self.get_output_normalizer()
        # delta_temperature = output_normalizer.inverse(network_output)
        # delta_temperature = network_output
        cur_temp = graph.temperature.expand(-1, self._time_window)
        next_temp = cur_temp + network_output

        mesh = create_mesh_from_arrays(mesh_pos, cells, "tetrahedron", mesh_pos.shape[1])
        u = dolfinx.fem.functionspace(mesh, ("CG", 1))

        rhocp = lambda T: 8351.910158 * (446.337248 + 0.14180844 * (T-273) - 61.431671211432 * np.e ** (-0.00031858431233904*((T-273)-525)**2) + 1054.9650568*np.e **(-0.00006287810196136*((T-273)-1545)**2))
        k = lambda T: 25   # W/(m·K)

        T_ = ufl.TrialFunction(u)
        v = ufl.TestFunction(u)
        T_n = dolfinx.fem.Function(u, name="Temperature (K)")
        T_n.x.array[:] = graph.temperature.detach().cpu().squeeze(-1)

        q = dolfinx.fem.Function(u, name="Heat Source")
        q.x.array[:] = graph.heat_source.detach().cpu().squeeze(-1)

        a = (1/self._timestep) * rhocp(T_n)*ufl.inner(T_, v)*ufl.dx + k(T_n)*ufl.inner(ufl.grad(T_), ufl.grad(v))*ufl.dx
        L = (1/self._timestep)*rhocp(T_n)*ufl.inner(T_n, v)*ufl.dx + ufl.inner(q, v)*ufl.dx
        problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "ilu"})
        stiffness = assemble_matrix(problem.a)
        stiffness.assemble()
        ai, aj, av = stiffness.getValuesCSR()
        stiffness_tensor = torch.sparse_csr_tensor(ai, aj, av, dtype = torch.float, device = self._device, requires_grad=True)
        # stiffness = stiffness.convert("dense")
        # stiffness_mat = stiffness.getDenseArray()
        load = assemble_vector(problem.L)
        load.assemble()
        load_vec = load.getArray()
        load_vec_tensor = torch.tensor(load_vec, dtype = torch.float, device = self._device, requires_grad=True)
        res = torch.mean((stiffness_tensor @ next_temp - load_vec_tensor)**2)
        return res

    def pde_loss(self, network_output, graph) :
        laplacian = cotangent_laplacian_tetrahedral(graph.mesh_pos, graph.cells)
        output_normalizer = self.get_output_normalizer()
        delta_temperature = output_normalizer.inverse(network_output)
        laplacian_T = torch.sparse.mm(laplacian, graph.temperature.squeeze(0))
        rhocp = lambda T: 8351.910158 * (446.337248 + 0.14180844 * (T-273) - 61.431671211432 * np.e ** (-0.00031858431233904*((T-273)-525)**2) + 1054.9650568*np.e **(-0.00006287810196136*((T-273)-1545)**2))
        k = lambda T: 25   # W/(m·K)
        lhs = rhocp(graph.temperature.squeeze(0)) * delta_temperature/self._timestep
        rhs = k(graph.temperature.squeeze(0)) * laplacian_T + graph.heat_source.squeeze(0)
        res = (lhs - rhs)**2
        return torch.mean(res)
    
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

def cotangent_laplacian_tetrahedral(mesh_pos, tets):
    """
    mesh_pos: [N, 3] (float tensor) - vertex positions
    tets: [M, 4] (long tensor) - tetrahedral connectivity (indices into mesh_pos)
    returns: sparse Laplacian matrix [N, N] in torch.sparse_coo_tensor format
    """
    assert mesh_pos.shape[-1] == 3, "Tetrahedral meshes require 3D positions"
    mesh_pos = mesh_pos.squeeze(0)
    tets = tets.squeeze(0)
    # Indices for tetrahedron vertices
    i0, i1, i2, i3 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    v0, v1, v2, v3 = mesh_pos[i0], mesh_pos[i1], mesh_pos[i2], mesh_pos[i3]

    def face_cotangent(p0, p1, p2, opposite):
        """
        Compute cotangent weight for a face (p0, p1, p2) with opposite vertex.
        This is based on the angle between the face normal and the opposite vertex.
        """
        # Face normal
        n = torch.cross(p1 - p0, p2 - p0, dim=1)
        n_norm = torch.norm(n, dim=1).clamp(min=1e-8)
        # Vector from face to opposite point
        vol_vec = opposite - p0
        height = (vol_vec * n).sum(dim=1).abs() / n_norm
        area = 0.5 * n_norm
        # Cotangent approximation: height / area ~ cot(angle)
        return 0.5 * height / area

    # For each face in the tetrahedron (4 faces per tetrahedron), compute cotangent weights
    cot01_2 = face_cotangent(v0, v1, v2, v3)
    cot01_3 = face_cotangent(v0, v1, v3, v2)
    cot02_3 = face_cotangent(v0, v2, v3, v1)
    cot12_3 = face_cotangent(v1, v2, v3, v0)

    rows = torch.cat([i0, i0, i0, i1, i1, i2])
    cols = torch.cat([i1, i2, i3, i2, i3, i3])
    weights = torch.cat([
        cot01_2 + cot01_3,  # (i0, i1)
        cot01_2 + cot02_3,  # (i0, i2)
        cot01_3 + cot02_3,  # (i0, i3)
        cot01_2 + cot12_3,  # (i1, i2)
        cot01_3 + cot12_3,  # (i1, i3)
        cot02_3 + cot12_3   # (i2, i3)
    ]) * 0.5

    # Since Laplacian is symmetric, duplicate for transpose entries
    all_rows = torch.cat([rows, cols])
    all_cols = torch.cat([cols, rows])
    all_weights = torch.cat([weights, weights])

    n = mesh_pos.shape[0]
    L_off = torch.sparse_coo_tensor(torch.stack([all_rows, all_cols]), all_weights, size=(n, n))

    # Diagonal entries
    diag_vals = torch.zeros(n, dtype=all_weights.dtype, device=all_weights.device).index_add(0, all_rows, all_weights)
    diag_indices = torch.arange(n, device=mesh_pos.device)
    L_diag = torch.sparse_coo_tensor(
        torch.stack([diag_indices, diag_indices]), -diag_vals, size=(n, n)
    )

    L = L_off + L_diag
    return L.coalesce()
