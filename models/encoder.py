# /home/zdy/Project2/models/encoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import math


def DifferentiableSphericalHarmonics(vectors: torch.Tensor, lmax: int = 2):
    if vectors.dim() == 3:
        x = vectors[..., 0:1]
        y = vectors[..., 1:2]
        z = vectors[..., 2:3]
    else:
        x = vectors[:, 0:1]
        y = vectors[:, 1:2]
        z = vectors[:, 2:3]

    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-8)
    r2 = r ** 2

    sh = [torch.full_like(r, 0.28209479177)]

    if lmax > 0:
        sh.append(-0.4886025119 * y / r)
        sh.append(-0.4886025119 * z / r)
        sh.append(-0.4886025119 * x / r)

    if lmax > 1:
        sh.append(0.5 * math.sqrt(3. / math.pi) * (x * y) / r2)
        sh.append(math.sqrt(3. / (4. * math.pi)) * (y * z) / r2)
        sh.append(0.25 * math.sqrt(5. / math.pi) * (2 * z ** 2 - x ** 2 - y ** 2) / r2)
        sh.append(math.sqrt(3. / (4. * math.pi)) * (x * z) / r2)
        sh.append(0.25 * math.sqrt(3. / math.pi) * (x ** 2 - y ** 2) / r2)

    res = torch.cat(sh, dim=-1)
    return res

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation="ReLU"):
        super().__init__()
        act_fn = getattr(nn, activation, nn.ReLU)
        layers = []
        curr = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(act_fn())
            curr = h
        layers.append(nn.Linear(curr, out_dim))
        self.mlp = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

# --- Encoder Block ---
class CooperativeSE3EncoderBlock(MessagePassing):
    def __init__(self, scalar_dim, vector_dim, edge_scalar_dim, l_max_sh=2, mlp_hidden_dims=[128, 128]):
        super().__init__(aggr='add', node_dim=-2)
        self.l_max_sh = l_max_sh
        sh_dim = (l_max_sh + 1) ** 2
        self.message_creation_mlp = MLP(scalar_dim * 2 + edge_scalar_dim, mlp_hidden_dims, scalar_dim)
        self.attention_mlp = MLP(scalar_dim, mlp_hidden_dims, 1)
        self.node_scalar_update_mlp = MLP(scalar_dim * 2, mlp_hidden_dims, scalar_dim)
        self.node_vector_update_mlp = MLP(scalar_dim, mlp_hidden_dims, 1, activation="SiLU")
        self.edge_scalar_update_mlp = MLP(edge_scalar_dim + sh_dim, mlp_hidden_dims, edge_scalar_dim)

    def forward(self, s, v, edge_index, edge_s, pos):
        if edge_index is None or edge_index.numel() == 0:
            return s, v, edge_s

        row, col = edge_index

        edge_v = pos[col] - pos[row]
        dist = torch.sqrt(torch.sum(edge_v ** 2, dim=-1, keepdim=True) + 1e-8)
        unit_edge_v = edge_v / dist

        hij = self.message_creation_mlp(torch.cat([s[row], s[col], edge_s], dim=-1))

        alpha_ij = softmax(self.attention_mlp(hij), row)  # [E, 1]

        mi = scatter_add(alpha_ij * hij, row, dim=0, dim_size=s.size(0))  # [N, hidden]

        s = s + self.node_scalar_update_mlp(torch.cat([s, mi], dim=-1))

        vi = scatter_add(alpha_ij * unit_edge_v, row, dim=0, dim_size=s.size(0))  # [N, 3]

        v_update = self.node_vector_update_mlp(mi).view(-1, 1) * vi
        v = v + v_update

        sh_feat = DifferentiableSphericalHarmonics(unit_edge_v, lmax=self.l_max_sh)  # [E, 9]

        combined = torch.cat([edge_s, sh_feat], dim=-1)
        edge_s = edge_s + self.edge_scalar_update_mlp(combined)

        return s, v, edge_s

# --- Top-Level Encoder ---
class CooperativeSE3Encoder(nn.Module):
    def __init__(self, in_scalar_dim, in_edge_scalar_dim,
                 hidden_scalar_dim, hidden_vector_dim,
                 num_layers, l_max_sh=2, mlp_hidden_dims=[128, 128]):
        super().__init__()
        self.in_scalar_dim = in_scalar_dim
        self.in_edge_scalar_dim = in_edge_scalar_dim
        self.hidden_scalar_dim = hidden_scalar_dim

        self.node_in = MLP(in_scalar_dim, mlp_hidden_dims, hidden_scalar_dim)
        self.edge_in = MLP(in_edge_scalar_dim, mlp_hidden_dims, hidden_scalar_dim)

        self.layers = nn.ModuleList([
            CooperativeSE3EncoderBlock(
                hidden_scalar_dim, hidden_vector_dim, hidden_scalar_dim, l_max_sh, mlp_hidden_dims
            ) for _ in range(num_layers)
        ])

    def forward(self, s, pos, edge_index, edge_s=None):
        num_edges = edge_index.size(1) if edge_index is not None else 0

        if edge_s is None or not torch.is_tensor(edge_s) or edge_s.dim() != 2 or edge_s.size(
                -1) != self.in_edge_scalar_dim:
            edge_s = torch.zeros((num_edges, self.in_edge_scalar_dim), device=s.device, dtype=s.dtype)

        if edge_s.size(0) != num_edges:
            edge_s = torch.zeros((num_edges, self.in_edge_scalar_dim), device=s.device, dtype=s.dtype)

        s = s.float()
        pos = pos.float()
        edge_s = edge_s.float()

        v = torch.zeros((s.size(0), 3), device=s.device, dtype=s.dtype)

        s = self.node_in(s)  # 23 -> 128
        edge_s = self.edge_in(edge_s)  # 8 -> 128

        if num_edges > 0:
            for layer in self.layers:
                if edge_s.size(-1) != 128:
                    raise RuntimeError(f"Dimension fatal error! edge_s became {edge_s.shape} before layer.")

                s, v, edge_s = layer(s, v, edge_index, edge_s, pos)

        return s, v, edge_s, None
