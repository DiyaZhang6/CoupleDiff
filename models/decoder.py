# /home/zdy/Project2/code/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation="ReLU"):
        super().__init__()
        activation_map = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}
        act_fn = activation_map.get(activation, nn.ReLU)
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(act_fn())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=20.0, num_kernels=16):
        super().__init__()
        offset = torch.linspace(start, stop, num_kernels)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist):
        # 增加 epsilon 保护，防止 dist 为 nan
        return torch.exp(self.coeff * torch.pow(dist - self.offset, 2))


class EquivariantDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        rbf_kernels,
        msg_hidden_dims,
        weight_hidden_dims,
        coord_update_aggregation="mean",
        coord_update_max_step=1.0,
        coord_scale_init=0.0,
        coord_scale_max=0.1,
        detach_coordinate_features=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.coord_update_aggregation = coord_update_aggregation
        self.coord_update_max_step = float(coord_update_max_step)
        self.coord_scale_max = float(coord_scale_max)
        self.detach_coordinate_features = bool(detach_coordinate_features)

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim)

        self.phi_m = MLP(embed_dim * 2 + rbf_kernels, msg_hidden_dims, embed_dim)
        # 坐标更新权重网络
        self.psi = MLP(embed_dim, weight_hidden_dims, 1)

        # 增加一个可学习的缩放因子，初始化为 0，让模型从恒等映射开始学习
        init_ratio = max(min(float(coord_scale_init) / max(self.coord_scale_max, 1e-8), 0.999), -0.999)
        self.coord_scale_raw = nn.Parameter(torch.atanh(torch.tensor([init_ratio], dtype=torch.float32)))

    def _forward_impl(self, h, r, edge_index, rbf_module):
        row, col = edge_index
        N = h.size(0)

        # --- 1. 安全的 Equivariant Attention ---
        q = self.w_q(h)
        k = self.w_k(h)
        v = self.w_v(h)

        # 计算 logits
        attn_logits = (q[row] * k[col]).sum(dim=-1) / (self.embed_dim ** 0.5)

        # 【修复 NaN】使用 Log-Sum-Exp 技巧防止 Softmax 爆炸
        max_logits = scatter_max(attn_logits, row, dim=0, dim_size=N)[0]
        attn_exp = torch.exp(attn_logits - max_logits[row])
        attn_sum = scatter_add(attn_exp, row, dim=0, dim_size=N) + 1e-6
        attn_probs = attn_exp / attn_sum[row]

        h_tilde = scatter_add(attn_probs.unsqueeze(-1) * v[col], row, dim=0, dim_size=N)
        h = self.ln_attn(h + h_tilde)

        if self.coord_scale_max <= 0:
            return h, r

        # --- 2. Message Passing ---
        rel_pos = r[row] - r[col]
        # 【修复 NaN】防止距离为 0 导致梯度爆炸
        d_ij = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-8)

        h_coord = h.detach() if self.detach_coordinate_features else h
        rbf_feat = rbf_module(d_ij)
        m_ij = self.phi_m(torch.cat([h_coord[row], h_coord[col], rbf_feat], dim=-1))
        del rbf_feat

        # --- 3. 稳定的坐标更新 (Coordinate Update) ---
        w_ij = self.psi(m_ij)

        # 【核心修复】限制单层位移幅度。使用 tanh 将输出限制在 [-1, 1]
        # 这样单层单对原子最大位移不会超过 1.0 埃
        unit_dir = rel_pos / d_ij
        delta_r_ij = torch.tanh(w_ij) * self.coord_update_max_step * unit_dir

        # 聚合位移
        if self.coord_update_aggregation == "add":
            delta_r = scatter_add(delta_r_ij, row, dim=0, dim_size=N)
        elif self.coord_update_aggregation == "mean":
            delta_r = scatter_mean(delta_r_ij, row, dim=0, dim_size=N)
        else:
            raise ValueError(f"Unsupported coord_update_aggregation: {self.coord_update_aggregation}")

        # 使用可学习缩放因子，进一步稳定初期训练
        coord_scale = torch.tanh(self.coord_scale_raw).to(dtype=delta_r.dtype) * self.coord_scale_max
        r = r + delta_r * coord_scale

        return h, r

    def forward(self, h, r, edge_index, rbf_module):
        if self.coord_scale_max > 0:
            with torch.cuda.amp.autocast(enabled=False):
                h_out, r_out = self._forward_impl(h.float(), r.float(), edge_index, rbf_module)
            return h_out.to(dtype=h.dtype), r_out.to(dtype=r.dtype)
        return self._forward_impl(h, r, edge_index, rbf_module)


class StructureDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        params = config['decoder_params']
        self.embed_dim = params['embed_dim']
        self.num_layers = params['num_layers']
        self.detach_coordinate_update = bool(params.get('detach_coordinate_update', True))
        self.direct_delta_enabled = bool(params.get('direct_delta_enabled', False))
        self.direct_delta_include_position = bool(params.get('direct_delta_include_position', True))
        self.direct_delta_detach_features = bool(params.get('direct_delta_detach_features', True))
        self.direct_delta_max_step = float(params.get('direct_delta_max_step', 1.0))
        self.direct_delta_scale_max = float(params.get('direct_delta_scale_max', 0.0))

        self.w_0 = nn.Linear(self.embed_dim, self.embed_dim)
        self.rbf_expansion = GaussianSmearing(
            stop=params['rbf']['d_max'],
            num_kernels=params['rbf']['num_kernels']
        )

        self.layers = nn.ModuleList([
            EquivariantDecoderLayer(
                self.embed_dim,
                params['num_heads'],
                params['rbf']['num_kernels'],
                params['message_mlp']['hidden_dims'],
                params['coord_weight_mlp']['hidden_dims'],
                coord_update_aggregation=params.get('coord_update_aggregation', 'mean'),
                coord_update_max_step=params.get('coord_update_max_step', 1.0),
                coord_scale_init=params.get('coord_scale_init', 0.0),
                coord_scale_max=params.get('coord_scale_max', 0.1),
                detach_coordinate_features=params.get('detach_coordinate_features', True),
            ) for _ in range(self.num_layers)
        ])

        if self.direct_delta_enabled and self.direct_delta_scale_max > 0:
            direct_delta_cfg = params.get('direct_delta_mlp', {})
            direct_delta_hidden_dims = direct_delta_cfg.get('hidden_dims', params['coord_weight_mlp']['hidden_dims'])
            direct_delta_in_dim = self.embed_dim + (3 if self.direct_delta_include_position else 0)
            self.direct_delta_head = MLP(direct_delta_in_dim, direct_delta_hidden_dims, 3)
            direct_scale_init = float(params.get('direct_delta_scale_init', 0.0))
            direct_init_ratio = max(
                min(direct_scale_init / max(self.direct_delta_scale_max, 1e-8), 0.999),
                -0.999,
            )
            self.direct_delta_scale_raw = nn.Parameter(
                torch.atanh(torch.tensor([direct_init_ratio], dtype=torch.float32))
            )
        else:
            self.direct_delta_head = None

    def forward(self, e_f_0, r_init, batch_index, edge_index):
        # 【防护】防止输入本身包含 NaN
        if torch.isnan(e_f_0).any() or torch.isnan(r_init).any():
            # 这里的逻辑可以根据你的训练循环调整，目前先做简单处理
            e_f_0 = torch.nan_to_num(e_f_0)
            r_init = torch.nan_to_num(r_init)

        h = self.w_0(e_f_0)
        r = r_init

        for layer in self.layers:
            if self.detach_coordinate_update:
                h, r_next = layer(h, r.detach(), edge_index, self.rbf_expansion)
                r = r_next
            else:
                h, r = layer(h, r, edge_index, self.rbf_expansion)
            # 每层结束后检查是否有 NaN 逃逸
            if torch.isnan(r).any():
                r = torch.nan_to_num(r)

        if self.direct_delta_head is not None:
            h_delta = h.detach() if self.direct_delta_detach_features else h
            if self.direct_delta_include_position:
                delta_input = torch.cat([h_delta, r.detach()], dim=-1)
            else:
                delta_input = h_delta
            direct_delta = torch.tanh(self.direct_delta_head(delta_input)) * self.direct_delta_max_step
            direct_scale = torch.tanh(self.direct_delta_scale_raw).to(dtype=direct_delta.dtype) * self.direct_delta_scale_max
            r = r + direct_delta * direct_scale

        # 质心归一化
        centroid = scatter_mean(r, batch_index, dim=0)
        r_hat = r - centroid[batch_index]

        return r_hat
