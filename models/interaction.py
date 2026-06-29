# /home/zdy/Project2/models/interaction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation="ReLU", dropout=0.1):
        super().__init__()
        activation_map = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}
        act_fn = activation_map.get(activation, nn.ReLU)
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PhysicsConstrainedCrossAttention(nn.Module):
    """
    Performs attention constrained by a physics matrix (S).
    Operates on single-sample tensors to avoid OOM.
    """

    def __init__(self, embed_dim, head_dim, query_chunk_size=None):
        super().__init__()
        self.head_dim = head_dim
        self.query_chunk_size = int(query_chunk_size) if query_chunk_size else None
        self.w_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, head_dim, bias=False)
        self.gamma = nn.Parameter(torch.ones(1))
        self.scale = head_dim ** -0.5

    def forward(self, q_flat, k_flat, v_flat, s_matrix):
        # q_flat: [N_q, dim], k_flat: [N_k, dim], s_matrix: [N_q, N_k]
        q = self.w_q(q_flat)
        k = self.w_k(k_flat)
        v = self.w_v(v_flat)
        s_matrix = s_matrix.to(device=q.device, dtype=q.dtype)

        def attend(q_part, s_part):
            attn_scores = torch.matmul(q_part, k.transpose(-2, -1))  # [N_q, N_k]
            constrained_scores = (attn_scores * self.scale) + (self.gamma * s_part)
            weights = F.softmax(constrained_scores, dim=-1)
            return torch.matmul(weights, v)  # [N_q, head_dim]

        if self.query_chunk_size is None or self.query_chunk_size <= 0 or q.size(0) <= self.query_chunk_size:
            return attend(q, s_matrix)

        outputs = []
        for start in range(0, q.size(0), self.query_chunk_size):
            end = min(start + self.query_chunk_size, q.size(0))
            outputs.append(attend(q[start:end], s_matrix[start:end]))
        return torch.cat(outputs, dim=0)


class InteractionModule(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config['embed_dim']
        num_heads = config['num_heads']
        mlp_hidden = config.get('mlp_hidden_dims', [512])
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        query_chunk_size = config.get('attention_query_chunk_size', 512)

        # Define 6 attention paths
        self.paths = nn.ModuleDict({
            's_to_b': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim, query_chunk_size) for _ in range(num_heads)]),
            'd_to_b': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim, query_chunk_size) for _ in range(num_heads)]),
            'b_to_s': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim, query_chunk_size) for _ in range(num_heads)]),
            'd_to_s': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim, query_chunk_size) for _ in range(num_heads)]),
            'b_to_d': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim, query_chunk_size) for _ in range(num_heads)]),
            's_to_d': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim, query_chunk_size) for _ in range(num_heads)]),
        })

        fusion_in = embed_dim * 3
        self.mlp_b = MLP(fusion_in, mlp_hidden, embed_dim, activation=config.get('mlp_activation', 'ReLU'))
        self.mlp_s = MLP(fusion_in, mlp_hidden, embed_dim, activation=config.get('mlp_activation', 'ReLU'))
        self.mlp_d = MLP(fusion_in, mlp_hidden, embed_dim, activation=config.get('mlp_activation', 'ReLU'))

    def _run_direction(self, path_key, query, key, value, s_list, q_batch, k_batch, transpose_s=False):
        if query.numel() == 0 or key.numel() == 0:
            return query.new_zeros((query.size(0), self.embed_dim))

        unique_batch_ids = torch.unique(q_batch)
        heads = self.paths[path_key]
        out_final = query.new_zeros((query.size(0), self.embed_dim))

        for b_idx in unique_batch_ids:
            b_idx_item = b_idx.item()
            q_mask = (q_batch == b_idx_item)
            k_mask = (k_batch == b_idx_item)

            if s_list is None or b_idx_item >= len(s_list):
                continue

            s = s_list[b_idx_item]

            if not q_mask.any() or not k_mask.any() or s is None:
                continue

            expected_q = q_mask.sum()
            expected_k = k_mask.sum()

            curr_s = s.transpose(-1, -2) if transpose_s else s

            if curr_s.size(0) != expected_q or curr_s.size(1) != expected_k:
                continue

            head_outputs = []
            for h in heads:
                head_out = h(query[q_mask], key[k_mask], value[k_mask], curr_s)
                head_outputs.append(head_out)

            out_final[q_mask] = torch.cat(head_outputs, dim=-1).to(dtype=out_final.dtype)

        return out_final

    @staticmethod
    def _num_graphs(*batch_tensors):
        max_id = None
        for batch in batch_tensors:
            if batch is not None and batch.numel() > 0:
                curr = batch.max()
                max_id = curr if max_id is None else torch.maximum(max_id, curr)
        return int(max_id.item()) + 1 if max_id is not None else 0

    def _scatter_context(self, values, batch, dim_size):
        if values.numel() == 0 or batch.numel() == 0 or dim_size == 0:
            return values.new_zeros((dim_size, self.embed_dim))
        return scatter_mean(values, batch, dim=0, dim_size=dim_size)

    def forward(self, h_b, h_s, h_d, s_bs_list, s_bd_list, s_sd_list, batch_b, batch_s, batch_d):
        # 1. Compute all 6 cross-interactions
        # Backbone
        msg_s2b = self._run_direction('s_to_b', h_b, h_s, h_s, s_bs_list, batch_b, batch_s)
        msg_d2b = self._run_direction('d_to_b', h_b, h_d, h_d, s_bd_list, batch_b, batch_d)

        # Sidechain
        msg_b2s = self._run_direction('b_to_s', h_s, h_b, h_b, s_bs_list, batch_s, batch_b, transpose_s=True)
        msg_d2s = self._run_direction('d_to_s', h_s, h_d, h_d, s_sd_list, batch_s, batch_d)

        # Drug
        msg_b2d = self._run_direction('b_to_d', h_d, h_b, h_b, s_bd_list, batch_d, batch_b, transpose_s=True)
        msg_s2d = self._run_direction('s_to_d', h_d, h_s, h_s, s_sd_list, batch_d, batch_s, transpose_s=True)

        # 2. Per-sample Context Summary (Scatter Mean within each complex)
        # Context FROM B: Average of what B sent to S and D
        num_graphs = self._num_graphs(batch_b, batch_s, batch_d)

        context_b = self._scatter_context(msg_b2s, batch_s, num_graphs) + \
                    self._scatter_context(msg_b2d, batch_d, num_graphs)

        # Context FROM S: Average of what S sent to B and D
        context_s = self._scatter_context(msg_s2b, batch_b, num_graphs) + \
                    self._scatter_context(msg_s2d, batch_d, num_graphs)

        # Context FROM D: Average of what D sent to B and S
        context_d = self._scatter_context(msg_d2b, batch_b, num_graphs) + \
                    self._scatter_context(msg_d2s, batch_s, num_graphs)

        # 3. Final Fusion with residual connection
        # Backbone Update
        h_f_b = h_b + self.mlp_b(torch.cat([msg_s2b, msg_d2b, context_b[batch_b]], dim=-1))

        # Sidechain Update (if exists)
        h_f_s = h_s
        if h_s.size(0) > 0:
            h_f_s = h_s + self.mlp_s(torch.cat([msg_b2s, msg_d2s, context_s[batch_s]], dim=-1))

        # Drug Update
        h_f_d = h_d + self.mlp_d(torch.cat([msg_b2d, msg_s2d, context_d[batch_d]], dim=-1))

        return h_f_b, h_f_s, h_f_d
