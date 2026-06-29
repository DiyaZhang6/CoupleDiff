# /home/zdy/Project2/models/model.py

import logging
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from .encoder import CooperativeSE3Encoder
from .interaction import InteractionModule, MLP
from .diffusion import DiffusionRefiner
from .decoder import StructureDecoder


class DynaModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config['model_params']
        self.debug_forward_stages = bool(config.get('training', {}).get('debug_model_forward_stages', False))

        # Encoders
        self.backbone_encoder = CooperativeSE3Encoder(
            in_scalar_dim=model_cfg['in_scalar_dim_bb'],
            in_edge_scalar_dim=model_cfg['in_edge_scalar_dim'],
            hidden_scalar_dim=model_cfg['hidden_scalar_dim'],
            hidden_vector_dim=model_cfg['hidden_vector_dim'],
            num_layers=model_cfg['num_encoder_layers'],
            l_max_sh=model_cfg['l_max_sh'],
            mlp_hidden_dims=model_cfg['mlp_hidden_dims']
        )
        # Sidechain and Drug use same class but different input dims
        self.sidechain_encoder = CooperativeSE3Encoder(
            in_scalar_dim=model_cfg['in_scalar_dim_sc'],
            in_edge_scalar_dim=model_cfg['in_edge_scalar_dim'],
            hidden_scalar_dim=model_cfg['hidden_scalar_dim'],
            hidden_vector_dim=model_cfg['hidden_vector_dim'],
            num_layers=model_cfg['num_encoder_layers'],
            l_max_sh=model_cfg['l_max_sh'],
            mlp_hidden_dims=model_cfg['mlp_hidden_dims']
        )
        self.drug_encoder = CooperativeSE3Encoder(
            in_scalar_dim=model_cfg['in_scalar_dim_drug'],
            in_edge_scalar_dim=model_cfg['in_edge_scalar_dim'],
            hidden_scalar_dim=model_cfg['hidden_scalar_dim'],
            hidden_vector_dim=model_cfg['hidden_vector_dim'],
            num_layers=model_cfg['num_encoder_layers'],
            l_max_sh=model_cfg['l_max_sh'],
            mlp_hidden_dims=model_cfg['mlp_hidden_dims']
        )

        self.interaction_module = InteractionModule(config['interaction_params'])
        self.diffusion_refiner = DiffusionRefiner(config)
        self.decoder = StructureDecoder(config)
        diffusion_cfg = config.get('diffusion_params', {})
        self.detach_diffusion_z0_for_noise = bool(diffusion_cfg.get('detach_z0_for_noise_loss', False))
        self.detach_diffusion_condition_for_noise = bool(diffusion_cfg.get('detach_condition_for_noise_loss', False))
        self.detach_pred_noise_for_structure = bool(diffusion_cfg.get('detach_pred_noise_for_structure', False))
        self.normalize_diffusion_inputs = bool(diffusion_cfg.get('normalize_inputs', False))
        self.diffusion_input_clip = diffusion_cfg.get('input_clip', None)

        if model_cfg.get('predict_affinity', False):
            self.affinity_head = MLP(model_cfg['hidden_scalar_dim'], model_cfg['affinity_mlp_hidden_dims'], 1)

    def _log_forward_stage(self, stage, device, **kwargs):
        if not self.debug_forward_stages:
            return
        logger = logging.getLogger("training")
        extras = " ".join(f"{key}={value}" for key, value in kwargs.items())
        if torch.cuda.is_available() and device.type == "cuda":
            logger.info(
                "Model forward stage=%s cuda_alloc=%.2fGiB cuda_reserved=%.2fGiB %s",
                stage,
                torch.cuda.memory_allocated(device) / (1024 ** 3),
                torch.cuda.memory_reserved(device) / (1024 ** 3),
                extras,
            )
        else:
            logger.info("Model forward stage=%s %s", stage, extras)

    def _prepare_diffusion_input(self, x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize_diffusion_inputs:
            x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        if self.diffusion_input_clip is not None:
            clip_value = float(self.diffusion_input_clip)
            if clip_value > 0:
                x = x.clamp(min=-clip_value, max=clip_value)
        return x

    def forward(self, batch, use_diffusion_refinement=False):

        def safe(x):
            if isinstance(x, dict):
                return next(iter(x.values()))
            return x

        device = batch['backbone'].x.device
        self._log_forward_stage("start", device)

        #print("===== DEBUG FEATURE DIM =====")
        #print("backbone:", batch['backbone'].x.shape)
        #print("Expected:", self.config['model_params']['in_scalar_dim_bb'])
        #print("============================")

        # --- 1. Backbone ---
        bb = batch['backbone']
        bb_edge_index = bb.edge_index if hasattr(bb, 'edge_index') else batch['backbone', 'backbone'].edge_index
        self._log_forward_stage(
            "before_backbone_encoder",
            device,
            nodes=int(bb.x.size(0)),
            edges=int(bb_edge_index.size(1)) if bb_edge_index is not None else 0,
        )
        h_b, v_b, e_b, _ = self.backbone_encoder(
            s=bb.x,
            pos=bb.pos,
            edge_index=bb_edge_index,
            edge_s=None
        )
        self._log_forward_stage("after_backbone_encoder", device)

        # --- 2. Sidechain ---
        if 'sidechain' in batch.node_types and batch['sidechain'].x.numel() > 0:
            sc = batch['sidechain']
            if hasattr(sc, 'edge_index'):
                sc_edge_index = sc.edge_index
            elif ('sidechain', 'sidechain') in batch.edge_types:
                sc_edge_index = batch['sidechain', 'sidechain'].edge_index
            else:
                sc_edge_index = torch.empty((2, 0), device=device, dtype=torch.long)

            self._log_forward_stage(
                "before_sidechain_encoder",
                device,
                nodes=int(sc.x.size(0)),
                edges=int(sc_edge_index.size(1)) if sc_edge_index is not None else 0,
            )
            h_s, _, _, _ = self.sidechain_encoder(
                s=sc.x,
                pos=sc.pos,
                edge_index=sc_edge_index,
                edge_s=None  # 显式设为 None
            )
            self._log_forward_stage("after_sidechain_encoder", device)
        else:
            h_s = torch.empty(0, self.config['model_params']['hidden_scalar_dim'], device=device)

        # --- 3. Drug ---
        dr = batch['drug']
        dr_edge_index = dr.edge_index if hasattr(dr, 'edge_index') else batch['drug', 'drug'].edge_index
        self._log_forward_stage(
            "before_drug_encoder",
            device,
            nodes=int(dr.x.size(0)),
            edges=int(dr_edge_index.size(1)) if dr_edge_index is not None else 0,
        )
        h_d, _, _, _ = self.drug_encoder(
            s=dr.x,
            pos=dr.pos,
            edge_index=dr_edge_index,
            edge_s=None  # 显式设为 None
        )
        self._log_forward_stage("after_drug_encoder", device)
        # --- 1. 精准提取物理特征列表 ---
        # 注意：data.py 里定义的 key 是 'lj_bs_list' (注意没有下划线前缀)
        s_bs = getattr(batch, 'lj_bs_list', [])
        s_bd = getattr(batch, 'lj_bd_list', [])
        s_sd = getattr(batch, 'lj_sd_list', [])

        # --- 2. 强力防御：确保即便属性丢失也不会让程序崩溃 ---
        # 如果因为 DataLoader 传输导致这些列表变成了 None，强制初始化为空列表
        if s_bs is None: s_bs = []
        if s_bd is None: s_bd = []
        if s_sd is None: s_sd = []

        # --- 3. 准备 Batch 索引 ---
        batch_b = batch['backbone'].batch
        batch_s = batch['sidechain'].batch if 'sidechain' in batch.node_types else torch.zeros(0, dtype=torch.long,
                                                                                               device=batch_b.device)
        batch_d = batch['drug'].batch

        # --- 4. 传入 Interaction 模块 ---
        self._log_forward_stage(
            "before_interaction",
            device,
            backbone=int(h_b.size(0)),
            sidechain=int(h_s.size(0)),
            drug=int(h_d.size(0)),
        )
        h_f_b, h_f_s, h_f_d = self.interaction_module(
            h_b, h_s, h_d,
            s_bs, s_bd, s_sd,
            batch_b, batch_s, batch_d
        )
        self._log_forward_stage("after_interaction", device)
        # --- Concatenation ---
        e_f = torch.cat([h_f_b, h_f_s, h_f_d], dim=0)  # 总节点数 N

        # --- 重点修正：计算每个样本的节点总数 ---
        # 确保 num_nodes_per_graph 对应 batch.num_graphs 个元素的列表/Tensor
        # 这里原来的写法在 batch_size > 1 时可能会把所有样本的节点数加在一起变成一个标量

        # 获取各类型节点在每个 sample 中的数量
        nb = batch['backbone'].ptr[1:] - batch['backbone'].ptr[:-1]
        nd = batch['drug'].ptr[1:] - batch['drug'].ptr[:-1]
        ns = (batch['sidechain'].ptr[1:] - batch['sidechain'].ptr[
                                           :-1]) if 'sidechain' in batch.node_types else torch.zeros_like(nb)

        # 每个样本的总节点数 (这是一个长度为 BatchSize 的 Tensor)
        nodes_per_sample = nb + ns + nd

        # --- 3. 执行拼接和累加 (global_batch_ptr) ---
        global_batch_ptr = torch.cat([torch.tensor([0], device=device), nodes_per_sample.cumsum(0)])

        if use_diffusion_refinement:
            self._log_forward_stage("before_diffusion_refiner", device)
            z_0_hat = self.diffusion_refiner(e_f, global_batch_ptr)
            pred_noise, true_noise = None, None
            self._log_forward_stage("after_diffusion_refiner", device)
        else:
            # --- 4. 修正时间步采样逻辑 ---
            # 为每个样本采一个 t
            t = torch.randint(0, self.config['diffusion_params']['num_timesteps'], (batch.num_graphs,),
                              device=device).long()

            # 【关键修复】确保 repeat 后的长度严格等于 e_f 的第一维
            t_per_node = torch.repeat_interleave(t, nodes_per_sample)

            # 这里的 true_noise 必须严格对应 e_f 的形状 [N, 128]
            true_noise = torch.randn_like(e_f)

            diffusion_z0 = e_f.detach() if self.detach_diffusion_z0_for_noise else e_f
            diffusion_condition = e_f.detach() if self.detach_diffusion_condition_for_noise else e_f
            diffusion_z0 = self._prepare_diffusion_input(diffusion_z0)
            diffusion_condition = self._prepare_diffusion_input(diffusion_condition)

            # 调用 q_sample
            z_t = self.diffusion_refiner.q_sample(diffusion_z0, t_per_node, true_noise)
            z_t = self._prepare_diffusion_input(z_t)

            # 后续 MLP
            self._log_forward_stage("before_noise_head", device, nodes=int(e_f.size(0)))
            time_emb = self.diffusion_refiner.time_embedding(t_per_node)
            pred_noise = self.diffusion_refiner.denoiser_mlp(torch.cat([z_t, diffusion_condition, time_emb], dim=-1))
            pred_noise = torch.nan_to_num(pred_noise, nan=0.0, posinf=0.0, neginf=0.0)
            self._log_forward_stage("after_noise_head", device)

            # 计算 z_0_hat 时也要确保 alpha 广播正确
            sqrt_alpha_bar = self.diffusion_refiner.sqrt_alphas_cumprod[t_per_node].view(-1, 1)
            sqrt_one_minus_alpha_bar = self.diffusion_refiner.sqrt_one_minus_alphas_cumprod[t_per_node].view(-1, 1)
            pred_noise_for_structure = pred_noise.detach() if self.detach_pred_noise_for_structure else pred_noise
            z_0_hat = (z_t - sqrt_one_minus_alpha_bar * pred_noise_for_structure) / (sqrt_alpha_bar + 1e-8)

            # --- Final Decoding ---

            # 1. 确保设备一致
            device = e_f.device

            # 2. 构造全图 batch 索引 (解决 full_batch_index 未定义)
            # 顺序必须与 e_f = torch.cat([h_f_b, h_f_s, h_f_d], dim=0) 严格对齐
            full_batch_index = torch.cat([
                batch['backbone'].batch,
                batch['sidechain'].batch if 'sidechain' in batch.node_types else torch.empty(0, dtype=torch.long,
                                                                                             device=device),
                batch['drug'].batch
            ], dim=0)

            # 3. 构造全图初始坐标 all_pos
            all_pos = torch.cat([
                batch['backbone'].pos,
                batch['sidechain'].pos if 'sidechain' in batch.node_types else torch.empty(0, 3, device=device),
                batch['drug'].pos
            ], dim=0)

            # 4. 安全获取 r_init (如果 data.py 没传过来，就用 all_pos 顶替)
            r_init = getattr(batch, 'r_init', None)
            if r_init is None:
                r_init = all_pos

            # 5. 获取/计算 global_edge_index
            global_edge_index = getattr(batch, 'global_edge_index', None)
            if global_edge_index is None:
                try:
                    from torch_cluster import radius_graph
                    # 从配置中获取截断半径
                    r_max = self.config.get('decoder_params', {}).get('rbf', {}).get('d_max', 20.0)
                    global_edge_index = radius_graph(
                        all_pos,
                        r=r_max,
                        batch=full_batch_index,
                        loop=False
                    )
                except ImportError:
                    # 最后的兜底：如果环境没装 torch_cluster，报错提醒
                    raise RuntimeError("Please install torch-cluster to compute radius graphs.")

            # 6. 正式调用 Decoder
            self._log_forward_stage(
                "before_decoder",
                device,
                nodes=int(z_0_hat.size(0)),
                edges=int(global_edge_index.size(1)) if global_edge_index is not None else 0,
            )
            pred_coords = self.decoder(
                z_0_hat,
                r_init,
                full_batch_index,
                global_edge_index
            )
            self._log_forward_stage("after_decoder", device)

            # 7. 统一返回结果
            return {
                "pred_coords": pred_coords,
                "true_coords": getattr(batch, 'r_true', None),
                "pred_noise": pred_noise,
                "true_noise": true_noise,
                "pred_affinity": self.affinity_head(scatter_mean(z_0_hat, full_batch_index, dim=0)).squeeze(
                    -1) if hasattr(self, 'affinity_head') else None,
                "true_affinity": getattr(batch, 'affinity', None)
            }
