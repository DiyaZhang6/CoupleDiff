# /home/zdy/Project2/models/diffusion.py

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron used as the backbone for the denoiser.
    """

    def __init__(self, in_dim, hidden_dims, out_dim, activation="ReLU", use_batch_norm=False):
        super().__init__()
        activation_map = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}
        act_fn = activation_map.get(activation, nn.ReLU)
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard sinusoidal embedding to encode the diffusion timestep t.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionRefiner(nn.Module):
    """
    Conditional Diffusion Refinement in Latent Interaction Space.
    Following DDPM principles to refine embeddings conditioned on interaction context e_f.
    """

    def __init__(self, config: dict):
        super().__init__()
        # Load parameters from config.yaml -> diffusion_params
        params = config['diffusion_params']
        self.embed_dim = params['embed_dim']
        self.num_timesteps = params['num_timesteps']

        schedule_type = params.get('schedule_type', 'linear')
        beta_start = params.get('beta_start', 0.0001)
        beta_end = params.get('beta_end', 0.02)

        denoiser_config = params['denoiser_mlp']
        self.time_embed_dim = denoiser_config['time_embed_dim']

        # --- Set up Gaussian Diffusion Schedule ---
        if schedule_type == 'linear':
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        else:
            raise ValueError(f"Schedule {schedule_type} is not supported.")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register schedule constants as buffers (fixed during training)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # Calculations for posterior q(z_{t-1} | z_t, z_0)
        self.register_buffer('posterior_variance',
                             betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

        # --- Denoiser Network (p_theta) ---
        self.time_embedding = SinusoidalTimestepEmbedding(self.time_embed_dim)

        # Input: current latent z_t (dim) + fused condition e_f (dim) + time_emb (dim)
        # Total input dimension = 128 + 128 + 128 = 384
        denoiser_input_dim = self.embed_dim + self.embed_dim + self.time_embed_dim

        self.denoiser_mlp = MLP(
            in_dim=denoiser_input_dim,
            hidden_dims=denoiser_config['hidden_dims'],  # [512, 512]
            out_dim=self.embed_dim  # Predicts noise epsilon
        )

    def q_sample(self, z_0, t, noise):
        if t.size(0) != z_0.size(0):
            t = t[:z_0.size(0)]

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, z_t, t_per_node, e_f):
        """
        Single reverse denoising step .
        Explicitly conditioned on the fused embedding e_f.
        """
        time_emb = self.time_embedding(t_per_node)
        # Explicit conditioning: concat(z_t, e_f, time_emb)
        net_input = torch.cat([z_t, e_f, time_emb], dim=-1)
        epsilon_theta = self.denoiser_mlp(net_input)

        alpha_t = self.alphas[t_per_node].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t_per_node].view(-1, 1)
        beta_t = self.betas[t_per_node].view(-1, 1)

        # Mean calculation
        # z_{t-1} = 1/sqrt(alpha_t) * (z_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_theta)
        model_mean = (1.0 / torch.sqrt(alpha_t)) * (
                z_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * epsilon_theta
        )

        if t_per_node[0] == 0:
            return model_mean
        else:
            # Add noise for Langevin dynamics (except at the final step)
            variance = self.posterior_variance[t_per_node].view(-1, 1)
            noise = torch.randn_like(z_t)
            return model_mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def forward(self, e_f, batch_ptr):
        """
        Full inference process.
        Starts from Gaussian noise z_T and iteratively denoises towards z_0.
        """
        device = e_f.device
        # Sample z_T ~ N(0, I)
        z_t = torch.randn_like(e_f)

        # From t = T down to 1
        for t in reversed(range(self.num_timesteps)):
            num_graphs = len(batch_ptr) - 1
            t_tensor = torch.full((num_graphs,), t, device=device, dtype=torch.long)
            # Broadcast graph-level timestep to all nodes in the batch
            t_per_node = t_tensor.repeat_interleave(torch.diff(batch_ptr))

            # Iterative reconstruction conditioned on e_f
            z_t = self.p_sample(z_t, t_per_node, e_f)

        # Result is refined embedding e_refined
        return z_t

    def get_training_loss(self, e_gt, e_f, batch_ptr):
        """
        Training objective: Loss_noise = E || epsilon - epsilon_theta ||^2.
        Trains the denoiser to predict the added noise at a random timestep.
        """
        num_graphs = len(batch_ptr) - 1
        device = e_gt.device

        # Randomly sample timesteps for each graph in the batch
        t_graph = torch.randint(0, self.num_timesteps, (num_graphs,), device=device).long()
        t_node = t_graph.repeat_interleave(torch.diff(batch_ptr))

        # Sample ground-truth noise epsilon
        epsilon = torch.randn_like(e_gt)

        # Generate corrupted latent z_t at sampled timestep
        z_t = self.q_sample(z_0=e_gt, t=t_node, noise=epsilon)

        # Predict noise using the network conditioned on e_f
        time_emb = self.time_embedding(t_node)
        net_input = torch.cat([z_t, e_f, time_emb], dim=-1)
        predicted_noise = self.denoiser_mlp(net_input)

        # L_noise: Mean Squared Error between predicted and actual noise
        return F.mse_loss(predicted_noise, epsilon)