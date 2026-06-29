#!/usr/bin/env python
# /home/zdy/Project2/scripts/train_encoders.py

import yaml
import logging
import datetime
import random
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

# --- Project-specific imports ---
from models.encoder import CooperativeSE3Encoder, MLP
from scripts.data import get_data_loader


# ==============================================================================
#  Helper Functions
# ==============================================================================

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_cfg: dict, project_root: Path, log_base_name: str = "train"):
    """Configures logging for the run."""
    log_dir = project_root / log_cfg['log_dir']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = log_dir / f"{log_base_name}_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    log_filepath = run_log_dir / "run.log"
    log_level = getattr(logging, log_cfg.get('log_level', 'INFO').upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
        force=True
    )
    logging.info(f"Logging configured. Logs will be saved to: {run_log_dir}")
    return run_log_dir


# ==============================================================================
#  Stage 1 Model: Encoders for Affinity Prediction
# ==============================================================================

class AffinityPredictionModel(nn.Module):
    """
    A model for pre-training encoders by predicting binding affinity.
    It consists of three encoders and a final MLP head.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config['model_params']

        # Shared encoder parameters
        encoder_args = {
            'hidden_scalar_dim': model_cfg['hidden_scalar_dim'],
            'hidden_vector_dim': model_cfg['hidden_vector_dim'],
            'num_layers': model_cfg['num_encoder_layers'],
            'l_max_sh': model_cfg['l_max_sh'],
            'mlp_hidden_dims': model_cfg['mlp_hidden_dims']
        }

        self.backbone_encoder = CooperativeSE3Encoder(
            in_scalar_dim=model_cfg['in_scalar_dim_bb'],
            in_edge_scalar_dim=model_cfg['in_edge_scalar_dim'],
            **encoder_args
        )
        self.sidechain_encoder = CooperativeSE3Encoder(
            in_scalar_dim=model_cfg['in_scalar_dim_sc'],
            in_edge_scalar_dim=model_cfg['in_edge_scalar_dim'],
            **encoder_args
        )
        self.drug_encoder = CooperativeSE3Encoder(
            in_scalar_dim=model_cfg['in_scalar_dim_drug'],
            in_edge_scalar_dim=model_cfg['in_edge_scalar_dim'],
            **encoder_args
        )

        # MLP head for predicting affinity from fused embeddings
        pooled_dim = model_cfg['hidden_scalar_dim']
        affinity_head_in_dim = pooled_dim * 3  # eb + es + ed
        self.affinity_head = MLP(
            affinity_head_in_dim,
            model_cfg['affinity_mlp_hidden_dims'],
            1
        )

    def forward(self, batch: Batch):
        # Encode each component
        h_b, _, _, _ = self.backbone_encoder(batch['backbone'])
        h_d, _, _, _ = self.drug_encoder(batch['drug'])

        # Handle cases where a batch might not have any sidechains
        if 'sidechain' in batch.node_types and batch['sidechain'].num_nodes > 0:
            h_s, _, _, _ = self.sidechain_encoder(batch['sidechain'])
            es = global_mean_pool(h_s, batch['sidechain'].batch)
        else:
            # If no sidechains, create a zero tensor as a placeholder
            es = torch.zeros(batch.num_graphs, self.config['model_params']['hidden_scalar_dim'], device=h_b.device)

        # Pool node features to get graph-level embeddings
        eb = global_mean_pool(h_b, batch['backbone'].batch)
        ed = global_mean_pool(h_d, batch['drug'].batch)

        # Fuse embeddings and predict affinity
        fused_embedding = torch.cat([eb, es, ed], dim=-1)
        pred_affinity = self.affinity_head(fused_embedding).squeeze(-1)

        return pred_affinity


def evaluate_encoders(model: nn.Module, loader: PyGDataLoader, loss_fn: nn.Module, device: torch.device):
    """Evaluates the encoder model on a given dataset."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(device)
            true_affinity = batch.affinity.to(device).float()

            # Filter out samples with NaN affinity
            nan_mask = ~torch.isnan(true_affinity)
            if not nan_mask.any():
                continue

            batch = batch.index_select(nan_mask)
            true_affinity = true_affinity[nan_mask]

            pred_affinity = model(batch)
            loss = loss_fn(pred_affinity, true_affinity)
            total_loss += loss.item()

    return total_loss / len(loader)


# ==============================================================================
#  Main Training Script
# ==============================================================================

def main():
    """Main training script for the encoder pre-training stage."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / 'config.yaml'
    config = load_config(config_path)

    project_root = config_path.parent
    train_cfg = config['training_encoders']

    # --- Initialization ---
    run_dir = setup_logging(config['logging']['training_encoders_log'], project_root, "training_encoders")
    set_seed(train_cfg['seed'])
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Starting Stage 1: Encoder Training for Affinity Prediction ---")
    logging.info(f"Using device: {device}")

    # Save a copy of the config for this run
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # --- Data Loading ---
    logging.info("Loading data...")
    train_loader = get_data_loader(config, 'train', train_cfg)
    val_loader = get_data_loader(config, 'val', train_cfg)
    if not train_loader or not val_loader:
        logging.critical("Failed to create data loaders. Exiting.")
        return
    logging.info(f"Train loader: {len(train_loader.dataset)} samples. Val loader: {len(val_loader.dataset)} samples.")

    # --- Model, Loss, Optimizer ---
    logging.info("Initializing model, loss function, and optimizer...")
    model = AffinityPredictionModel(config).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])

    # --- Training Loop ---
    best_val_loss = float('inf')
    logging.info("Starting training loop...")
    for epoch in range(train_cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")

        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)
            true_affinity = batch.affinity.to(device).float()

            # Filter out samples with NaN affinity for this batch
            nan_mask = ~torch.isnan(true_affinity)
            if not nan_mask.any():
                continue  # Skip batch if all affinities are NaN

            # Select only the valid samples for this batch
            batch = batch.index_select(nan_mask)
            true_affinity = true_affinity[nan_mask]

            optimizer.zero_grad()
            pred_affinity = model(batch)
            loss = loss_fn(pred_affinity, true_affinity)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % train_cfg['log_every_n_steps'] == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        if (epoch + 1) % train_cfg['validate_every_n_epochs'] == 0:
            val_loss = evaluate_encoders(model, val_loader, loss_fn, device)
            logging.info(f"Epoch {epoch + 1} | Validation Loss: {val_loss:.4f}")

            # --- Save Best Model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save each encoder's state_dict separately
                torch.save(model.backbone_encoder.state_dict(), checkpoint_dir / 'backbone_encoder_best.pt')
                torch.save(model.sidechain_encoder.state_dict(), checkpoint_dir / 'sidechain_encoder_best.pt')
                torch.save(model.drug_encoder.state_dict(), checkpoint_dir / 'drug_encoder_best.pt')

                logging.info(f"New best encoders saved to {checkpoint_dir}")

    logging.info("Encoder training finished.")


if __name__ == '__main__':
    # Add project root to sys.path to allow for imports like `from models.encoder import ...`
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    main()