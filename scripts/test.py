#!/usr/bin/env python
# /home/zdy/Project2/scripts/test.py

import yaml
import logging
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.data import Batch
from tqdm import tqdm

# --- Project-specific imports ---
from models.model import DynaModel
from scripts.train import load_config, setup_logging  # Reuse helpers from train.py
from scripts.data import get_data_loader


def calculate_rmsd(pred_coords, true_coords):
    """Calculates the Root Mean Square Deviation."""
    if pred_coords.numel() == 0 or true_coords.numel() == 0:
        return torch.tensor(0.0)
    return torch.sqrt(torch.mean((pred_coords.float() - true_coords.float()) ** 2))


@torch.no_grad()
def evaluate_on_test_set(model: nn.Module, loader: PyGDataLoader, device: torch.device):
    """
    Evaluates the model on a given test set and computes all metrics.
    """
    model.eval()

    per_sample_metrics = []
    all_pred_affinities, all_true_affinities = [], []

    logging.info(f"Running evaluation on {len(loader.dataset)} samples...")
    for batch in tqdm(loader, desc="Evaluating Test Set", leave=False):
        batch = batch.to(device)

        # --- Model Inference ---
        output = model(batch, use_diffusion_refinement=True)

        if output.get('pred_affinity') is not None:
            all_pred_affinities.append(output['pred_affinity'].cpu())
            all_true_affinities.append(output['true_affinity'].cpu())

        # --- Calculate per-sample structural metrics ---
        ptr = batch.ptr.cpu()
        group_ids_full = batch.atom_group_ids.cpu()
        pred_coords_full = output['pred_coords'].cpu()
        true_coords_full = output['true_coords'].cpu()

        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i + 1]

            sample_pred = pred_coords_full[start:end]
            sample_true = true_coords_full[start:end]
            sample_groups = group_ids_full[start:end]

            ligand_mask = (sample_groups == 2)
            sc_mask = (sample_groups == 1)

            sample_metrics = {
                'overall_rmsd': calculate_rmsd(sample_pred, sample_true).item(),
                'ligand_rmsd': calculate_rmsd(sample_pred[ligand_mask], sample_true[ligand_mask]).item(),
                'sc_rmsd': calculate_rmsd(sample_pred[sc_mask], sample_true[sc_mask]).item()
            }
            per_sample_metrics.append(sample_metrics)

    # --- Aggregate and Finalize Metrics ---
    metrics_df = pd.DataFrame(per_sample_metrics)

    overall_rmsd_avg = metrics_df['overall_rmsd'].mean()
    ligand_rmsd_avg = metrics_df['ligand_rmsd'].mean()
    sc_rmsd_avg = metrics_df['sc_rmsd'].mean()

    overall_success_rate = (metrics_df['overall_rmsd'] < 2.0).mean() * 100
    ligand_success_rate = (metrics_df['ligand_rmsd'] < 2.0).mean() * 100
    sc_success_rate = (metrics_df['sc_rmsd'] < 2.0).mean() * 100

    affinity_metrics = {}
    if all_pred_affinities:
        pred_affinities_all = torch.cat(all_pred_affinities, dim=0).numpy()
        true_affinities_all = torch.cat(all_true_affinities, dim=0).numpy()

        valid_indices = ~np.isnan(true_affinities_all)
        pred_affinities_valid = pred_affinities_all[valid_indices]
        true_affinities_valid = true_affinities_all[valid_indices]

        if len(pred_affinities_valid) > 1:
            affinity_metrics['PCC'] = pearsonr(pred_affinities_valid, true_affinities_valid)[0]
            affinity_metrics['Spearman'] = spearmanr(pred_affinities_valid, true_affinities_valid)[0]
            affinity_metrics['RMSE'] = np.sqrt(mean_squared_error(true_affinities_valid, pred_affinities_valid))
            affinity_metrics['MAE'] = mean_absolute_error(true_affinities_valid, pred_affinities_valid)

    results = {
        "Structural Metrics": {
            "Overall RMSD": overall_rmsd_avg,
            "Ligand RMSD (L-RMSD)": ligand_rmsd_avg,
            "Side-chain RMSD (sc-RMSD)": sc_rmsd_avg,
        },
        "Success Rates (< 2A)": {
            "Overall Success Rate (%)": overall_success_rate,
            "Ligand Success Rate (%)": ligand_success_rate,
            "Side-chain Success Rate (%)": sc_success_rate,
        },
        "Functional (Affinity) Metrics": affinity_metrics
    }

    return results


def main():
    """Main testing function."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / 'config.yaml'
    config = load_config(config_path)

    project_root = config_path.parent
    test_cfg = config['testing']

    config['training']['batch_size'] = test_cfg.get('batch_size', 16)

    log_cfg = config['logging']['training_log']
    setup_logging(log_cfg, project_root, log_base_name="evaluation")

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Starting Evaluation ---")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    checkpoint_path = project_root / test_cfg['checkpoint_path']
    if not checkpoint_path.is_file():
        logging.critical(f"Checkpoint file not found at: {checkpoint_path}")
        return

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = DynaModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # --- Identify and Run on Test Sets ---
    test_sets_to_run = test_cfg.get('test_on_sets', [])
    all_configured_sets = [ts['name'] for ts in config['data_loading'].get('test_sets', [])]

    if not test_sets_to_run:
        test_sets_to_run = all_configured_sets
        logging.info("No specific test sets provided. Running on all configured test sets.")

    for test_set_name in test_sets_to_run:
        logging.info(f"\n{'=' * 20} Evaluating on: {test_set_name} {'=' * 20}")

        test_loader = get_data_loader(config, test_set_name)
        if not test_loader:
            logging.error(f"Failed to create data loader for {test_set_name}. Skipping.")
            continue

        results = evaluate_on_test_set(model, test_loader, device)

        logging.info(f"--- Results for {test_set_name} ---")

        logging.info("[Structural Metrics]")
        for metric, value in results["Structural Metrics"].items():
            logging.info(f"  {metric:<30}: {value:.4f}")

        logging.info("[Success Rates (< 2Å)]")
        for metric, value in results["Success Rates (< 2A)"].items():
            logging.info(f"  {metric:<30}: {value:.2f}%")

        if results["Functional (Affinity) Metrics"]:
            logging.info("[Functional (Affinity) Metrics]")
            for metric, value in results["Functional (Affinity) Metrics"].items():
                logging.info(f"  {metric:<30}: {value:.4f}")
        else:
            logging.info("[Functional (Affinity) Metrics]")
            logging.info("  (Not available for this dataset or model did not predict affinity)")

    logging.info("\nEvaluation finished.")


if __name__ == '__main__':
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    main()