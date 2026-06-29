#!/usr/bin/env python
# backbone_graph.py
# --- Creates residue-level graph for the protein backbone ---
# Reads backbone PDB files and saves graph data, including residue identifiers for coordinate mapping.

import os
import argparse
import logging
import logging.handlers
import yaml
import datetime
import math
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from typing import Optional, Dict
from collections import defaultdict

# --- Third-party imports ---
try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.vectors import calc_dihedral, Vector, rotaxis
except ImportError as e:
    print(f"FATAL: Missing required libraries. Please install BioPython. Error: {e}")
    exit(1)

# --- Global Variables ---
CONFIG: Optional[dict] = None
PROJECT_ROOT: Optional[Path] = None


# --- Framework Functions ---
def setup_logging(log_config_key: str):
    global CONFIG, PROJECT_ROOT
    if not CONFIG or not PROJECT_ROOT: print("FATAL: Config not loaded."); return
    log_config = CONFIG.get('logging', {}).get(log_config_key, {})
    if not log_config: print(f"FATAL: Log config for '{log_config_key}' not found."); return
    log_level = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    use_timestamp = log_config.get('use_timestamp_in_log_name', True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{log_base_name}_{timestamp}.log" if use_timestamp else f"{log_base_name}.log"
    log_path_abs = PROJECT_ROOT / log_dir_rel / log_file_name
    log_path_abs.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.getLevelName(log_level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path_abs, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info(f"Logging configured. Level: {log_level}. Output file: {log_path_abs}")


def load_config(config_path: Path):
    global CONFIG, PROJECT_ROOT
    if not config_path.is_file(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        try:
            CONFIG = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            CONFIG = yaml.safe_load(f)
    PROJECT_ROOT = config_path.parent.resolve()


# --- Feature Engineering Functions ---
def one_hot_encode(value, allowed_list):
    encoding = [0.0] * len(allowed_list)
    try:
        idx = allowed_list.index(value)
        encoding[idx] = 1.0
    except ValueError:
        if 'UNK' in allowed_list:
            encoding[allowed_list.index('UNK')] = 1.0
    return encoding


def normalize_vector(v: np.ndarray, default_zero=True) -> np.ndarray:
    norm = np.linalg.norm(v)
    return np.zeros_like(v) if norm < 1e-6 and default_zero else v / norm


def get_residue_id_tuple(residue) -> tuple:
    """Returns a tuple representation of the residue ID for mapping."""
    res_id = residue.get_id()
    chain_id = residue.get_parent().id
    return (chain_id, res_id[1], res_id[2].strip())


def get_node_features(residue, prev_res, next_res, residue_idx, config):
    res_name = residue.get_resname()
    if res_name not in config['allowed_amino_acids']: res_name = 'UNK'
    aa_one_hot = one_hot_encode(res_name, config['allowed_amino_acids'])
    seq_id_norm = float(residue_idx) / config['seq_len_norm_factor']
    hydrophobicity = config['hydrophobicity_scale'].get(res_name, 0.0)
    node_scalar = np.concatenate([aa_one_hot, [seq_id_norm, hydrophobicity]])
    try:
        ca_coord, n_coord, c_coord = residue['CA'].get_coord(), residue['N'].get_coord(), residue['C'].get_coord()
    except KeyError:
        return None, None
    phi, psi = np.nan, np.nan
    if prev_res:
        try:
            phi = calc_dihedral(prev_res['C'].get_vector(), residue['N'].get_vector(), residue['CA'].get_vector(),
                                residue['C'].get_vector())
        except KeyError:
            pass
    if next_res:
        try:
            psi = calc_dihedral(residue['N'].get_vector(), residue['CA'].get_vector(), residue['C'].get_vector(),
                                next_res['N'].get_vector())
        except KeyError:
            pass
    dihedrals = np.array([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)])
    dihedrals[np.isnan(dihedrals)] = 0.0
    orientation = np.zeros((4, 3))
    orientation[0] = normalize_vector(n_coord - ca_coord)
    orientation[1] = normalize_vector(c_coord - ca_coord)
    if 'O' in residue: orientation[2] = normalize_vector(residue['O'].get_coord() - ca_coord)
    if res_name == 'GLY':
        rot = rotaxis(math.radians(-120.0), Vector(c_coord) - Vector(ca_coord))
        orientation[3] = normalize_vector((Vector(n_coord) - Vector(ca_coord)).left_multiply(rot).get_array())
    elif 'CB' in residue:
        orientation[3] = normalize_vector(residue['CB'].get_coord() - ca_coord)
    node_vector = {'ca_coord': ca_coord, 'dihedrals': dihedrals, 'orientation': orientation}
    return node_scalar.astype(np.float32), node_vector


def get_edge_features(res1, res2):
    try:
        ca1, ca2 = res1['CA'].get_coord(), res2['CA'].get_coord()
    except KeyError:
        return None, None
    distance = np.linalg.norm(ca1 - ca2)
    is_peptide = 1.0 if res1.get_parent() == res2.get_parent() and abs(res1.id[1] - res2.id[1]) == 1 else 0.0
    edge_scalar = np.array([distance, is_peptide]).astype(np.float32)
    edge_vector = {'direction': normalize_vector(ca2 - ca1).astype(np.float32)}
    return edge_scalar, edge_vector


# --- Main Graph Creation Function ---
def create_and_save_backbone_graph(pdb_id: str, config: dict):
    """Processes a single PDB, creates the graph, and saves it with residue identifiers."""
    task_config = config['backbone_graph_task']
    input_dir = PROJECT_ROOT / task_config['input_dir']
    output_dir = PROJECT_ROOT / task_config['output_dir']
    backbone_file = input_dir / pdb_id / f"{pdb_id}{task_config['backbone_suffix']}"
    output_file = output_dir / pdb_id / f"{pdb_id}_backbone.pt"

    if not task_config.get('overwrite_existing', False) and output_file.exists():
        logging.debug(f"Skipping {pdb_id}: Output file already exists.")
        return 'skipped'

    if not backbone_file.exists():
        logging.warning(f"[{pdb_id}] Backbone file not found: {backbone_file}. Skipping.")
        return 'no_input'

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(backbone_file))
        residues = sorted([r for r in structure.get_residues() if is_aa(r)], key=lambda r: (r.get_parent().id, r.id[1]))
    except Exception as e:
        logging.error(f"[{pdb_id}] Could not parse or extract residues from {backbone_file.name}: {e}")
        return 'parse_error'

    if len(residues) < 2:
        logging.info(f"[{pdb_id}] Not enough standard residues (<2) in {backbone_file.name} to build a graph.")
        return 'too_few_residues'

    node_s, node_v_ca, node_v_dih, node_v_ori = [], [], [], []

    # Create lists to store valid residues and their identifiers
    valid_residues = []
    residue_ids_for_map = []

    for i, res in enumerate(residues):
        s_feats, v_feats = get_node_features(res, residues[i - 1] if i > 0 else None,
                                             residues[i + 1] if i < len(residues) - 1 else None, i, task_config)
        if s_feats is None:
            continue

        node_s.append(s_feats)
        node_v_ca.append(v_feats['ca_coord'])
        node_v_dih.append(v_feats['dihedrals'])
        node_v_ori.append(v_feats['orientation'])

        # If the node features were successfully generated, add the residue and its ID
        valid_residues.append(res)
        residue_ids_for_map.append(get_residue_id_tuple(res))

    if not node_s:
        logging.error(f"[{pdb_id}] Failed to generate any valid nodes.")
        return 'no_valid_nodes'

    # --- Edge creation using only the valid residues ---
    edge_s, edge_v_dir, edge_idx = [], [], []
    valid_res_to_idx_map = {res.get_full_id(): i for i, res in enumerate(valid_residues)}

    for i in range(len(valid_residues) - 1):
        res1, res2 = valid_residues[i], valid_residues[i + 1]
        if res1.get_parent() == res2.get_parent() and res2.id[1] - res1.id[1] == 1:
            e_s_feats, e_v_feats = get_edge_features(res1, res2)
            if e_s_feats is not None:
                idx1 = valid_res_to_idx_map[res1.get_full_id()]
                idx2 = valid_res_to_idx_map[res2.get_full_id()]
                edge_idx.extend([[idx1, idx2], [idx2, idx1]])
                edge_s.extend([e_s_feats, e_s_feats])
                edge_v_dir.extend([e_v_feats['direction'], -e_v_feats['direction']])

    graph_data = {
        'node_s': torch.from_numpy(np.array(node_s, dtype=np.float32)),
        'node_v': {
            'ca_coord': torch.from_numpy(np.array(node_v_ca, dtype=np.float32)),
            'dihedrals': torch.from_numpy(np.array(node_v_dih, dtype=np.float32)),
            'orientation': torch.from_numpy(np.array(node_v_ori, dtype=np.float32))
        },
        'edge_index': torch.tensor(np.array(edge_idx).T, dtype=torch.long) if edge_idx else torch.empty((2, 0),
                                                                                                        dtype=torch.long),
        'edge_s': torch.from_numpy(np.array(edge_s, dtype=np.float32)) if edge_s else torch.empty((0, 2),
                                                                                                  dtype=torch.float32),
        'edge_v': {
            'direction': torch.from_numpy(np.array(edge_v_dir, dtype=np.float32)) if edge_v_dir else torch.empty((0, 3),
                                                                                                                 dtype=torch.float32)
        },

        # Add the list of residue identifiers to the dictionary
        'residue_ids': residue_ids_for_map,
        'pdb_id': pdb_id
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, output_file)
    logging.info(f"[{pdb_id}] Successfully created and saved backbone graph ({len(node_s)} nodes).")
    return 'success'


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Generate residue-level graphs for protein backbones.")
    script_dir = Path(__file__).parent
    default_config_path = script_dir.parent / 'config.yaml'
    parser.add_argument('--config', type=str, default=str(default_config_path),
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        load_config(config_path)
        setup_logging('backbone_graph_log')
    except Exception as e:
        print(f"FATAL: Failed to initialize script: {e}")
        exit(1)

    logging.info("--- Starting Backbone Graph Generation ---")
    task_config = CONFIG.get('backbone_graph_task', {})
    if not task_config:
        logging.critical("'backbone_graph_task' not found in config.yaml. Aborting.")
        return

    input_dir = PROJECT_ROOT / task_config['input_dir']
    pdb_id_folders = sorted([d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if not pdb_id_folders:
        logging.warning(f"No PDB ID subdirectories found in {input_dir}.")
        return

    logging.info(f"Found {len(pdb_id_folders)} PDB ID folders to process in {input_dir}.")

    results = defaultdict(int)
    failed_ids = defaultdict(list)

    for folder in tqdm(pdb_id_folders, desc="Processing PDBs"):
        pdb_id = folder.name
        status = create_and_save_backbone_graph(pdb_id, CONFIG)
        results[status] += 1
        if status not in ['success', 'skipped']:
            failed_ids[status].append(pdb_id)

    logging.info("--- Backbone Graph Generation Complete ---")
    total_processed = len(pdb_id_folders)
    logging.info(f"Total PDBs considered: {total_processed}")

    for status, count in sorted(results.items()):
        logging.info(f"  - {status.replace('_', ' ').capitalize():<20}: {count}")

    total_failures = sum(len(v) for k, v in failed_ids.items())
    if total_failures > 0:
        logging.info("\n--- Failure Report ---")
        logging.warning(f"A total of {total_failures} PDBs failed to process.")
        for status, ids in sorted(failed_ids.items()):
            logging.warning(f"\nReason: {status.replace('_', ' ').capitalize()} ({len(ids)} PDBs):")
            chunk_size = 10
            id_chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
            for chunk in id_chunks:
                logging.warning(f"  IDs: {', '.join(chunk)}")
    else:
        logging.info("All PDBs were processed successfully or skipped as intended.")


if __name__ == "__main__":
    main()