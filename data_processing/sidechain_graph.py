#!/usr/bin/env python
# sidechain_graph.py
# --- Creates atom-level graphs for protein sidechains ---
# Relies on split.py output and uses RDKit for chemical feature extraction.

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
from typing import Optional, List, Tuple, Dict
from collections import defaultdict

# --- Third-party imports ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, GetPeriodicTable
    from rdkit import RDLogger
    from rdkit.Geometry import Point3D
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.vectors import calc_dihedral

    RDLogger.DisableLog('rdApp.*')
except ImportError as e:
    print(f"FATAL: Missing required libraries. Please install RDKit and BioPython. Error: {e}")
    exit(1)

# --- Global Variables ---
CONFIG: Optional[dict] = None
PROJECT_ROOT: Optional[Path] = None
PERIODIC_TABLE = GetPeriodicTable()


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
def one_hot_encode(value, allowed_list, fallback_to_other=True):
    encoding = [0.0] * len(allowed_list)
    try:
        idx = allowed_list.index(value)
        encoding[idx] = 1.0
    except ValueError:
        if fallback_to_other and 'Other' in allowed_list:
            encoding[allowed_list.index('Other')] = 1.0
    return encoding


def get_residue_id(residue) -> Tuple[str, int, str]:
    res_id = residue.get_id()
    chain_id = residue.get_parent().id
    insertion_code = res_id[2].strip()
    return chain_id, res_id[1], insertion_code


def build_rdkit_mol_from_atoms(atoms: List, pdb_id: str, res_id_str: str, config: dict):
    mol_rw = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    for i, atom in enumerate(atoms):
        try:
            element_symbol = atom.element.strip().capitalize()
            atomic_num = PERIODIC_TABLE.GetAtomicNumber(element_symbol)
            rdkit_atom = Chem.Atom(atomic_num)
            mol_rw.AddAtom(rdkit_atom)
            coords = atom.get_coord()
            point = Point3D(float(coords[0]), float(coords[1]), float(coords[2]))
            conformer.SetAtomPosition(i, point)
        except Exception as e:
            logging.debug(f"[{pdb_id} - {res_id_str}] Could not process atom {atom.get_name()}: {e}")
            return None
    mol_rw.AddConformer(conformer, assignId=True)
    mol = mol_rw.GetMol()
    if mol is None or mol.GetNumAtoms() == 0: return None
    editable_mol = Chem.RWMol(mol)
    dist_matrix = Chem.Get3DDistanceMatrix(mol)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            try:
                radius_i = PERIODIC_TABLE.GetRcovalent(atoms[i].element.strip()) * config['bond_perception_scale']
                radius_j = PERIODIC_TABLE.GetRcovalent(atoms[j].element.strip()) * config['bond_perception_scale']
                if 0.1 < dist_matrix[i, j] < (radius_i + radius_j):
                    if editable_mol.GetBondBetweenAtoms(i, j) is None:
                        editable_mol.AddBond(i, j, Chem.BondType.SINGLE)
            except Exception:
                continue
    final_mol = editable_mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol, catchErrors=True)
        final_mol.UpdatePropertyCache(strict=False)
    except Exception as e:
        logging.debug(f"[{pdb_id} - {res_id_str}] RDKit sanitization failed: {e}")
    return final_mol


def get_atom_features(atom: Chem.Atom, b_factor: float, config: dict):
    atom_symbols = config['atom_symbols']
    hybrid_types_str = config['hybridization_types']
    hybrid_map = {h: getattr(Chem.HybridizationType, h) for h in hybrid_types_str}
    features = []
    features.extend(one_hot_encode(atom.GetSymbol(), atom_symbols))
    features.append(float(atom.GetAtomicNum()));
    features.append(b_factor)
    features.append(float(atom.GetDegree()));
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()));
    features.append(float(atom.GetTotalNumHs(includeNeighbors=True)))
    features.append(float(atom.IsInRing()));
    features.append(float(atom.GetIsAromatic()))
    features.extend(one_hot_encode(atom.GetHybridization(), list(hybrid_map.values())))
    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond, conformer: Chem.Conformer, config: dict):
    bond_types_str = config['bond_types']
    bond_map = {b: getattr(Chem.BondType, b) for b in bond_types_str}
    features = one_hot_encode(bond.GetBondType(), list(bond_map.values()))
    length = AllChem.GetBondLength(conformer, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    features.append(length)
    features.append(float(bond.IsInRing()))
    return np.array(features, dtype=np.float32)


def calculate_chi_angles(residue: 'Bio.PDB.Residue', config: dict):
    chi_defs = config['chi_angle_definitions']
    res_name = residue.get_resname()
    chi_sincos = np.zeros((5, 2), dtype=np.float32)
    if res_name not in chi_defs: return chi_sincos
    for i, atom_names in enumerate(chi_defs[res_name]):
        if i >= 5: break
        try:
            vecs = [residue[name].get_vector() for name in atom_names]
            angle_rad = calc_dihedral(vecs[0], vecs[1], vecs[2], vecs[3])
            chi_sincos[i, 0] = math.sin(angle_rad)
            chi_sincos[i, 1] = math.cos(angle_rad)
        except KeyError:
            continue
    return chi_sincos


# --- Main Workflow Function ---
def create_sidechain_graphs_for_pdb(pdb_id: str, config: dict):
    """Processes a single PDB, creates all sidechain graphs, and saves them as a list with identifiers."""
    task_config = config.get('pipeline_tasks', {}).get('sidechain_graph_task', {})
    input_dir = PROJECT_ROOT / task_config['input_dir']
    output_dir = PROJECT_ROOT / task_config['output_dir']
    sidechain_file = input_dir / pdb_id / f"{pdb_id}{task_config['sidechain_suffix']}"
    backbone_file = input_dir / pdb_id / f"{pdb_id}{task_config['backbone_suffix']}"
    output_file = output_dir / pdb_id / f"{pdb_id}_sidechain.pt"

    if not task_config.get('overwrite_existing', False) and output_file.exists():
        logging.debug(f"[{pdb_id}] Skipping: Output file already exists.")
        return 'skipped'

    if not sidechain_file.exists() or not backbone_file.exists():
        logging.warning(f"[{pdb_id}] Sidechain or backbone file not found. Skipping.")
        return 'no_input'

    try:
        parser = PDBParser(QUIET=True)
        sidechain_struct = parser.get_structure(f"{pdb_id}_sc", str(sidechain_file))
        backbone_struct = parser.get_structure(f"{pdb_id}_bb", str(backbone_file))
        full_residues_map = {get_residue_id(res): res for res in backbone_struct.get_residues() if
                             is_aa(res, standard=True)}

        all_sidechain_graphs = []
        for residue in sidechain_struct.get_residues():
            res_id_tuple = get_residue_id(residue)
            if not is_aa(residue, standard=True) or residue.get_resname() == 'GLY':
                continue

            sidechain_atoms = list(residue.get_atoms())
            if not sidechain_atoms: continue

            atom_names_in_order = [atom.get_name().strip() for atom in sidechain_atoms]

            res_id_str = f"{res_id_tuple[0]}_{res_id_tuple[1]}"
            sidechain_mol = build_rdkit_mol_from_atoms(sidechain_atoms, pdb_id, res_id_str, task_config)
            if not sidechain_mol or sidechain_mol.GetNumAtoms() == 0: continue

            conformer = sidechain_mol.GetConformer()

            node_s_list = [get_atom_features(atom, sidechain_atoms[i].get_bfactor(), task_config) for i, atom in
                           enumerate(sidechain_mol.GetAtoms())]
            node_v_coords = np.array([conformer.GetAtomPosition(i) for i in range(sidechain_mol.GetNumAtoms())],
                                     dtype=np.float32)

            edge_idx, edge_s_list, edge_v_list = [], [], []
            for bond in sidechain_mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_idx.extend([[i, j], [j, i]])

                bond_feats = get_bond_features(bond, conformer, task_config)
                edge_s_list.extend([bond_feats, bond_feats])

                v_ij = node_v_coords[j] - node_v_coords[i]
                v_ji = node_v_coords[i] - node_v_coords[j]
                edge_v_list.extend([v_ij, v_ji])

            full_residue_context = full_residues_map.get(res_id_tuple)
            chi_angles = calculate_chi_angles(full_residue_context, task_config) if full_residue_context else np.zeros(
                (5, 2), dtype=np.float32)

            graph_data = {
                'residue_id': res_id_tuple,
                'atom_names': atom_names_in_order,
                'node_s': torch.from_numpy(np.array(node_s_list, dtype=np.float32)),
                'node_v_coords': torch.from_numpy(node_v_coords),
                'edge_index': torch.tensor(np.array(edge_idx).T, dtype=torch.long) if edge_idx else torch.empty((2, 0),
                                                                                                                dtype=torch.long),
                'edge_s': torch.from_numpy(np.array(edge_s_list, dtype=np.float32)),
                'edge_v': torch.from_numpy(np.array(edge_v_list, dtype=np.float32)) if edge_v_list else torch.empty(
                    (0, 3)),
                'chi_angles': torch.from_numpy(chi_angles)
            }
            all_sidechain_graphs.append(graph_data)

        if not all_sidechain_graphs:
            logging.info(f"[{pdb_id}] No valid sidechain graphs were generated (e.g., only GLY residues).")
            return 'no_graphs_generated'

        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_sidechain_graphs, output_file)
        logging.info(f"[{pdb_id}] Successfully created and saved {len(all_sidechain_graphs)} sidechain graphs.")
        return 'success'

    except Exception as e:
        logging.error(f"[{pdb_id}] An unexpected error occurred: {e}", exc_info=True)
        return 'processing_error'


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Generate atom-level graphs for protein sidechains.")
    script_dir = Path(__file__).parent
    default_config_path = script_dir.parent / 'config.yaml'
    parser.add_argument('--config', type=str, default=str(default_config_path),
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        load_config(config_path)
        setup_logging('sidechain_graph_log')
    except Exception as e:
        print(f"FATAL: Failed to initialize script: {e}")
        exit(1)

    logging.info("--- Starting Sidechain Graph Generation ---")
    task_config = CONFIG.get('pipeline_tasks', {}).get('sidechain_graph_task', {})
    if not task_config:
        logging.critical("'sidechain_graph_task' not found in config.yaml. Aborting.")
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
        status = create_sidechain_graphs_for_pdb(pdb_id, CONFIG)
        results[status] += 1
        if status not in ['success', 'skipped', 'no_graphs_generated']:
            failed_ids[status].append(pdb_id)

    logging.info("\n--- Sidechain Graph Generation Complete ---")
    total_processed = len(pdb_id_folders)
    logging.info(f"Total PDBs considered: {total_processed}")

    for status, count in sorted(results.items()):
        logging.info(f"  - {status.replace('_', ' ').capitalize():<25}: {count}")

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
        logging.info("All PDBs were processed successfully, skipped, or had no sidechains to graph.")


if __name__ == "__main__":
    main()