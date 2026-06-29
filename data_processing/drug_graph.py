# /home/zdy/Project2/data_processing/drug_graph.py
# --- Creates atom-level graph for the drug molecule using RDKit ---
# Reads ligand SDF files, featurizes them, and saves as PyTorch tensor files.

import yaml
import numpy as np
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# --- RDKit Imports ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, GetPeriodicTable
    from rdkit.Chem import rdMolTransforms
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("Error: RDKit is required. Please install it (e.g., pip install rdkit or conda install rdkit).")
    exit(1)

# --- Define Project Root and Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / 'config.yaml'


# --- Utility Functions ---
def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Could not parse YAML configuration file: {e}")
        exit(1)


def setup_logging(log_config, project_root, log_base_name):
    """Sets up logging based on the configuration."""
    log_dir = Path(project_root) / log_config.get('log_dir', 'logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, log_config.get('log_level', 'INFO').upper(), logging.INFO)

    if log_config.get('use_timestamp_in_log_name', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_base_name}_{timestamp}.log"
    else:
        log_filename = f"{log_base_name}.log"

    log_filepath = log_dir / log_filename
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


def one_hot_encode(value, allowed_list):
    """Creates a one-hot encoding for a value in a list."""
    encoding = [0.0] * len(allowed_list)
    try:
        idx = allowed_list.index(str(value))
        encoding[idx] = 1.0
    except ValueError:
        if 'Other' in allowed_list:
            encoding[allowed_list.index('Other')] = 1.0
    return encoding


# --- Feature Calculation Functions ---
PERIODIC_TABLE = GetPeriodicTable()


def get_atom_scalar_features(rdkit_atom, cfg):
    """Calculates scalar features for a single RDKit atom based on config."""
    features = []
    features.extend(one_hot_encode(rdkit_atom.GetSymbol(), cfg['atom_symbols']))
    features.append(float(rdkit_atom.GetProp('_GasteigerCharge')) if rdkit_atom.HasProp('_GasteigerCharge') else 0.0)
    features.extend(one_hot_encode(rdkit_atom.GetHybridization(), cfg['hybridization_types']))
    features.extend(one_hot_encode(rdkit_atom.GetChiralTag(), cfg['chiral_tags']))
    features.append(float(rdkit_atom.GetIsAromatic()))
    features.append(rdkit_atom.GetMass())
    try:
        vdw_radius = PERIODIC_TABLE.GetRvdw(rdkit_atom.GetAtomicNum())
    except Exception:
        vdw_radius = 1.5
    features.append(vdw_radius)
    features.append(float(rdkit_atom.GetDegree()))
    features.append(float(rdkit_atom.IsInRing()))
    return np.array(features, dtype=np.float32)


def get_bond_scalar_features(rdkit_bond, conformer, cfg):
    """Calculates scalar features for a single RDKit bond based on config."""
    features = []
    features.extend(one_hot_encode(rdkit_bond.GetBondType(), cfg['bond_types']))
    try:
        length = rdMolTransforms.GetBondLength(conformer, rdkit_bond.GetBeginAtomIdx(), rdkit_bond.GetEndAtomIdx())
        features.append(length)
    except Exception:
        features.append(1.5)
    features.append(float(rdkit_bond.IsInRing()))
    return np.array(features, dtype=np.float32)


def get_geometric_features(rdkit_mol, logger):
    """
    Calculates geometric features for the molecule, including coordinates,
    bond angles, and dihedral angles.
    """
    geom_data = {
        'atom_coordinates': None,
        'bond_angles': [],
        'dihedral_angles': []
    }
    try:
        conformer = rdkit_mol.GetConformer()
        if conformer.GetNumAtoms() == 0:
            return geom_data

        geom_data['atom_coordinates'] = conformer.GetPositions().astype(np.float32)

        # --- Bond Angle Calculation ---
        for i in range(rdkit_mol.GetNumAtoms()):
            atom_i = rdkit_mol.GetAtomWithIdx(i)
            neighbors = [n.GetIdx() for n in atom_i.GetNeighbors()]
            if len(neighbors) >= 2:
                import itertools
                for j, k in itertools.combinations(neighbors, 2):
                    angle_rad = rdMolTransforms.GetAngleRad(conformer, j, i, k)
                    geom_data['bond_angles'].append(((j, i, k), angle_rad))

        # --- Dihedral Angle Calculation ---
        for bond in rdkit_mol.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            atom_j = rdkit_mol.GetAtomWithIdx(j)
            atom_k = rdkit_mol.GetAtomWithIdx(k)

            if atom_j.GetDegree() > 1 and atom_k.GetDegree() > 1:
                for neighbor_of_j in atom_j.GetNeighbors():
                    i = neighbor_of_j.GetIdx()
                    if i == k: continue
                    for neighbor_of_k in atom_k.GetNeighbors():
                        l = neighbor_of_k.GetIdx()
                        if l == j: continue
                        if len({i, j, k, l}) == 4:
                            dihedral_rad = rdMolTransforms.GetDihedralRad(conformer, i, j, k, l)
                            geom_data['dihedral_angles'].append(((i, j, k, l), dihedral_rad))

    except Exception as e:
        logger.error(f"Error in get_geometric_features: {e}", exc_info=True)

    return geom_data


# --- Main Graph Creation Function ---
def create_drug_graph_from_rdkit_mol(rdkit_mol, pdb_id, task_config, logger):
    """Creates an atom-level graph dictionary from an RDKit molecule."""
    if not rdkit_mol or rdkit_mol.GetNumAtoms() == 0: return None

    # Geometric features must be calculated first to get coordinates
    geometry = get_geometric_features(rdkit_mol, logger)
    if geometry['atom_coordinates'] is None: return None
    atom_coordinates_np = geometry['atom_coordinates']

    # Node scalar features
    node_features = [get_atom_scalar_features(atom, task_config) for atom in rdkit_mol.GetAtoms()]
    if not node_features: return None
    node_features_np = np.array(node_features, dtype=np.float32)

    # Edge features
    edge_indices, edge_scalar_features, edge_vector_features = [], [], []
    for rdkit_bond in rdkit_mol.GetBonds():
        i, j = rdkit_bond.GetBeginAtomIdx(), rdkit_bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

        bond_feat = get_bond_scalar_features(rdkit_bond, rdkit_mol.GetConformer(), task_config)
        edge_scalar_features.extend([bond_feat, bond_feat])

        v_ij = atom_coordinates_np[j] - atom_coordinates_np[i]
        v_ji = atom_coordinates_np[i] - atom_coordinates_np[j]
        edge_vector_features.extend([v_ij, v_ji])

    edge_index_np = np.array(edge_indices, dtype=np.int64).T if edge_indices else np.empty((2, 0), dtype=np.int64)
    edge_scalar_features_np = np.array(edge_scalar_features, dtype=np.float32)
    edge_vector_features_np = np.array(edge_vector_features, dtype=np.float32) if edge_vector_features else np.empty(
        (0, 3), dtype=np.float32)

    graph_data = {
        'node_scalar_features': torch.from_numpy(node_features_np),
        'edge_index': torch.from_numpy(edge_index_np),
        'edge_scalar_features': torch.from_numpy(edge_scalar_features_np),
        'edge_vector_features': torch.from_numpy(edge_vector_features_np),
        'atom_coordinates': torch.from_numpy(atom_coordinates_np),
        'bond_angles': geometry.get('bond_angles', []),
        'dihedral_angles': geometry.get('dihedral_angles', []),
    }

    logger.info(
        f"[{pdb_id}] Created drug graph: {graph_data['node_scalar_features'].shape[0]} atoms, "
        f"{rdkit_mol.GetNumBonds()} bonds."
    )
    return graph_data


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    logger = setup_logging(config['logging']['drug_graph_log'], PROJECT_ROOT, log_base_name="drug_graph")

    task_config = config.get('pipeline_tasks', {}).get('drug_graph_task', {})
    input_dir = PROJECT_ROOT / task_config['input_dir']
    output_dir = PROJECT_ROOT / task_config['output_dir']
    ligand_suffix = task_config['ligand_suffix']
    overwrite = task_config['overwrite_existing']

    logger.info("--- Starting Drug Graph Generation (with full geometry) ---")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    pdb_id_folders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not pdb_id_folders:
        logger.warning(f"No PDB ID subdirectories found in {input_dir}.")
        exit(0)
    logger.info(f"Found {len(pdb_id_folders)} PDB ID subdirectories to process.")

    results = defaultdict(int)
    failed_ids = defaultdict(list)

    for folder in tqdm(pdb_id_folders, desc="Generating Drug Graphs"):
        pdb_id = folder.name
        try:
            input_ligand_path = folder / f"{pdb_id}{ligand_suffix}"
            output_subdir = output_dir / pdb_id
            output_graph_path = output_subdir / f"{pdb_id}_drug.pt"

            if not overwrite and output_graph_path.exists():
                results['skipped'] += 1
                continue
            if not input_ligand_path.exists():
                logger.warning(f"[{pdb_id}] Input SDF file not found at: {input_ligand_path}")
                results['no_input_file'] += 1
                failed_ids['no_input_file'].append(pdb_id)
                continue

            suppl = Chem.SDMolSupplier(str(input_ligand_path), removeHs=False, sanitize=False)
            mol = next(suppl, None)

            if mol is None:
                logger.warning(f"[{pdb_id}] Could not load molecule from SDF: {input_ligand_path}")
                results['sdf_load_error'] += 1
                failed_ids['sdf_load_error'].append(pdb_id)
                continue

            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception as e:
                logger.warning(f"[{pdb_id}] SanitizeMol failed, molecule may be invalid: {e}")

            AllChem.ComputeGasteigerCharges(mol)

            drug_graph = create_drug_graph_from_rdkit_mol(mol, pdb_id, task_config, logger)

            if drug_graph is None:
                logger.error(f"[{pdb_id}] Failed to generate graph from molecule.")
                results['graph_creation_error'] += 1
                failed_ids['graph_creation_error'].append(pdb_id)
                continue

            output_subdir.mkdir(parents=True, exist_ok=True)
            torch.save(drug_graph, output_graph_path)
            results['success'] += 1

        except Exception as e:
            logger.error(f"[{pdb_id}] An unexpected error occurred: {e}", exc_info=True)
            results['unexpected_error'] += 1
            failed_ids['unexpected_error'].append(pdb_id)

    # --- Final Summary ---
    logger.info("\n--- Drug Graph Generation Summary ---")
    total_considered = len(pdb_id_folders)
    logger.info(f"Total PDB IDs considered: {total_considered}")

    for status, count in sorted(results.items()):
        logger.info(f"  - {status.replace('_', ' ').capitalize():<25}: {count}")

    total_failures = sum(len(v) for k, v in failed_ids.items())
    if total_failures > 0:
        logger.warning("\n--- Failure Report ---")
        for status, ids in sorted(failed_ids.items()):
            logger.warning(f"\nReason: {status.replace('_', ' ').capitalize()} ({len(ids)} PDBs):")
            chunk_size = 10
            id_chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
            for chunk in id_chunks:
                logger.warning(f"  IDs: {', '.join(chunk)}")
    else:
        logger.info("All PDBs were processed successfully or skipped as intended.")