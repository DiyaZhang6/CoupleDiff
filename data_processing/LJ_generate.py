#!/usr/bin/env python
# /home/zdy/Project2/data_processing/LJ_generate.py (GPU-enabled)
# Reads pre-split protein backbone, sidechain, and ligand files (PDB/SDF),
# and computes Lennard-Jones interaction matrices between these components.
# Outputs matrices as .npy files in a structured directory.
# All parameters and paths are controlled by config.yaml.

import os
import logging
import logging.handlers
import yaml
import datetime
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# --- Third-party imports ---
try:
    from Bio.PDB import PDBParser
    from openbabel import pybel
except ImportError as e:
    print(f"FATAL: Missing required libraries. Please install BioPython and OpenBabel. Error: {e}")
    exit(1)

# --- GPU Support ---
try:
    import cupy

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# --- Global Variables ---
CONFIG: Optional[Dict] = None
PROJECT_ROOT: Optional[Path] = None

def setup_logging(log_config_key: str) -> None:
    if not CONFIG or not PROJECT_ROOT:
        print("FATAL: Configuration not loaded before setting up logging.")
        return

    log_config = CONFIG.get('logging', {}).get(log_config_key, {})
    if not log_config:
        print(f"FATAL: Logging configuration for '{log_config_key}' not found in config.yaml.")
        return

    log_level_str = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    use_timestamp = log_config.get('use_timestamp_in_log_name', True)

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{log_base_name}_{timestamp}.log"
    else:
        log_file_name = f"{log_base_name}.log"

    log_path_abs = PROJECT_ROOT / log_dir_rel / log_file_name

    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.getLevelName(log_level_str))

    log_path_abs.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path_abs, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info(f"Logging for '{log_config_key}' configured. Level: {log_level_str}. Output file: {log_path_abs}")


def load_config(config_path: Path) -> None:
    global CONFIG, PROJECT_ROOT
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    config_dir = config_path.parent
    project_base_relative = CONFIG.get('project_base_dir', '.')
    PROJECT_ROOT = (config_dir / project_base_relative).resolve()


AtomData = List[Tuple[np.ndarray, str]]


def read_atoms_from_pdb(file_path: Path) -> AtomData:
    atoms_data = []
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(file_path.stem, str(file_path))
        for atom in structure.get_atoms():
            element = getattr(atom, 'element', '').strip().upper()
            if element and element not in ('H', 'D'):
                atoms_data.append((atom.get_coord(), element))
    except Exception as e:
        logging.error(f"Error parsing PDB file {file_path}: {e}")
    return atoms_data


def read_atoms_from_sdf(file_path: Path) -> AtomData:
    atoms_data = []
    try:
        mol_generator = pybel.readfile(format='sdf', filename=str(file_path))
        mol = next(mol_generator)
        for atom in mol.atoms:
            if atom.atomicnum != 1:
                element_symbol = pybel.ob.GetSymbol(atom.atomicnum).strip().upper()
                atoms_data.append((np.array(atom.coords), element_symbol))
    except StopIteration:
        logging.warning(f"No molecules found in SDF file: {file_path}")
    except Exception as e:
        logging.error(f"Error parsing SDF file {file_path}: {e}")
    return atoms_data


def get_lj_params_from_config(element_symbol: str) -> Tuple[float, float]:
    task_config = CONFIG.get('pipeline_tasks', {}).get('lj_generation_task', {})
    lj_params_dict = task_config.get('lj_params', {})
    default_params = tuple(task_config.get('default_lj_params', (3.5, 0.1)))
    params = lj_params_dict.get(element_symbol)
    if params is None:
        logging.warning(f"Using default LJ parameters for element '{element_symbol}'.")
        return default_params
    return tuple(params)


def compute_lj_matrix_cpu(atoms1_data: AtomData, atoms2_data: AtomData) -> np.ndarray:
    num_atoms1, num_atoms2 = len(atoms1_data), len(atoms2_data)
    mat = np.zeros((num_atoms1, num_atoms2))
    params1 = [get_lj_params_from_config(elem) for _, elem in atoms1_data]
    params2 = [get_lj_params_from_config(elem) for _, elem in atoms2_data]
    for i, (pos1, _) in enumerate(atoms1_data):
        σ1, ε1 = params1[i]
        for j, (pos2, _) in enumerate(atoms2_data):
            σ2, ε2 = params2[j]
            σ_comb = (σ1 + σ2) / 2.0
            ε_comb = np.sqrt(ε1 * ε2)
            r = np.linalg.norm(pos1 - pos2)
            if r < 1e-6:
                mat[i, j] = np.inf
                continue
            term6 = (σ_comb / r) ** 6
            lj = 4 * ε_comb * (term6 ** 2 - term6)
            mat[i, j] = lj
    return mat


def compute_lj_matrix_gpu(atoms1_data: AtomData, atoms2_data: AtomData) -> np.ndarray:
    pos1_cpu = np.array([d[0] for d in atoms1_data])
    pos2_cpu = np.array([d[0] for d in atoms2_data])
    params1_cpu = np.array([get_lj_params_from_config(d[1]) for d in atoms1_data])
    params2_cpu = np.array([get_lj_params_from_config(d[1]) for d in atoms2_data])
    pos1_gpu, pos2_gpu = cupy.asarray(pos1_cpu), cupy.asarray(pos2_cpu)
    params1_gpu, params2_gpu = cupy.asarray(params1_cpu), cupy.asarray(params2_cpu)
    dist_vec = pos1_gpu[:, None, :] - pos2_gpu[None, :, :]
    r = cupy.linalg.norm(dist_vec, axis=-1)
    r[r < 1e-6] = 1e-6
    sigma1, epsilon1 = params1_gpu[:, 0], params1_gpu[:, 1]
    sigma2, epsilon2 = params2_gpu[:, 0], params2_gpu[:, 1]
    sigma_comb = (sigma1[:, None] + sigma2[None, :]) / 2.0
    epsilon_comb = cupy.sqrt(epsilon1[:, None] * epsilon2[None, :])
    term6 = (sigma_comb / r) ** 6
    lj_potential = 4 * epsilon_comb * (term6 ** 2 - term6)
    return cupy.asnumpy(lj_potential)


# --- Main Workflow Function ---
def run_lj_generation() -> None:
    if not CONFIG or not PROJECT_ROOT:
        logging.error("Configuration is not loaded.")
        return

    task_config = CONFIG.get('pipeline_tasks', {}).get('lj_generation_task', {})
    if not task_config:
        logging.error("'lj_generation_task' configuration not found in config file.")
        return

    # --- Get overwrite setting from config ---
    overwrite_existing = task_config.get('overwrite_existing', False)
    logging.info(f"Overwrite mode: {overwrite_existing}")

    # Determine whether to use GPU or CPU
    use_gpu = task_config.get('use_gpu', False)
    if use_gpu and not CUPY_AVAILABLE:
        logging.warning("GPU acceleration requested, but CuPy is not installed. Falling back to CPU.")
        use_gpu = False

    compute_func = compute_lj_matrix_gpu if use_gpu else compute_lj_matrix_cpu

    if use_gpu:
        device_id = task_config.get('gpu_device_id', 0)
        cupy.cuda.Device(device_id).use()
        logging.info(f"Using GPU acceleration on device {device_id}.")
    else:
        logging.info("Using CPU for LJ calculations.")

    input_dir = PROJECT_ROOT / task_config['input_dir']
    output_base_dir = PROJECT_ROOT / task_config['output_dir']

    logging.info(f"Scanning for split component files in: {input_dir}")
    pdb_id_folders = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not pdb_id_folders:
        logging.warning(f"No PDB ID subdirectories found in {input_dir}. Exiting.")
        return

    total_systems_processed, total_errors, total_skipped = 0, 0, 0

    for system_input_dir in pdb_id_folders:
        pdb_id = system_input_dir.name

        # --- Check if output files exist and should be skipped ---
        system_output_dir = output_base_dir / pdb_id
        bs_out = system_output_dir / 'bs.npy'
        bd_out = system_output_dir / 'bd.npy'
        sd_out = system_output_dir / 'sd.npy'

        if not overwrite_existing and all(f.exists() for f in [bs_out, bd_out, sd_out]):
            logging.info(f"Skipping system {pdb_id}: All output files already exist.")
            total_skipped += 1
            continue

        logging.info(f"Processing system: {pdb_id}")

        backbone_file = system_input_dir / f"{pdb_id}{task_config['backbone_suffix']}"
        sidechain_file = system_input_dir / f"{pdb_id}{task_config['sidechain_suffix']}"
        ligand_file = system_input_dir / f"{pdb_id}{task_config['ligand_suffix']}"

        if not all(f.exists() for f in [backbone_file, sidechain_file, ligand_file]):
            logging.warning(f"  Missing one or more component files for {pdb_id}. Skipping.")
            total_errors += 1
            continue

        backbone_atoms = read_atoms_from_pdb(backbone_file)
        sidechain_atoms = read_atoms_from_pdb(sidechain_file)
        ligand_atoms = read_atoms_from_sdf(ligand_file)

        if not backbone_atoms and not sidechain_atoms:
            logging.warning(f"  No protein atoms (backbone or sidechain) found for {pdb_id}. Skipping.")
            total_errors += 1
            continue
        if not ligand_atoms:
            logging.warning(f"  No ligand atoms found for {pdb_id}. Skipping.")
            total_errors += 1
            continue

        system_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if backbone_atoms and sidechain_atoms:
                logging.info(f"  Calculating Backbone-Sidechain (bs) matrix...")
                lj_matrix_bs = compute_func(backbone_atoms, sidechain_atoms)
                np.save(bs_out, lj_matrix_bs)
                logging.info(f"    Saved bs.npy ({lj_matrix_bs.shape}) for {pdb_id}")

            if backbone_atoms and ligand_atoms:
                logging.info(f"  Calculating Backbone-Ligand (bd) matrix...")
                lj_matrix_bd = compute_func(backbone_atoms, ligand_atoms)
                np.save(bd_out, lj_matrix_bd)
                logging.info(f"    Saved bd.npy ({lj_matrix_bd.shape}) for {pdb_id}")

            if sidechain_atoms and ligand_atoms:
                logging.info(f"  Calculating Sidechain-Ligand (sd) matrix...")
                lj_matrix_sd = compute_func(sidechain_atoms, ligand_atoms)
                np.save(sd_out, lj_matrix_sd)
                logging.info(f"    Saved sd.npy ({lj_matrix_sd.shape}) for {pdb_id}")

            total_systems_processed += 1
        except Exception as e:
            logging.error(f"  Error during LJ matrix calculation or saving for {pdb_id}: {e}", exc_info=True)
            total_errors += 1

    logging.info("--- LJ Matrix Generation Complete ---")
    logging.info(f"Successfully processed systems: {total_systems_processed}")
    logging.info(f"Systems skipped (already exist): {total_skipped}")
    logging.info(f"Systems with errors/missing input: {total_errors}")


if __name__ == '__main__':
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        config_file_path = project_root / 'config.yaml'

        print(f"Loading configuration from: {config_file_path}")
        load_config(config_file_path)
        setup_logging(log_config_key='lj_generate_log')

        logging.info("--- Executing LJ_generate.py in standalone mode ---")
        run_lj_generation()

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        if logging.getLogger().hasHandlers():
            logging.critical(f"FATAL ERROR: {e}", exc_info=True)
    except Exception as e:
        if logging.getLogger().hasHandlers():
            logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        else:
            print(f"FATAL CRITICAL ERROR: {e}")