#!/usr/bin/env python
# /home/zdy/Project2/data_processing/split.py
# A script to process multiple protein-ligand datasets. It splits proteins into backbone/sidechain and extracts the best ligand pose.

import os
import re
import logging.handlers
import warnings
import yaml
import multiprocessing
import datetime
import math
from typing import Optional, Dict, List, TypedDict, Generator

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from tqdm import tqdm

from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import RWMol
from rdkit.Geometry import Point3D
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Geometry import Point3D

# --- Global Variables & Constants ---
CONFIG: Optional[Dict] = None
PROJECT_ROOT: Optional[Path] = None
log_queue: Optional[multiprocessing.Queue] = None
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# --- Covalent radii for bond guessing ---
_COV_RAD = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07, 'S': 1.05,
    'CL': 1.02, 'BR': 1.20, 'I': 1.39, 'B': 0.85, 'SI': 1.11, 'FE': 1.25, 'MG': 1.30
}

_AUTODOCK_TYPE_MAP = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "OA": "O",
    "HD": "H",
    "O": "O",
    "S": "S",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
}

# --- Type Definitions ---
class ProteinComponents(TypedDict):
    backbone_atoms: List[Atom]
    sidechain_atoms: List[Atom]


class Task(TypedDict):
    pdb_id: str
    protein_path: Path
    complex_path: Path


class WorkerResult(TypedDict):
    status: str
    pdb_id: str
    message: Optional[str]


def _autodock_type_to_element(atom_type: str) -> str:
    """Map AutoDock atom type to standard element symbol."""
    return _AUTODOCK_TYPE_MAP.get(atom_type.upper(), "C")  # 默认C


def _parse_pdbqt_get_first_ligand_block(pdbqt_path: Path):
    """Return lines of the first ligand pose (skip receptor)."""
    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()

    ligand_lines = []
    in_ligand = False
    for line in lines:
        if line.startswith("ROOT"):
            in_ligand = True
            continue
        if line.startswith("ENDROOT"):
            in_ligand = False
            break
        if in_ligand:
            if line.startswith(("ATOM", "HETATM")):
                ligand_lines.append(line)
    return ligand_lines


def _should_bond(dist: float, elem1: str, elem2: str) -> bool:
    """Simple covalent bond guess based on distance (Å)."""
    covalent_radii = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "F": 0.57,
        "Cl": 1.02, "Br": 1.20, "I": 1.39
    }
    r1 = covalent_radii.get(elem1, 0.77)
    r2 = covalent_radii.get(elem2, 0.77)
    return dist < (r1 + r2 + 0.45)  # 0.45 Å 容差


# --- Logging Setup ---
def setup_main_logging_handlers(log_config_key: str) -> None:
    if not CONFIG or not PROJECT_ROOT:
        return
    log_config = CONFIG.get('logging', {}).get(log_config_key, {})
    log_level_str = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{log_base_name}_{timestamp}.log"
    log_path_abs = PROJECT_ROOT / log_dir_rel / log_file_name

    logger = logging.getLogger()
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            if not isinstance(handler, logging.handlers.QueueHandler):
                logger.removeHandler(handler)
    logger.setLevel(logging.getLevelName(log_level_str))
    log_path_abs.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path_abs, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.getLevelName(log_level_str))
    logger.addHandler(stream_handler)
    logging.info(f"Logging for '{log_config_key}' configured. File: {log_path_abs}")


def setup_worker_logging(q: multiprocessing.Queue) -> None:
    queue_handler = logging.handlers.QueueHandler(q)
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.handlers.QueueHandler) for h in root_logger.handlers):
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.INFO)


# --- Initialization and Configuration ---
def initialize_worker(q: multiprocessing.Queue, cfg: dict, pr_root: Path):
    global log_queue, CONFIG, PROJECT_ROOT
    log_queue, CONFIG, PROJECT_ROOT = q, cfg, pr_root
    setup_worker_logging(log_queue)


def load_config(config_path: Path) -> None:
    global CONFIG, PROJECT_ROOT
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    PROJECT_ROOT = config_path.parent.resolve()
    logging.info(f"Project root set to: {PROJECT_ROOT}")


# --- Ligand Extraction Helpers ---
def _guess_element_from_atom_name(atom_name: str) -> str:
    letters = ''.join(ch for ch in atom_name if ch.isalpha()).upper()
    if len(letters) >= 2 and letters[:2] in _COV_RAD:
        return letters[:2]
    if letters and letters[0] in _COV_RAD:
        return letters[0]
    return letters[0] if letters else 'C'


def _covalent_radius(elem: str) -> float:
    return _COV_RAD.get(elem.upper(), 0.77)


def _should_bond(dist: float, elem1: str, elem2: str, scale: float = 1.25) -> bool:
    return dist <= (_covalent_radius(elem1) + _covalent_radius(elem2)) * scale


def _parse_pdbqt_get_first_ligand_block(pdbqt_path: Path) -> List[str]:
    lines = pdbqt_path.read_text().splitlines()
    models, current = [], []
    recording = False
    for ln in lines:
        if ln.startswith('MODEL'):
            current = [ln]
            recording = True
            continue
        if ln.startswith('ENDMDL'):
            if recording:
                current.append(ln)
                models.append(current)
                current = []
            recording = False
            continue
        if recording:
            current.append(ln)
    # return first with ATOM/HETATM
    for m in models:
        if any(ln.startswith(('ATOM', 'HETATM')) for ln in m):
            return m
    return []


def save_ligand_from_complex(complex_pdbqt_path: Path, output_sdf_path: Path) -> bool:
    """
    Extract the best ligand pose from a Vina PDBQT file and save as SDF (safe version).
    """
    try:
        atom_lines = _parse_pdbqt_get_first_ligand_block(complex_pdbqt_path)
        if not atom_lines:
            logging.error(f"No ligand model found in {complex_pdbqt_path}")
            return False

        coords, elems = [], []
        for ln in atom_lines:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            parts = ln.split()
            if len(parts) < 6:
                continue

            try:
                x = float(ln[30:38].strip())
                y = float(ln[38:46].strip())
                z = float(ln[46:54].strip())
            except ValueError:
                logging.warning(f"Could not parse coordinates in line: {ln}")
                continue

            atom_type = parts[-1] if len(parts) > 8 else parts[2]
            elem = _autodock_type_to_element(atom_type)
            elems.append(elem)
            coords.append((x, y, z))

        n = len(elems)
        if n == 0:
            logging.error(f"No valid atoms found in ligand {complex_pdbqt_path}")
            return False

        rwmol = Chem.RWMol()
        for el in elems:
            rwmol.AddAtom(Chem.Atom(el))

        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                dist = math.dist((xi, yi, zi), (xj, yj, zj))
                if _should_bond(dist, elems[i], elems[j]):
                    try:
                        rwmol.AddBond(i, j, Chem.BondType.SINGLE)
                    except Exception:
                        pass

        mol = rwmol.GetMol()
        conf = Chem.Conformer(n)
        for idx, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(idx, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        writer = Chem.SDWriter(str(output_sdf_path))
        writer.write(mol)
        writer.close()
        return True

    except Exception as e:
        logging.error(f"RDKit failed to parse ligand block from {complex_pdbqt_path}: {e}")
        return False


# --- Protein Split ---
def save_atoms_as_pdb(atoms: List[Atom], structure: Structure, output_path: Path) -> None:
    class AtomSelect:
        def __init__(self, atom_list: List[Atom]):
            self.atom_set = {a.get_full_id() for a in atom_list}
        def accept_model(self, model): return 1
        def accept_chain(self, chain): return 1
        def accept_residue(self, residue): return 1
        def accept_atom(self, atom): return atom.get_full_id() in self.atom_set

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), select=AtomSelect(atoms))


def process_task(task: Task) -> WorkerResult:
    pdb_id = task['pdb_id']
    protein_path = task['protein_path']
    complex_path = task['complex_path']
    cfg = CONFIG.get('pipeline_tasks', {}).get('split_task')
    output_dir = PROJECT_ROOT / cfg['output_dir'] / pdb_id

    if cfg.get('skip_if_exists', True) and output_dir.exists():
        msg = f"Output for {pdb_id} exists. Skipping."
        logging.info(msg)
        return WorkerResult(status='skipped', pdb_id=pdb_id, message=msg)

    if not protein_path.is_file():
        return WorkerResult(status='failed', pdb_id=pdb_id, message="Protein file not found")
    if not complex_path.is_file():
        return WorkerResult(status='failed', pdb_id=pdb_id, message="Complex file not found")

    output_dir.mkdir(parents=True, exist_ok=True)

    parser = PDBParser(QUIET=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = parser.get_structure(pdb_id, str(protein_path))
    except Exception as e:
        return WorkerResult(status='failed', pdb_id=pdb_id, message=f"Bio.PDB failed: {e}")

    backbone, sidechain = [], []
    for atom in structure.get_atoms():
        if atom.get_parent().id[0] == ' ':
            if atom.get_name().strip().upper() in _BACKBONE_ATOMS:
                backbone.append(atom)
            else:
                sidechain.append(atom)

    if not backbone:
        return WorkerResult(status='failed', pdb_id=pdb_id, message="No backbone atoms found")

    try:
        backbone_path = output_dir / f"{pdb_id}{cfg['backbone_suffix']}"
        save_atoms_as_pdb(backbone, structure, backbone_path)

        sidechain_path = output_dir / f"{pdb_id}{cfg['sidechain_suffix']}"
        save_atoms_as_pdb(sidechain, structure, sidechain_path)

        ligand_path = output_dir / f"{pdb_id}{cfg['ligand_suffix']}"
        ligand_ok = save_ligand_from_complex(complex_path, ligand_path)
        if not ligand_ok:
            return WorkerResult(status='failed', pdb_id=pdb_id, message="Ligand save failed")

        logging.info(f"Successfully processed {pdb_id}.")
        return WorkerResult(status='success', pdb_id=pdb_id, message=None)

    except Exception as e:
        return WorkerResult(status='failed', pdb_id=pdb_id, message=f"Unexpected save error: {e}")


def discover_tasks(sources: List[Dict]) -> Generator[Task, None, None]:
    seen_ids = set()
    for source in sources:
        logging.info(f"--- Discovering tasks from source: {source['name']} ---")
        base_dir = PROJECT_ROOT / source['base_dir']
        protein_glob = source['protein_glob']
        id_regex = re.compile(source['protein_id_regex'])

        protein_files = list(base_dir.glob(protein_glob))
        logging.info(f"Found {len(protein_files)} potential protein files.")

        for protein_path in protein_files:
            match = id_regex.search(str(protein_path))
            if not match:
                continue
            pdb_id = match.group(1).lower()
            if pdb_id in seen_ids:
                continue
            complex_dir = PROJECT_ROOT / source['complex_dir'] / pdb_id
            complex_path = complex_dir / f"{pdb_id}{source['complex_suffix']}"
            yield Task(pdb_id=pdb_id, protein_path=protein_path, complex_path=complex_path)
            seen_ids.add(pdb_id)


def run_all_tasks_in_parallel() -> None:
    if not CONFIG:
        return
    task_config = CONFIG.get('pipeline_tasks', {}).get('split_task')
    tasks = list(discover_tasks(task_config['sources']))

    if not tasks:
        logging.warning("No tasks discovered.")
        return

    logging.info(f"Discovered {len(tasks)} tasks.")

    mp_config = CONFIG.get('multiprocessing', {})
    num_processes = mp_config.get('num_workers') or os.cpu_count() or 1
    chunk_size = mp_config.get('chunk_size', 1)

    logging.info(f"Starting parallel processing with {num_processes} workers...")
    with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker,
                              initargs=(log_queue, CONFIG, PROJECT_ROOT)) as pool:
        results = list(tqdm(pool.imap_unordered(process_task, tasks, chunksize=chunk_size),
                            total=len(tasks), desc="Processing PDBs"))

    logging.info("\n--- Splitting Process Complete ---")
    total = len(results)
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')

    logging.info(f"Total tasks processed: {total}")
    logging.info(f"  - Success: {success_count}")
    logging.info(f"  - Failed : {fail_count}")
    logging.info(f"  - Skipped: {skipped_count}")

    if fail_count > 0:
        logging.error("\n--- Failure Details ---")
        for r in results:
            if r['status'] == 'failed':
                logging.error(f"  - PDB ID: {r['pdb_id']}, Reason: {r['message']}")


if __name__ == '__main__':
    listener = None
    log_queue = multiprocessing.Queue(-1)
    try:
        config_file_path = Path(__file__).resolve().parent.parent / 'config.yaml'
        load_config(config_file_path)
        setup_main_logging_handlers('split_log')

        listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
        listener.start()

        logging.info("--- Executing split.py in parallel mode ---")
        run_all_tasks_in_parallel()

    except FileNotFoundError as e:
        logging.critical(f"Essential file not found: {e}", exc_info=False)
    except Exception as e:
        logging.critical("An unexpected critical error occurred.", exc_info=True)
    finally:
        if listener:
            listener.stop()
        if log_queue:
            log_queue.close()
            log_queue.join_thread()