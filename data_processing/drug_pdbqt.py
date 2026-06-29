#!/usr/bin/env python
# Location: /home/zdy/Project2/data_processing/drug_pdbqt.py
# Log location: /home/zdy/Project2/logs/drug_prep/drug_preparation_2025m13_163714.log
# Converts ligand SDF files from multiple configured data sources into
# Vina-compatible PDBQT format using the OpenBabel (pybel) library.
# The process is run in parallel with safe logging and configurable options.

import re
import logging
import logging.handlers
import yaml
import multiprocessing
import queue
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TypedDict
from collections import defaultdict

from tqdm import tqdm
from openbabel import pybel

# --- Global Variables & Type Definitions ---
CONFIG: Optional[Dict] = None
PROJECT_ROOT: Optional[Path] = None


class ConversionResult(TypedDict):
    status: str
    sdf_file: str
    message: Optional[str]


def load_config(config_path: Path) -> None:
    """Loads the global configuration from a YAML file."""
    global CONFIG, PROJECT_ROOT
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    PROJECT_ROOT = config_path.parent


def setup_main_logging(log_cfg: Dict):
    """Sets up the main process logger to listen to the queue."""
    log_dir_rel = log_cfg.get('log_dir', 'logs/drug_prep')
    log_base_name = log_cfg.get('log_base_name', 'drug_preparation_openbabel')
    log_level = log_cfg.get('log_level', 'INFO').upper()
    log_dir_abs = PROJECT_ROOT / log_dir_rel
    log_dir_abs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Ym%d_%H%M%S")
    log_file_path = log_dir_abs / f"{log_base_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

    logging.info(f"Main logger configured. Log file: {log_file_path}")


def worker_initializer(q: multiprocessing.Queue):
    """Initializes each worker process with a queue handler for logging."""
    queue_handler = logging.handlers.QueueHandler(q)
    worker_logger = logging.getLogger()
    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()
    worker_logger.setLevel(logging.INFO)
    worker_logger.addHandler(queue_handler)


def convert_ligand_worker(
        sdf_file_path_str: str,
        output_pdbqt_path_str: str,
        charge_model: str
) -> ConversionResult:
    """
    Worker function that uses OpenBabel to convert a single SDF to PDBQT,
    with robust handling for different pybel versions.
    """
    sdf_file_path = Path(sdf_file_path_str)
    output_pdbqt_path = Path(output_pdbqt_path_str)
    base_sdf_file = sdf_file_path.name

    try:
        molecules = list(pybel.readfile(format='sdf', filename=str(sdf_file_path)))
        if not molecules:
            return ConversionResult(status='error_no_molecule', sdf_file=base_sdf_file,
                                    message="SDF file is empty or unreadable.")

        mol = molecules[0]
        mol.addh()

        charge_calculation_ok = False  # Assume failure first
        try:
            # Attempt to calculate charges
            result = mol.calccharges(model=charge_model)

            if isinstance(result, list) or result == 0:
                charge_calculation_ok = True
            else:
                # It returned something unexpected (like a non-zero integer), treat as warning.
                charge_calculation_ok = False
                logging.warning(f"Charge calculation for {base_sdf_file} returned non-standard code: {result}")

        except Exception as e:
            # Any exception during calccharges is a failure
            charge_calculation_ok = False
            logging.warning(f"Charge calculation for {base_sdf_file} raised an exception: {e}")

        # --- Continue with writing the file ---
        output_pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        mol.write(format='pdbqt', filename=str(output_pdbqt_path), overwrite=True)

        if not output_pdbqt_path.exists() or output_pdbqt_path.stat().st_size == 0:
            return ConversionResult(status='error_write_failed', sdf_file=base_sdf_file,
                                    message="PDBQT file was not written or is empty.")

        if charge_calculation_ok:
            return ConversionResult(status='success', sdf_file=base_sdf_file, message=None)
        else:
            return ConversionResult(status='warning_charge', sdf_file=base_sdf_file,
                                    message=f"Molecule failed to charge using '{charge_model}'.")

    except Exception as e:
        return ConversionResult(status='critical_error', sdf_file=base_sdf_file, message=str(e))


def find_ligand_files(source_cfg: Dict) -> List[Tuple[str, str]]:
    """A finder function that locates ligand SDF files based on the source name."""
    name = source_cfg['name']
    input_dir = PROJECT_ROOT / source_cfg['input_dir']
    logging.info(f"Scanning source: '{name}' in '{input_dir}'")

    found_pairs = []
    if not input_dir.is_dir():
        logging.warning(f"Input directory for '{name}' not found: {input_dir}")
        return []

    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    for pdb_dir in tqdm(subdirs, desc=f"Scanning {name}", unit="dir"):
        sdf_path, pdb_id = None, ""

        if name == "PDBbind_v2020":
            pdb_id = pdb_dir.name
            prot_dir_glob = list(pdb_dir.glob("*_prot"))
            if prot_dir_glob:
                target_path = prot_dir_glob[0] / f"{pdb_id}_l.sdf"
                if target_path.is_file(): sdf_path = target_path

        elif name == "CASF_v2016":
            pdb_id = pdb_dir.name
            target_path = pdb_dir / f"{pdb_id}_ligand.sdf"
            if target_path.is_file(): sdf_path = target_path

        elif name == "PoseBusters_Benchmark":
            pdb_id = pdb_dir.name[:4]
            ligand_file = list(pdb_dir.glob("*_ligand.sdf"))
            if ligand_file: sdf_path = ligand_file[0]

        elif name == "Astex_Diverse_Set":
            pdb_id = pdb_dir.name[:4]
            ligand_file = list(pdb_dir.glob("*_ligand.sdf"))
            if not ligand_file:
                target_path = pdb_dir / "ligand.sdf"
                if target_path.is_file(): sdf_path = target_path
            else:
                sdf_path = ligand_file[0]

        if sdf_path and pdb_id:
            found_pairs.append((pdb_id, str(sdf_path)))
        else:
            logging.debug(f"No matching ligand file found in {pdb_dir} for source '{name}'")

    logging.info(f"Found {len(found_pairs)} ligand files for source '{name}'.")
    return found_pairs


def main():
    """Main orchestration function."""
    config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
    try:
        load_config(config_path)
        setup_main_logging(CONFIG.get('logging', {}).get('drug_prep', {}))
    except Exception as e:
        print(f"FATAL: Could not initialize. Error: {e}")
        return

    logging.info("--- Starting Ligand SDF to PDBQT Conversion using OpenBabel (pybel) ---")

    prep_cfg = CONFIG.get('drug_preparation', {})
    mp_cfg = CONFIG.get('multiprocessing', {})
    if not all([prep_cfg, mp_cfg]):
        logging.critical("Missing 'drug_preparation' or 'multiprocessing' section in config.yaml. Aborting.")
        return

    skip_if_exists = prep_cfg.get('skip_if_exists', True)
    charge_model = prep_cfg.get('charge_model', 'gasteiger')
    output_base_dir = PROJECT_ROOT / prep_cfg['output_base_dir']
    output_filename_template = prep_cfg['output_filename_template']

    tasks_to_run = []
    skipped_count = 0

    for source in prep_cfg.get('sources', []):
        for pdb_id, sdf_file_path in find_ligand_files(source):
            pdb_id_lower = pdb_id.lower()
            output_dir = output_base_dir / pdb_id_lower
            output_file = output_dir / output_filename_template.format(pdb_id=pdb_id_lower)

            if skip_if_exists and output_file.exists():
                skipped_count += 1
                continue

            tasks_to_run.append((sdf_file_path, str(output_file), charge_model))

    logging.info(f"Found a total of {len(tasks_to_run)} new ligand files to process across all sources.")
    if skipped_count > 0: logging.info(f"Skipped {skipped_count} files as PDBQT output already exists.")
    if not tasks_to_run: logging.info("All tasks are complete. Exiting."); return

    num_workers = mp_cfg.get('num_workers') or multiprocessing.cpu_count()
    chunk_size = mp_cfg.get('chunk_size', 1)

    log_queue = multiprocessing.Queue(-1)
    log_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    log_listener.start()

    logging.info(f"Starting parallel processing with {num_workers} workers...")

    results = []
    with multiprocessing.Pool(processes=num_workers, initializer=worker_initializer, initargs=(log_queue,)) as pool:
        pbar = tqdm(total=len(tasks_to_run), desc="Processing Ligands")
        for result in pool.starmap(convert_ligand_worker, tasks_to_run, chunksize=chunk_size):
            results.append(result)
            pbar.update(1)
        pbar.close()

    log_listener.stop()

    logging.info("--- All Ligand Preparation Tasks Finished ---")
    status_counts = defaultdict(int)
    failed_messages = defaultdict(list)

    for r in results:
        status_counts[r['status']] += 1
        if r['status'] != 'success':
            failed_messages[r['status']].append(f"{r['sdf_file']} (Reason: {r['message']})")

    logging.info(f"Total SDF files submitted for processing: {len(tasks_to_run)}")
    logging.info(f"Total SDF files skipped (already existed): {skipped_count}")
    logging.info(f"Successfully converted: {status_counts['success']}")

    if status_counts['warning_charge'] > 0:
        logging.warning(f"PDBQTs written with charge calculation warnings: {status_counts['warning_charge']}")
        logging.warning("--- SDFs with Charge Calculation Warnings ---")
        for msg in failed_messages['warning_charge']: logging.warning(f"  {msg}")

    total_failed = sum(v for k, v in status_counts.items() if 'error' in k)
    if total_failed > 0:
        logging.error(f"Total FAILED files that did not produce a PDBQT: {total_failed}")
        for status, messages in failed_messages.items():
            if 'error' in status:
                logging.error(f"--- FAILED: {status} ({len(messages)}) ---")
                for msg in messages: logging.error(f"  {msg}")


if __name__ == '__main__':
    main()