#!/usr/bin/env python
# Location: /home/zdy/Project2/data_processing/protein_pdbqt.py
# Log location: /home/zdy/Project2/logs/protein_prep/protein_preparation_20250613_150602.log
# Converts protein PDB files from multiple configured data sources into
# Vina-compatible PDBQT format using MGLTools' `prepare_receptor4.py`.
# The process is run in parallel with safe logging and configurable options.

import re
import logging
import logging.handlers
import subprocess
import yaml
import multiprocessing
import queue
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TypedDict

from tqdm import tqdm

# --- Global Variables ---
CONFIG: Optional[Dict] = None
PROJECT_ROOT: Optional[Path] = None


# --- Type Definitions ---
class WorkerResult(TypedDict):
    status: str
    pdb_file: str
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
    log_dir_rel = log_cfg.get('log_dir', 'logs/protein_prep')
    log_base_name = log_cfg.get('log_base_name', 'protein_preparation')
    log_level = log_cfg.get('log_level', 'INFO').upper()

    log_dir_abs = PROJECT_ROOT / log_dir_rel
    log_dir_abs.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir_abs / f"{log_base_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

    logging.info(f"Main logger configured. Log file: {log_file_path}")


def worker_initializer(q: multiprocessing.Queue):
    """Initializes each worker process with a queue handler for logging."""
    # All log messages from workers will be sent to the queue
    queue_handler = logging.handlers.QueueHandler(q)
    worker_logger = logging.getLogger()
    worker_logger.setLevel(logging.INFO)
    worker_logger.addHandler(queue_handler)


def prepare_receptor_worker(
        pdb_file_path_str: str,
        output_pdbqt_path_str: str,
        mgltools_paths: Dict[str, str]
) -> WorkerResult:
    """
    Worker function that calls the external prepare_receptor4.py script for a single file.
    """
    pdb_file_path = Path(pdb_file_path_str)
    output_pdbqt_path = Path(output_pdbqt_path_str)
    base_pdb_file = pdb_file_path.name

    pythonsh_path = mgltools_paths['pythonsh_path']
    script_path = mgltools_paths['prepare_receptor_path']

    command = [
        pythonsh_path, script_path,
        "-r", str(pdb_file_path),
        "-o", str(output_pdbqt_path),
        "-A", "hydrogens",  # Add hydrogens
        "-U", "nphs_lps_waters"  # Cleanup: merge non-polar H, lone pairs, and remove water
    ]

    logging.info(f"Processing: {base_pdb_file}")

    try:
        output_pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.returncode == 0 and output_pdbqt_path.exists():
            return WorkerResult(status='success', pdb_file=base_pdb_file, message=None)
        else:
            # Capture a concise error message
            error_msg = process.stderr.strip().split('\n')[-1]  # Often the last line is most informative
            full_error = f"prepare_receptor4.py failed. Code: {process.returncode}. Error: {error_msg}"
            return WorkerResult(status='error', pdb_file=base_pdb_file, message=full_error)

    except Exception as e:
        return WorkerResult(status='critical_error', pdb_file=base_pdb_file, message=str(e))


def main():
    """Main orchestration function."""
    config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
    try:
        load_config(config_path)
        setup_main_logging(CONFIG.get('logging', {}).get('protein_prep', {}))
    except Exception as e:
        print(f"FATAL: Could not initialize. Error: {e}")
        return

    logging.info("--- Starting Protein PDB to PDBQT Conversion ---")

    # --- 1. Gather all tasks from configuration ---
    prep_cfg = CONFIG.get('protein_preparation', {})
    mgl_cfg = CONFIG.get('mgltools', {})
    mp_cfg = CONFIG.get('multiprocessing', {})

    if not all([prep_cfg, mgl_cfg]):
        logging.critical("Missing 'protein_preparation' or 'mgltools' section in config.yaml. Aborting.")
        return

    skip_if_exists = prep_cfg.get('skip_if_exists', True)
    output_base_dir = PROJECT_ROOT / prep_cfg['output_base_dir']
    output_filename_template = prep_cfg['output_filename_template']

    tasks_to_run = []
    skipped_count = 0

    for source in prep_cfg.get('sources', []):
        name = source['name']
        input_dir = PROJECT_ROOT / source['input_dir']
        glob_pattern = source['file_glob_pattern']
        pdb_id_regex = re.compile(source['pdb_id_regex'])

        logging.info(f"Scanning source: '{name}' in '{input_dir}'")

        found_files = list(input_dir.glob(glob_pattern))
        if not found_files:
            logging.warning(f"No files found for source '{name}' with pattern '{glob_pattern}'")
            continue

        for pdb_file in found_files:
            match = pdb_id_regex.search(str(pdb_file))
            if not match:
                logging.warning(f"Could not extract PDB ID from path: {pdb_file}")
                continue

            pdb_id = match.group(1).lower()

            output_dir = output_base_dir / pdb_id
            output_file = output_dir / output_filename_template.format(pdb_id=pdb_id)

            if skip_if_exists and output_file.exists():
                skipped_count += 1
                continue

            tasks_to_run.append((str(pdb_file), str(output_file), mgl_cfg))

    logging.info(f"Found {len(tasks_to_run)} new protein files to process.")
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} files as their PDBQT output already exists.")

    if not tasks_to_run:
        logging.info("All tasks are complete. Exiting.")
        return

    # --- 2. Setup Multiprocessing and Run ---
    num_workers = mp_cfg.get('num_workers') or multiprocessing.cpu_count()
    chunk_size = mp_cfg.get('chunk_size', 1)

    log_queue = multiprocessing.Queue(-1)
    log_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    log_listener.start()

    logging.info(f"Starting parallel processing with {num_workers} workers...")

    results = []
    with multiprocessing.Pool(processes=num_workers, initializer=worker_initializer, initargs=(log_queue,)) as pool:
        # Use tqdm to show a progress bar
        pbar = tqdm(total=len(tasks_to_run), desc="Processing Proteins")
        for result in pool.starmap(prepare_receptor_worker, tasks_to_run, chunksize=chunk_size):
            results.append(result)
            pbar.update(1)
        pbar.close()

    log_listener.stop()

    # --- 3. Final Summary ---
    logging.info("--- All Protein Preparation Tasks Finished ---")
    logging.info(f"Total PDB files submitted for processing: {len(tasks_to_run)}")
    logging.info(f"Total PDB files skipped (already existed): {skipped_count}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    critical_error_count = sum(1 for r in results if r['status'] == 'critical_error')

    logging.info(f"Successfully written: {success_count}")
    logging.info(f"Total FAILED files (not written): {error_count + critical_error_count}")

    if error_count > 0:
        logging.error(f"--- FAILED: prepare_receptor4.py errors ({error_count}) ---")
        for res in results:
            if res['status'] == 'error':
                logging.error(f"  {res['pdb_file']} (Reason: {res['message']})")

    if critical_error_count > 0:
        logging.error(f"--- FAILED: Critical processing errors ({critical_error_count}) ---")
        for res in results:
            if res['status'] == 'critical_error':
                logging.error(f"  {res['pdb_file']} (Reason: {res['message']})")


if __name__ == '__main__':
    main()