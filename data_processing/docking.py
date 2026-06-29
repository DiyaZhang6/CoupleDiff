#!/usr/bin/env python
# /home/zdy/Project2/data_processing/docking.py
# Performs rigid molecular docking in parallel for all pre-prepared protein-ligand
# pairs found in a specified input directory. All parameters and paths are
# controlled by a central config.yaml file.
# pdbbind_v2020 and CASF2016 are in "/home/zdy/Project2/logs/docking/docking_20250828_100952.log"

import os
import logging
import logging.handlers
import subprocess
import yaml
import multiprocessing
import queue
import datetime
from pathlib import Path
from typing import Optional, Dict, List, TypedDict

# --- Global Variables ---
CONFIG: Optional[Dict] = None
PROJECT_ROOT: Optional[Path] = None
log_queue: Optional[multiprocessing.Queue] = None


# --- Type Definitions ---
class WorkerResult(TypedDict):
    status: str  # 'success', 'vina_error', 'file_not_found', 'subprocess_error', 'skipped'
    pdb_id: str
    message: Optional[str]


# --- Logging Setup ---
def setup_main_logging_handlers(log_config_key: str) -> None:
    if not CONFIG or not PROJECT_ROOT:
        print("FATAL: CONFIG or PROJECT_ROOT not loaded.")
        return

    log_config = CONFIG.get('logging', {}).get(log_config_key, {})
    if not log_config:
        print(f"FATAL: Logging configuration for '{log_config_key}' not found.")
        return

    log_level_str = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    use_timestamp = log_config.get('use_timestamp_in_log_name', False)

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{log_base_name}_{timestamp}.log"
    else:
        log_file_name = f"{log_base_name}.log"

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

    logging.info(f"Logging for '{log_config_key}' configured. Level: {log_level_str}. File: {log_path_abs}")


def setup_worker_logging(q: multiprocessing.Queue) -> None:
    queue_handler = logging.handlers.QueueHandler(q)
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.handlers.QueueHandler) for h in root_logger.handlers):
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.DEBUG)


# --- Multiprocessing Initialization ---
def initialize_worker(q: multiprocessing.Queue, cfg: dict, pr_root: Path):
    global log_queue, CONFIG, PROJECT_ROOT
    log_queue = q
    CONFIG = cfg
    PROJECT_ROOT = pr_root
    setup_worker_logging(log_queue)


# --- Configuration Loading ---
def load_config(config_path: str) -> None:
    global CONFIG, PROJECT_ROOT
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        CONFIG = yaml.safe_load(f)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Core Worker Function ---
def worker_run_vina_docking(pdb_id: str, base_cmd: List[str]) -> WorkerResult:
    logger = logging.getLogger(multiprocessing.current_process().name)
    docking_config = CONFIG.get('docking_input', {})

    # Get paths and suffixes from config
    input_dir = PROJECT_ROOT / docking_config['data_path']
    receptor_suffix = docking_config['receptor_suffix']
    ligand_suffix = docking_config['ligand_suffix']

    current_dir = input_dir / pdb_id
    receptor_path = current_dir / f"{pdb_id}{receptor_suffix}"
    ligand_path = current_dir / f"{pdb_id}{ligand_suffix}"
    output_path = current_dir / f"{pdb_id}.pdbqt"
    log_path = current_dir / f"{pdb_id}_docking.log"

    # Check if docking should be skipped based on config
    should_skip = docking_config.get('skip_if_exists', True)
    if should_skip and output_path.exists() and output_path.stat().st_size > 0:
        logger.debug(f"Skipping {pdb_id}: Output file '{output_path.name}' already exists.")
        return WorkerResult(status='skipped', pdb_id=pdb_id, message='Output file exists.')

    logger.info(f"Starting docking for PDB ID: {pdb_id}")

    if not receptor_path.exists() or not ligand_path.exists():
        msg = f"Receptor or ligand file not found for {pdb_id} in worker."
        logger.warning(msg)
        return WorkerResult(status='file_not_found', pdb_id=pdb_id, message=msg)

    # Build the final command for Vina, WITHOUT the --log argument
    final_cmd = base_cmd + [
        '--receptor', str(receptor_path),
        '--ligand', str(ligand_path),
        '--out', str(output_path),
    ]

    command_to_run_str = ' '.join(final_cmd)

    try:
        # Run the Vina process and capture its stdout and stderr
        process = subprocess.run(final_cmd, capture_output=True, text=True, check=False)

        # Manually write the Vina output (stdout) and any errors (stderr) to our log file
        vina_output_log = process.stdout + "\n" + process.stderr
        with open(log_path, 'w') as f:
            f.write(f"--- Vina Command ---\n{command_to_run_str}\n\n")
            f.write(f"--- Vina Output ---\n{vina_output_log}")

        if process.returncode == 0:
            logger.info(f"Docking successful for {pdb_id}. Log saved to {log_path.name}")
            return WorkerResult(status='success', pdb_id=pdb_id, message=None)
        else:
            # If Vina fails, log the error
            stderr_msg = process.stderr.strip()
            msg = f"Vina failed for {pdb_id}. Return code: {process.returncode}. Stderr: {stderr_msg}"
            logger.error(msg)
            return WorkerResult(status='vina_error', pdb_id=pdb_id, message=stderr_msg)

    except FileNotFoundError:
        msg = f"Vina executable '{base_cmd[0]}' not found. Check config."
        logger.error(msg)
        return WorkerResult(status='subprocess_error', pdb_id=pdb_id, message=msg)
    except Exception as e:
        msg = f"An unexpected error occurred during subprocess for {pdb_id}: {e}"
        logger.error(msg, exc_info=True)
        return WorkerResult(status='subprocess_error', pdb_id=pdb_id, message=str(e))


# --- Main Task Orchestration ---
def run_all_docking_tasks() -> None:
    if not CONFIG or not PROJECT_ROOT:
        logging.error("Configuration is not loaded. Aborting.")
        return

    docking_config = CONFIG.get('docking_input', {})
    vina_settings = CONFIG.get('vina_settings', {})
    if not docking_config or not vina_settings:
        logging.error("'docking_input' or 'vina_settings' not found in config.yaml.")
        return

    # Build the base command from settings
    base_cmd = [
        str(vina_settings['executable']),
        '--center_x', str(vina_settings['center_x']), '--center_y', str(vina_settings['center_y']), '--center_z',
        str(vina_settings['center_z']),
        '--size_x', str(vina_settings['size_x']), '--size_y', str(vina_settings['size_y']), '--size_z',
        str(vina_settings['size_z']),
        '--exhaustiveness', str(vina_settings['exhaustiveness']), '--num_modes', str(vina_settings['num_modes']),
        '--energy_range', str(vina_settings['energy_range']), '--seed', str(vina_settings['random_seed'])
    ]
    if vina_settings.get('use_gpu', False):
        if 'gpu_device_id' in vina_settings:
            base_cmd.extend(['--gpu_id', str(vina_settings['gpu_device_id'])])
    else:
        base_cmd.extend(['--cpu', str(vina_settings['cpu_threads'])])

    logging.info(f"Base Vina command configured. GPU enabled: {vina_settings.get('use_gpu', False)}")

    # Find all valid systems to process
    input_dir = PROJECT_ROOT / docking_config['data_path']
    if not input_dir.is_dir():
        logging.error(f"Docking input directory not found: {input_dir}")
        return

    tasks = []
    pdb_id_folders = [d for d in os.listdir(input_dir) if (input_dir / d).is_dir()]
    for pdb_id in pdb_id_folders:
        receptor_path = input_dir / pdb_id / f"{pdb_id}{docking_config['receptor_suffix']}"
        ligand_path = input_dir / pdb_id / f"{pdb_id}{docking_config['ligand_suffix']}"
        if receptor_path.exists() and ligand_path.exists():
            tasks.append((pdb_id, base_cmd))
        else:
            logging.warning(f"Skipping {pdb_id}: Receptor or ligand file not found.")

    if not tasks:
        logging.info("No valid protein-ligand pairs found to dock. Exiting.")
        return

    logging.info(f"Found {len(tasks)} valid systems to process. Starting parallel processing...")

    # Set up and run the multiprocessing pool
    mp_config = CONFIG.get('multiprocessing', {})
    num_processes = mp_config.get('num_workers') or os.cpu_count() or 1

    with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker,
                              initargs=(log_queue, CONFIG, PROJECT_ROOT)) as pool:
        results: List[WorkerResult] = pool.starmap(worker_run_vina_docking, tasks)

    # --- Detailed Summary Logging ---
    logging.info("--- All Docking Tasks Finished ---")
    logging.info(f"Total systems checked: {len(tasks)}")

    success_count = 0
    skipped_count = 0
    failures = []

    for res in results:
        if res['status'] == 'success':
            success_count += 1
        elif res['status'] == 'skipped':
            skipped_count += 1
        else:
            failures.append(res)

    logging.info(f"Total successful new dockings: {success_count}")
    logging.info(f"Total systems skipped (already done): {skipped_count}")
    logging.error(f"Total failures: {len(failures)}")

    if failures:
        logging.error("--- List of Failures ---")
        for f in failures:
            logging.error(f"  PDB ID: {f['pdb_id']}, Status: {f['status']}, Message: {f['message']}")


# --- Main Entry Point ---
if __name__ == '__main__':
    listener = None
    log_queue = None
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        config_file_path = project_root / 'config.yaml'

        load_config(str(config_file_path))

        log_queue = multiprocessing.Queue(-1)
        setup_main_logging_handlers(log_config_key='docking_log')

        listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
        listener.start()

        root_logger = logging.getLogger()
        root_logger.info("--- Executing docking.py in parallel mode ---")

        run_all_docking_tasks()

    except FileNotFoundError as e:
        logging.critical(f"Configuration or essential file not found: {e}", exc_info=True)
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        if listener:
            listener.stop()
        if log_queue:
            log_queue.close()
            log_queue.join_thread()