#!/usr/bin/env python
# Location: /home/zdy/Project2/data_processing/split_dataset.py
# A script to prepare and split a protein-ligand dataset based on a YAML config.

import re
import logging
import subprocess
import random
import json
import datetime
from pathlib import Path
from typing import Set, List, Dict, Any

import yaml
import pandas as pd
from tqdm import tqdm

# --- Constants ---
PROJECT_ROOT = None
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'TRP': 'W', 'THR': 'T',
    'TYR': 'Y', 'VAL': 'V'
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML configuration file and sets the project root."""
    global PROJECT_ROOT
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    PROJECT_ROOT = config_path.parent
    return cfg


def setup_logging(log_cfg: Dict[str, Any]):
    """Configures logging for the script."""
    log_dir_rel = log_cfg.get('log_dir', 'logs/split_dataset')
    log_base_name = log_cfg.get('log_base_name', 'split_dataset')
    log_level_str = log_cfg.get('log_level', 'INFO').upper()
    use_timestamp = log_cfg.get('use_timestamp_in_log_name', True)

    log_dir_abs = PROJECT_ROOT / log_dir_rel
    log_dir_abs.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{log_base_name}_{timestamp}.log" if use_timestamp else f"{log_base_name}.log"
    log_file_path = log_dir_abs / log_file_name

    logging.basicConfig(
        level=logging.getLevelName(log_level_str),
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logging configured. Log file: {log_file_path}")


def get_pdb_ids_from_dir(dir_path: Path) -> Set[str]:
    """Scans a directory to get PDB IDs from subdirectory names (in lowercase)."""
    if not dir_path.is_dir():
        logging.warning(f"Directory not found: {dir_path}. Returning empty set.")
        return set()
    ids = {d.name.lower() for d in dir_path.iterdir() if d.is_dir() and len(d.name) == 4}
    logging.info(f"Found {len(ids)} potential PDB IDs in: {dir_path}")
    return ids


def parse_log_for_ids(log_file: Path, pattern_str: str) -> Set[str]:
    """A generic function to parse a log file for PDB IDs (in lowercase)."""
    if not log_file.is_file():
        logging.warning(f"Log file not found: {log_file}. No IDs will be excluded from this log.")
        return set()

    ids = set()
    pattern = re.compile(pattern_str, re.IGNORECASE)
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()

    matches = pattern.findall(content)
    for match in matches:
        cleaned_ids = [pid.strip() for pid in match.replace('\n', '').split(',')]
        ids.update(pid.lower() for pid in cleaned_ids if len(pid.strip()) == 4)

    logging.info(f"Extracted {len(ids)} unique IDs to exclude from {log_file.name}.")
    return ids


def get_ids_from_benchmark_file(file_path: Path) -> Set[str]:
    """Reads a benchmark file, extracts the first 4 characters as PDB ID, and converts to lowercase."""
    if not file_path.is_file():
        logging.warning(f"Benchmark file not found: {file_path}. No IDs extracted.")
        return set()
    ids = {line.strip()[:4].lower() for line in open(file_path, 'r', errors='ignore') if len(line.strip()) >= 4}
    logging.info(f"Found {len(ids)} IDs in benchmark file: {file_path.name}")
    return ids


def load_affinity_data(index_file_path: Path) -> Dict[str, float]:
    """Parses the raw PDBbind index file, extracting -logKd/Ki and ensuring PDB IDs are lowercase."""
    if not index_file_path.is_file():
        raise FileNotFoundError(f"PDBbind index file not found: {index_file_path}")
    affinity_map = {}
    with open(index_file_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    pdb_id = parts[0].lower()
                    affinity_value = float(parts[3])
                    affinity_map[pdb_id] = affinity_value
                except (ValueError, IndexError):
                    continue
    logging.info(f"Loaded affinity data for {len(affinity_map)} complexes.")
    return affinity_map


def extract_sequence_from_pdbqt(pdb_id: str, protein_dir: Path) -> str:
    """Extracts 1-letter amino acid sequence from a PDBQT file."""
    pdbqt_file = protein_dir / f"{pdb_id}_protein.pdbqt"  # Simplified path
    if not pdbqt_file.is_file():
        # Fallback to check inside a subdirectory
        pdbqt_file = protein_dir / pdb_id / f"{pdb_id}_protein.pdbqt"
        if not pdbqt_file.is_file():
            return ""

    sequence, last_res_id = [], None
    with open(pdbqt_file, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    res_name = line[17:20].strip()
                    if res_name in AA_3_TO_1:
                        res_id = (line[21], int(line[22:26]), line[26].strip())
                        if res_id != last_res_id:
                            sequence.append(AA_3_TO_1[res_name])
                            last_res_id = res_id
                except (ValueError, IndexError):
                    continue
    return "".join(sequence)


def run_cd_hit(fasta_path: Path, output_prefix: Path, identity: float, threads: int, cd_hit_exe_path: str) -> Path:
    """Runs CD-HIT to cluster sequences."""
    if not Path(cd_hit_exe_path).is_file():
        raise EnvironmentError(f"CD-HIT executable not found at specified path: {cd_hit_exe_path}")
    clstr_file = output_prefix.with_suffix(".clstr")
    command = [
        cd_hit_exe_path, "-i", str(fasta_path), "-o", str(output_prefix),
        "-c", str(identity), "-n", "5", "-d", "0", "-T", str(threads), "-M", "0"
    ]
    logging.info(f"Running CD-HIT with command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logging.error(f"CD-HIT execution failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        raise RuntimeError("CD-HIT failed.")
    logging.info("CD-HIT clustering completed successfully.")
    return clstr_file


def parse_cd_hit_clusters(clstr_file: Path) -> List[List[str]]:
    """Parses a .clstr file into a list of clusters of PDB IDs."""
    clusters = []
    with open(clstr_file, 'r', errors='ignore') as f:
        current_cluster = []
        for line in f:
            if line.startswith(">Cluster"):
                if current_cluster: clusters.append(current_cluster)
                current_cluster = []
            else:
                match = re.search(r'>(\w{4})', line)
                if match:
                    current_cluster.append(match.group(1).lower())
        if current_cluster: clusters.append(current_cluster)
    logging.info(f"Parsed {len(clusters)} clusters from CD-HIT output.")
    return clusters


def enrich_and_save_df(pdb_ids: Set[str], affinity_map: Dict, output_path: Path, require_affinity: bool = True):
    """Creates a dataframe, enriches it with affinity, and saves to CSV."""
    if not pdb_ids:
        logging.warning(f"Received an empty set of IDs for {output_path.name}. No file will be created.")
        return

    df = pd.DataFrame(sorted(list(pdb_ids)), columns=['pdb_id'])
    df['affinity'] = df['pdb_id'].map(affinity_map)
    initial_count = len(df)

    if require_affinity:
        df.dropna(subset=['affinity'], inplace=True)
        final_count = len(df)
        dropped_count = initial_count - final_count
        log_msg = f"Saved {final_count} IDs to {output_path.name} (dropped {dropped_count} due to missing affinity)."
    else:
        final_count = len(df)
        log_msg = f"Saved {final_count} IDs to {output_path.name} (affinity data not required)."

    if final_count > 0:
        df.to_csv(output_path, index=False)
        logging.info(log_msg)
    else:
        logging.warning(f"No entries remaining for {output_path.name} after filtering. No file was created.")


def main():
    """Main execution function."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        config = load_config(config_path)
        setup_logging(config['logging']['split_dataset'])
    except Exception as e:
        logging.critical(f"FATAL: Could not initialize script. Error: {e}", exc_info=True)
        return

    try:
        cfg = config['dataset_split']

        # --- 1. Path Resolution ---
        logging.info("--- Step 1: Resolving Paths ---")
        output_dir = PROJECT_ROOT / cfg['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        pdbbind_dir = PROJECT_ROOT / cfg['pdbbind_dir']
        pdbbind_index_file = PROJECT_ROOT / cfg['pdbbind_index_file']
        casf_dir = PROJECT_ROOT / cfg['casf_dir']
        protein_dir = PROJECT_ROOT / cfg['prepared_protein_dir']
        temp_dir = PROJECT_ROOT / cfg['temp_dir']
        temp_dir.mkdir(parents=True, exist_ok=True)
        docking_log = PROJECT_ROOT / cfg['docking_log_file']
        backbone_log = PROJECT_ROOT / cfg['backbone_graph_log_file']
        posebusters_file = PROJECT_ROOT / cfg['posebusters_ids_file']
        astex_file = PROJECT_ROOT / cfg['astex_ids_file']

        # --- 2. Initial Data Loading and Filtering ---
        logging.info("\n--- Step 2: Initial Data Loading and Filtering ---")
        master_ids = get_pdb_ids_from_dir(pdbbind_dir)
        affinity_map = load_affinity_data(pdbbind_index_file)

        docking_failed_ids = parse_log_for_ids(docking_log, cfg['docking_failure_regex'])
        backbone_failed_ids = parse_log_for_ids(backbone_log, cfg['backbone_failure_regex'])

        all_failed_ids = docking_failed_ids.union(backbone_failed_ids)
        logging.info(f"Total unique IDs to exclude from logs: {len(all_failed_ids)}")
        master_ids = master_ids - all_failed_ids
        logging.info(f"After filtering based on logs, {len(master_ids)} IDs remain in the master pool.")

        # --- 3. Partitioning of Test & Benchmark Sets ---
        logging.info("\n--- Step 3: Partitioning Test and Benchmark Sets ---")

        # Partition 1: CASF Test Set
        casf_ids = get_pdb_ids_from_dir(casf_dir)
        test_ids = master_ids.intersection(casf_ids)
        enrich_and_save_df(test_ids, affinity_map, output_dir / "test.csv", require_affinity=True)
        master_ids = master_ids - test_ids
        logging.info(f"CASF test set created. {len(master_ids)} IDs remaining for train/val/astex.")

        # Partition 2: PoseBusters Set (Benchmark, affinity not required)
        posebusters_ids = get_ids_from_benchmark_file(posebusters_file)
        enrich_and_save_df(posebusters_ids, affinity_map, output_dir / "posebusters.csv", require_affinity=False)

        # Partition 3: Astex Set
        astex_ids = get_ids_from_benchmark_file(astex_file)
        astex_final_ids = master_ids.intersection(astex_ids)
        enrich_and_save_df(astex_final_ids, affinity_map, output_dir / "astex.csv", require_affinity=True)

        # PoseBusters and Astex IDs should also be removed from the master pool to avoid data leakage.
        master_ids = master_ids - astex_final_ids
        logging.info(f"Astex set created. {len(master_ids)} IDs remaining for train/val.")

        train_val_pool = master_ids - posebusters_ids  # Ensure PoseBusters IDs are not in train/val
        logging.info(
            f"After removing all test/benchmark sets, {len(train_val_pool)} IDs remain for training/validation pool.")

        # --- 4. Sequence Clustering for Train/Val Split ---
        logging.info("\n--- Step 4: Sequence Clustering for Train/Val Split ---")
        fasta_file = temp_dir / "train_val_pool.fasta"

        with open(fasta_file, 'w') as f:
            for pdb_id in tqdm(train_val_pool, desc="Generating FASTA file"):
                sequence = extract_sequence_from_pdbqt(pdb_id, protein_dir)
                if sequence: f.write(f">{pdb_id.upper()}\n{sequence}\n")

        try:
            cluster_file = run_cd_hit(fasta_file, temp_dir / "cd_hit_output", cfg['cd_hit_identity'],
                                      cfg.get('cd_hit_threads', 8), cfg['cd_hit_executable_path'])
            clusters = parse_cd_hit_clusters(cluster_file)
        except (EnvironmentError, RuntimeError) as e:
            logging.error(f"CD-HIT failed: {e}. Aborting.")
            return

        # --- 5. Final Splitting and Saving ---
        logging.info("\n--- Step 5: Final Splitting and Saving ---")
        random.seed(cfg['random_seed'])
        random.shuffle(clusters)

        split_idx = int(len(clusters) * (1.0 - cfg['validation_set_ratio']))
        train_clusters, valid_clusters = clusters[:split_idx], clusters[split_idx:]
        logging.info(
            f"Splitting {len(clusters)} clusters -> Train: {len(train_clusters)}, Valid: {len(valid_clusters)}")

        train_ids = {pid for cl in train_clusters for pid in cl}
        valid_ids = {pid for cl in valid_clusters for pid in cl}

        enrich_and_save_df(train_ids, affinity_map, output_dir / "train.csv", require_affinity=True)
        enrich_and_save_df(valid_ids, affinity_map, output_dir / "valid.csv", require_affinity=True)

        logging.info("--- Dataset Splitting Process Finished Successfully ---")

    except Exception as e:
        logging.critical(f"An unexpected error occurred during the process: {e}", exc_info=True)


if __name__ == "__main__":
    main()