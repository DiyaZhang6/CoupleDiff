#!/usr/bin/env python
# /home/zdy/Project2/data_processing/combine.py

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
import yaml
from tqdm import tqdm


def load_config() -> Tuple[Dict[str, Any], Path]:
    """Load the project config from the repository root."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    base_dir = Path(config["project_base_dir"])
    return config, base_dir


def require_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Required config section '{key}' is missing or invalid.")
    return value


def setup_logging(config: Dict[str, Any], base_dir: Path) -> Tuple[logging.Logger, Path]:
    logging_cfg = require_mapping(config, "logging")
    log_cfg = require_mapping(logging_cfg, "combine_log")

    log_dir = base_dir / log_cfg["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    log_base_name = log_cfg["log_base_name"]
    use_timestamp = bool(log_cfg.get("use_timestamp_in_log_name", True))
    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{log_base_name}_{timestamp}.log"
    else:
        log_file = log_dir / f"{log_base_name}.log"

    log_level = getattr(logging, str(log_cfg.get("log_level", "INFO")).upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    logger = logging.getLogger("combine")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_file


def resolve_project_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def read_split_ids(csv_path: Path, logger: logging.Logger) -> List[str]:
    if not csv_path.is_file():
        logger.warning(f"Split file not found, skipping: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
        if "pdb_id" in df.columns:
            values = df["pdb_id"]
        else:
            df = pd.read_csv(csv_path, header=None)
            values = df.iloc[:, 0]
    except pd.errors.EmptyDataError:
        logger.warning(f"Split file is empty, skipping: {csv_path}")
        return []

    ids = []
    for value in values.dropna().astype(str):
        pdb_id = value.strip().lower()
        if pdb_id and pdb_id != "pdb_id":
            ids.append(pdb_id)

    return sorted(set(ids))


def collect_dataset_ids(config: Dict[str, Any], base_dir: Path, logger: logging.Logger) -> Tuple[Dict[str, List[str]], List[str]]:
    data_cfg = require_mapping(config, "data_loading")
    dataset_stats: Dict[str, List[str]] = {}
    all_pdb_ids: List[str] = []

    split_keys = {
        "train_split_file": "train",
        "val_split_file": "val",
    }
    for key, dataset_name in split_keys.items():
        if key not in data_cfg:
            continue
        split_path = resolve_project_path(base_dir, data_cfg[key])
        ids = read_split_ids(split_path, logger)
        if ids:
            dataset_stats[dataset_name] = ids
            all_pdb_ids.extend(ids)
            logger.info(f"Loaded {len(ids)} IDs from {dataset_name}: {split_path}")

    for test_set in data_cfg.get("test_sets", []):
        if not isinstance(test_set, dict) or "name" not in test_set or "path" not in test_set:
            logger.warning(f"Invalid test set config, skipping: {test_set}")
            continue
        split_path = resolve_project_path(base_dir, test_set["path"])
        ids = read_split_ids(split_path, logger)
        if ids:
            dataset_stats[test_set["name"]] = ids
            all_pdb_ids.extend(ids)
            logger.info(f"Loaded {len(ids)} IDs from {test_set['name']}: {split_path}")

    return dataset_stats, sorted(set(all_pdb_ids))


def ensure_combined_output_dir(data_cfg: Dict[str, Any], base_dir: Path, logger: logging.Logger) -> Path:
    logic_dir = resolve_project_path(base_dir, data_cfg["combined_dir"])
    physical_dir_cfg = data_cfg.get("physical_combined_dir")

    if physical_dir_cfg:
        physical_dir = resolve_project_path(base_dir, physical_dir_cfg)
        physical_dir.mkdir(parents=True, exist_ok=True)

        if logic_dir.exists():
            return logic_dir

        logic_dir.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(physical_dir, logic_dir)
        logger.info(f"Created symlink: {logic_dir} -> {physical_dir}")
        return logic_dir

    logic_dir.mkdir(parents=True, exist_ok=True)
    return logic_dir


def build_phy_geo_labels(raw_lbl: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bond": {
            "indices": raw_lbl.get("bond_indices"),
            "values": raw_lbl.get("ref_bond_lengths"),
        },
        "angle": {
            "indices": raw_lbl.get("angle_indices"),
            "values": raw_lbl.get("ref_angles"),
        },
        "dihedral": {
            "indices": raw_lbl.get("dihedral_indices"),
            "values": raw_lbl.get("true_dihedrals"),
        },
        "vdw": {
            "indices": raw_lbl.get("vdw_indices"),
            "values": raw_lbl.get("vdw_radii"),
        },
        "electro": {
            "indices": raw_lbl.get("electro_indices"),
        },
        "hbond": {
            "indices": raw_lbl.get("hbond_indices"),
        },
        "pi_pi": raw_lbl.get("pi_pi_ring_pair_indices", []),
        "partial_charges": raw_lbl.get("partial_charges"),
        "dipole_vectors": raw_lbl.get("dipole_vectors"),
    }


def validate_raw_labels(raw_lbl: Dict[str, Any], pdb_id: str) -> None:
    required_keys = [
        "r_true",
        "r_init",
        "atom_group_ids",
        "bond_indices",
        "ref_bond_lengths",
        "angle_indices",
        "ref_angles",
        "dihedral_indices",
        "true_dihedrals",
        "vdw_indices",
        "vdw_radii",
        "electro_indices",
        "partial_charges",
    ]
    missing = [key for key in required_keys if key not in raw_lbl]
    if missing:
        raise KeyError(f"{pdb_id}: label file is missing required keys: {', '.join(missing)}")


def save_combined_payload(
    pdb_id: str,
    graph_dir: Path,
    label_dir: Path,
    save_path: Path,
) -> None:
    p_bb = graph_dir / pdb_id / f"{pdb_id}_backbone.pt"
    p_drug = graph_dir / pdb_id / f"{pdb_id}_drug.pt"
    p_sidechain = graph_dir / pdb_id / f"{pdb_id}_sidechain.pt"
    p_label = label_dir / pdb_id / f"{pdb_id}_labels.pt"

    missing = [p.name for p in (p_bb, p_drug, p_label) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

    raw_lbl = torch.load(p_label, map_location="cpu")
    if not isinstance(raw_lbl, dict):
        raise TypeError(f"{p_label} did not contain a dict payload.")
    validate_raw_labels(raw_lbl, pdb_id)

    payload = {
        "pdb_id": pdb_id,
        "backbone": torch.load(p_bb, map_location="cpu"),
        "drug": torch.load(p_drug, map_location="cpu"),
        "sidechain": torch.load(p_sidechain, map_location="cpu") if p_sidechain.is_file() else None,
        "phy_geo_labels": build_phy_geo_labels(raw_lbl),
        "labels": raw_lbl,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, save_path)


def write_audit_report(
    log_file: Path,
    dataset_stats: Dict[str, List[str]],
    unique_pdb_ids: Iterable[str],
    results: Dict[str, List[str]],
    failure_reasons: Dict[str, str],
) -> None:
    success_set = set(results["success"])
    failed_set = set(results["failed"])
    skipped_set = set(results["skipped"])
    unique_pdb_ids = list(unique_pdb_ids)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"COMBINATION AUDIT REPORT - {datetime.datetime.now()}\n")
        f.write("=" * 60 + "\n")
        f.write("OVERALL SUMMARY:\n")
        f.write(f"Total Unique IDs:      {len(unique_pdb_ids)}\n")
        f.write(f"Successfully Combined: {len(results['success'])}\n")
        f.write(f"Skipped Existing:      {len(results['skipped'])}\n")
        f.write(f"Failed Total:          {len(results['failed'])}\n")
        f.write("-" * 60 + "\n")

        f.write("BREAKDOWN BY DATASET:\n")
        for dataset_name, ids in dataset_stats.items():
            s_ids = [pid for pid in ids if pid in success_set]
            f_ids = [pid for pid in ids if pid in failed_set]
            k_ids = [pid for pid in ids if pid in skipped_set]
            f.write(
                f"- {dataset_name:25}: Total {len(ids):5} | "
                f"Success {len(s_ids):5} | Failed {len(f_ids):5} | Skipped {len(k_ids):5}\n"
            )

        if results["failed"]:
            f.write("-" * 60 + "\n")
            f.write("DETAILED FAILURE LIST:\n")
            for pdb_id in results["failed"]:
                f.write(f"PDB_ID: {pdb_id:8} | REASON: {failure_reasons.get(pdb_id, 'Unknown')}\n")
        f.write("=" * 60 + "\n")


def combine_data() -> None:
    config, base_dir = load_config()
    logger, log_file = setup_logging(config, base_dir)
    logger.info("--- Starting Data Combination Task ---")
    logger.info(f"Project base directory: {base_dir}")

    data_cfg = require_mapping(config, "data_loading")
    combination_cfg = require_mapping(config, "data_combination")

    output_dir = ensure_combined_output_dir(data_cfg, base_dir, logger)
    graph_dir = resolve_project_path(base_dir, data_cfg["graph_dir"])
    label_dir = resolve_project_path(base_dir, data_cfg["labels_dir"])
    overwrite = bool(combination_cfg["overwrite"])

    logger.info(f"Graph directory: {graph_dir}")
    logger.info(f"Label directory: {label_dir}")
    logger.info(f"Combined output directory: {output_dir}")
    logger.info(f"Overwrite existing files: {overwrite}")

    dataset_stats, unique_pdb_ids = collect_dataset_ids(config, base_dir, logger)
    logger.info(f"Total unique PDB IDs to process: {len(unique_pdb_ids)}")
    if not unique_pdb_ids:
        logger.warning("No PDB IDs were found from configured split files. Nothing to combine.")
        return

    results = {"success": [], "failed": [], "skipped": []}
    failure_reasons: Dict[str, str] = {}

    for pdb_id in tqdm(unique_pdb_ids, desc="Combining PDBs"):
        save_path = output_dir / f"{pdb_id}.pt"

        if save_path.exists() and not overwrite:
            results["skipped"].append(pdb_id)
            continue

        try:
            save_combined_payload(pdb_id, graph_dir, label_dir, save_path)
            results["success"].append(pdb_id)
        except Exception as exc:
            results["failed"].append(pdb_id)
            failure_reasons[pdb_id] = str(exc)
            logger.error(f"PDB {pdb_id} failed: {exc}")

    write_audit_report(log_file, dataset_stats, unique_pdb_ids, results, failure_reasons)
    logger.info(f"Task complete. Detailed audit report saved to: {log_file}")


if __name__ == "__main__":
    combine_data()
