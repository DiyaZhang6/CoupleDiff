#!/usr/bin/env python
# /home/zdy/Project2/scripts/data.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.data import HeteroData, Batch
except ImportError:
    print("FATAL: PyTorch Geometric is not installed.")
    raise


REQUIRED_DATA_LOADING_KEYS = [
    "combined_dir",
    "lj_dir",
    "train_split_file",
    "val_split_file",
    "load_lj_matrices",
    "allow_missing_lj",
    "use_phy_geo_labels",
    "phy_geo_label_mode",
    "phy_geo_drop_unmapped",
    "phy_geo_nearest_max_distance",
    "coordinate_source",
    "allow_coordinate_fallback",
    "center_coordinates",
    "max_load_attempts",
    "retry_on_error",
    "log_load_errors",
    "pin_memory",
    "shuffle_train",
    "drop_last",
    "lj_backbone_atom_stride",
    "lj_backbone_ca_atom_offset",
    "r_init_noise_std",
    "r_init_noise_min_mse",
    "r_init_noise_apply_to",
    "r_init_noise_seed",
    "allow_group_coordinate_fallback",
    "max_total_nodes",
    "max_init_mse",
]

REQUIRED_MODEL_KEYS = [
    "in_scalar_dim_bb",
    "in_scalar_dim_sc",
    "in_scalar_dim_drug",
]


def require_config_keys(section_name: str, section: Dict[str, Any], keys: List[str]) -> None:
    missing = [key for key in keys if key not in section]
    if not missing:
        return

    missing_text = ", ".join(missing)
    raise KeyError(
        f"Missing config.yaml keys under '{section_name}': {missing_text}. "
        "Please update config.yaml before running scripts/data.py."
    )


def ensure_tensor(obj: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if obj is None:
        tensor = torch.tensor([])
    elif torch.is_tensor(obj):
        tensor = obj.detach().clone()
    elif isinstance(obj, np.ndarray):
        tensor = torch.from_numpy(obj)
    elif isinstance(obj, list):
        tensor = torch.tensor(obj)
    else:
        try:
            tensor = torch.from_numpy(np.asarray(obj))
        except Exception:
            tensor = torch.tensor([])

    return tensor.to(dtype=dtype) if dtype is not None else tensor


def resolve_project_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def read_split_table(split_file: Path) -> Tuple[List[str], Dict[str, float]]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    try:
        df = pd.read_csv(split_file)
        if "pdb_id" not in df.columns:
            df = pd.read_csv(split_file, header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Split file is empty: {split_file}") from exc

    if "pdb_id" in df.columns:
        id_series = df["pdb_id"]
        affinity_series = df["affinity"] if "affinity" in df.columns else None
    else:
        id_series = df.iloc[:, 0]
        affinity_series = df.iloc[:, 1] if df.shape[1] > 1 else None

    pdb_ids: List[str] = []
    affinity_by_id: Dict[str, float] = {}
    for row_idx, raw_id in id_series.dropna().items():
        pdb_id = str(raw_id).strip().lower()
        if not pdb_id or pdb_id == "pdb_id":
            continue
        pdb_ids.append(pdb_id)

        if affinity_series is not None:
            try:
                affinity_by_id[pdb_id] = float(affinity_series.loc[row_idx])
            except Exception:
                affinity_by_id[pdb_id] = float("nan")

    return pdb_ids, affinity_by_id


def empty_edge_index() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def empty_index(width: int) -> torch.Tensor:
    return torch.empty((0, width), dtype=torch.long)


def pad_or_trim_features(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((0, target_dim), dtype=torch.float32)
    if x.size(-1) == target_dim:
        return x.float()
    if x.size(-1) > target_dim:
        return x[:, :target_dim].float()
    pad = torch.zeros((x.size(0), target_dim - x.size(-1)), dtype=x.dtype)
    return torch.cat([x, pad], dim=-1).float()


class ProteinLigandGraphDataset(Dataset):
    def __init__(self, pdb_ids: List[str], config: Dict[str, Any], affinity_by_id: Optional[Dict[str, float]] = None):
        super().__init__()
        self.pdb_ids = [str(pid).strip().lower() for pid in pdb_ids if str(pid).strip()]
        self.config = config
        self.affinity_by_id = affinity_by_id or {}

        self.base_dir = Path(config["project_base_dir"])
        self.data_cfg = config["data_loading"]
        self.model_cfg = config["model_params"]
        require_config_keys("data_loading", self.data_cfg, REQUIRED_DATA_LOADING_KEYS)
        require_config_keys("model_params", self.model_cfg, REQUIRED_MODEL_KEYS)

        self.combined_dir = resolve_project_path(self.base_dir, self.data_cfg["combined_dir"])
        self.lj_dir = resolve_project_path(self.base_dir, self.data_cfg["lj_dir"])

        self.load_lj_matrices = bool(self.data_cfg["load_lj_matrices"])
        self.allow_missing_lj = bool(self.data_cfg["allow_missing_lj"])
        self.use_phy_geo_labels = bool(self.data_cfg["use_phy_geo_labels"])
        self.phy_geo_label_mode = str(self.data_cfg["phy_geo_label_mode"]).lower()
        self.phy_geo_drop_unmapped = bool(self.data_cfg["phy_geo_drop_unmapped"])
        self.phy_geo_nearest_max_distance = float(self.data_cfg["phy_geo_nearest_max_distance"])
        self.coordinate_source = str(self.data_cfg["coordinate_source"]).lower()
        self.allow_coordinate_fallback = bool(self.data_cfg["allow_coordinate_fallback"])
        self.center_coordinates = bool(self.data_cfg["center_coordinates"])
        self.max_load_attempts = int(self.data_cfg["max_load_attempts"])
        self.retry_on_error = bool(self.data_cfg["retry_on_error"])
        self.log_load_errors = bool(self.data_cfg["log_load_errors"])
        self.lj_backbone_atom_stride = int(self.data_cfg["lj_backbone_atom_stride"])
        self.lj_backbone_ca_atom_offset = int(self.data_cfg["lj_backbone_ca_atom_offset"])
        self.r_init_noise_std = float(self.data_cfg["r_init_noise_std"])
        self.r_init_noise_min_mse = float(self.data_cfg["r_init_noise_min_mse"])
        self.r_init_noise_apply_to = str(self.data_cfg["r_init_noise_apply_to"]).lower()
        self.r_init_noise_seed = int(self.data_cfg["r_init_noise_seed"])
        self.allow_group_coordinate_fallback = bool(self.data_cfg["allow_group_coordinate_fallback"])
        self.max_total_nodes = self.data_cfg["max_total_nodes"]
        self.max_init_mse = self.data_cfg["max_init_mse"]

        self.bb_in_dim = int(self.model_cfg["in_scalar_dim_bb"])
        self.sc_in_dim = int(self.model_cfg["in_scalar_dim_sc"])
        self.drug_in_dim = int(self.model_cfg["in_scalar_dim_drug"])

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> HeteroData:
        dataset_size = len(self.pdb_ids)
        if dataset_size == 0:
            raise IndexError("ProteinLigandGraphDataset is empty.")

        idx = int(idx) % dataset_size
        max_attempts = max(1, self.max_load_attempts)
        if self.retry_on_error:
            max_attempts = max(max_attempts, dataset_size)

        attempts = 0
        last_error: Optional[Exception] = None

        while attempts < max_attempts:
            current_idx = (idx + attempts) % dataset_size
            pdb_id = self.pdb_ids[current_idx]
            try:
                return self._load_one(pdb_id)
            except Exception as exc:
                last_error = exc
                if self.log_load_errors:
                    print(f"[data.py] Failed to load {pdb_id}: {exc}")
                attempts += 1
                if not self.retry_on_error:
                    break

        raise RuntimeError(f"Could not load a valid sample after {attempts} attempt(s). Last error: {last_error}")

    def _load_one(self, pdb_id: str) -> HeteroData:
        combined_path = self.combined_dir / f"{pdb_id}.pt"
        if not combined_path.is_file():
            raise FileNotFoundError(f"Combined file not found: {combined_path}")

        payload = torch.load(combined_path, map_location="cpu")
        if "labels" not in payload:
            raise KeyError(f"{pdb_id}: combined payload has no 'labels'. Re-run data_processing/combine.py first.")

        backbone = self._read_backbone(payload["backbone"])
        sidechain = self._read_sidechain(payload.get("sidechain"))
        drug = self._read_drug(payload["drug"])
        labels = payload["labels"]

        coords = self._build_model_coordinates(labels, backbone, sidechain, drug, pdb_id)
        r_init, r_true = coords["r_init"], coords["r_true"]
        n_bb, n_sc, n_drug = backbone["num_nodes"], sidechain["num_nodes"], drug["num_nodes"]
        r_init = self._maybe_add_r_init_noise(r_init, r_true, pdb_id)
        total_nodes = int(n_bb + n_sc + n_drug)

        if self.max_total_nodes is not None and total_nodes > int(self.max_total_nodes):
            raise ValueError(f"{pdb_id}: total model nodes {total_nodes} exceeds data_loading.max_total_nodes={self.max_total_nodes}.")

        if self.center_coordinates:
            r_init = r_init - r_init.mean(dim=0, keepdim=True)
            r_true = r_true - r_true.mean(dim=0, keepdim=True)

        init_mse = torch.mean((r_init.float() - r_true.float()) ** 2).item()
        if self.max_init_mse is not None and init_mse > float(self.max_init_mse):
            raise ValueError(f"{pdb_id}: init MSE {init_mse:.6g} exceeds data_loading.max_init_mse={self.max_init_mse}.")

        bb_slice = slice(0, n_bb)
        sc_slice = slice(n_bb, n_bb + n_sc)
        drug_slice = slice(n_bb + n_sc, n_bb + n_sc + n_drug)

        data = HeteroData()
        data.pdb_id = pdb_id

        data["backbone"].x = backbone["x"]
        data["backbone"].pos = r_init[bb_slice]
        data["backbone", "backbone"].edge_index = backbone["edge_index"]

        data["sidechain"].x = sidechain["x"]
        data["sidechain"].pos = r_init[sc_slice]
        data["sidechain", "sidechain"].edge_index = sidechain["edge_index"]

        data["drug"].x = drug["x"]
        data["drug"].pos = r_init[drug_slice]
        data["drug", "drug"].edge_index = drug["edge_index"]

        data.r_init = r_init
        data.r_true = r_true
        data.atom_group_ids = torch.cat(
            [
                torch.zeros(n_bb, dtype=torch.long),
                torch.ones(n_sc, dtype=torch.long),
                torch.full((n_drug,), 2, dtype=torch.long),
            ],
            dim=0,
        )

        data.affinity = torch.tensor([self.affinity_by_id.get(pdb_id, float("nan"))], dtype=torch.float32)

        self._attach_phy_geo_attrs(data, labels, backbone, sidechain, drug)
        self._attach_lj_matrices(data, pdb_id, n_bb, n_sc, n_drug)

        return data

    def _read_backbone(self, bb: Dict[str, Any]) -> Dict[str, Any]:
        x = pad_or_trim_features(ensure_tensor(bb.get("node_s"), torch.float32), self.bb_in_dim)
        node_v = bb.get("node_v", {})
        if not isinstance(node_v, dict) or "ca_coord" not in node_v:
            raise KeyError("Backbone graph is missing node_v['ca_coord'].")
        pos = ensure_tensor(node_v["ca_coord"], torch.float32)
        edge_index = ensure_tensor(bb.get("edge_index"), torch.long)
        if edge_index.numel() == 0:
            edge_index = empty_edge_index()
        return {"x": x, "pos": pos, "edge_index": edge_index, "num_nodes": x.size(0)}

    def _read_sidechain(self, sc_payload: Any) -> Dict[str, Any]:
        if sc_payload is None:
            return {
                "x": torch.zeros((0, self.sc_in_dim), dtype=torch.float32),
                "pos": torch.zeros((0, 3), dtype=torch.float32),
                "edge_index": empty_edge_index(),
                "num_nodes": 0,
            }

        graphs = sc_payload if isinstance(sc_payload, list) else [sc_payload]
        x_parts: List[torch.Tensor] = []
        pos_parts: List[torch.Tensor] = []
        edge_parts: List[torch.Tensor] = []
        offset = 0

        for graph in graphs:
            if not isinstance(graph, dict):
                continue
            x = pad_or_trim_features(ensure_tensor(graph.get("node_s"), torch.float32), self.sc_in_dim)
            pos = ensure_tensor(graph.get("node_v_coords", graph.get("node_v")), torch.float32)
            if x.numel() == 0 or pos.numel() == 0:
                continue

            edge_index = ensure_tensor(graph.get("edge_index"), torch.long)
            if edge_index.numel() > 0:
                edge_parts.append(edge_index + offset)

            x_parts.append(x)
            pos_parts.append(pos)
            offset += x.size(0)

        if not x_parts:
            return {
                "x": torch.zeros((0, self.sc_in_dim), dtype=torch.float32),
                "pos": torch.zeros((0, 3), dtype=torch.float32),
                "edge_index": empty_edge_index(),
                "num_nodes": 0,
            }

        x_all = torch.cat(x_parts, dim=0)
        pos_all = torch.cat(pos_parts, dim=0)
        edge_index_all = torch.cat(edge_parts, dim=1) if edge_parts else empty_edge_index()
        return {"x": x_all, "pos": pos_all, "edge_index": edge_index_all, "num_nodes": x_all.size(0)}

    def _read_drug(self, drug: Dict[str, Any]) -> Dict[str, Any]:
        x = pad_or_trim_features(ensure_tensor(drug.get("node_scalar_features"), torch.float32), self.drug_in_dim)
        pos = ensure_tensor(drug.get("atom_coordinates"), torch.float32)
        edge_index = ensure_tensor(drug.get("edge_index"), torch.long)
        if edge_index.numel() == 0:
            edge_index = empty_edge_index()
        return {"x": x, "pos": pos, "edge_index": edge_index, "num_nodes": x.size(0)}

    def _build_model_coordinates(
        self,
        labels: Dict[str, Any],
        backbone: Dict[str, Any],
        sidechain: Dict[str, Any],
        drug: Dict[str, Any],
        pdb_id: str,
    ) -> Dict[str, torch.Tensor]:
        graph_init = torch.cat([backbone["pos"], sidechain["pos"], drug["pos"]], dim=0).float()

        if self.coordinate_source == "graph":
            return {"r_init": graph_init.clone(), "r_true": graph_init.clone()}

        if self.coordinate_source not in {"labels", "graph_to_labels"}:
            raise ValueError(f"Unsupported data_loading.coordinate_source: {self.coordinate_source}")

        try:
            r_true = ensure_tensor(labels["r_true"], torch.float32)
            group_ids = ensure_tensor(labels["atom_group_ids"], torch.long)

            true_parts = self._select_model_ordered_coords(
                r_true,
                group_ids,
                backbone,
                sidechain,
                drug,
                allow_group_fallback=self.allow_group_coordinate_fallback,
            )

            if self.coordinate_source == "graph_to_labels":
                return {
                    "r_init": graph_init.clone(),
                    "r_true": torch.cat(true_parts, dim=0),
                }

            r_init = ensure_tensor(labels["r_init"], torch.float32)
            init_parts = self._select_model_ordered_coords(
                r_init,
                group_ids,
                backbone,
                sidechain,
                drug,
                allow_group_fallback=self.allow_group_coordinate_fallback,
            )
            return {
                "r_init": torch.cat(init_parts, dim=0),
                "r_true": torch.cat(true_parts, dim=0),
            }
        except Exception as exc:
            if not self.allow_coordinate_fallback:
                raise
            if self.log_load_errors:
                print(f"[data.py] {pdb_id}: falling back to graph coordinates because label coordinate mapping failed: {exc}")
            return {"r_init": graph_init.clone(), "r_true": graph_init.clone()}

    def _select_model_ordered_coords(
        self,
        coords: torch.Tensor,
        group_ids: torch.Tensor,
        backbone: Dict[str, Any],
        sidechain: Dict[str, Any],
        drug: Dict[str, Any],
        allow_group_fallback: bool,
    ) -> List[torch.Tensor]:
        n_bb, n_sc, n_drug = backbone["num_nodes"], sidechain["num_nodes"], drug["num_nodes"]

        bb_all = coords[group_ids == 0]
        sc_all = coords[group_ids == 1]
        drug_all = coords[group_ids == 2]

        bb_coords = self._select_backbone_ca_coords(bb_all, n_bb)
        sc_coords = self._select_group_or_fallback(
            sc_all,
            sidechain["pos"],
            n_sc,
            "sidechain",
            allow_group_fallback=allow_group_fallback,
        )
        drug_coords = self._select_group_or_fallback(
            drug_all,
            drug["pos"],
            n_drug,
            "drug",
            allow_group_fallback=allow_group_fallback,
        )

        return [bb_coords, sc_coords, drug_coords]

    def _select_backbone_ca_coords(self, bb_all: torch.Tensor, n_bb: int) -> torch.Tensor:
        if n_bb == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        if bb_all.size(0) == n_bb:
            return bb_all

        stride = self.lj_backbone_atom_stride
        offset = self.lj_backbone_ca_atom_offset
        if stride > 0 and bb_all.size(0) >= offset + stride * (n_bb - 1) + 1:
            selected = bb_all[offset : offset + stride * n_bb : stride]
            if selected.size(0) == n_bb:
                return selected

        if bb_all.size(0) % n_bb == 0:
            dynamic_stride = bb_all.size(0) // n_bb
            dynamic_offset = min(offset, dynamic_stride - 1)
            selected = bb_all[dynamic_offset::dynamic_stride][:n_bb]
            if selected.size(0) == n_bb:
                return selected

        raise ValueError(f"Cannot map {bb_all.size(0)} backbone label atoms to {n_bb} backbone graph nodes.")

    def _select_group_or_fallback(
        self,
        label_coords: torch.Tensor,
        fallback_coords: torch.Tensor,
        expected_count: int,
        group_name: str,
        allow_group_fallback: bool,
    ) -> torch.Tensor:
        if expected_count == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        if label_coords.size(0) == expected_count:
            return label_coords
        if allow_group_fallback and fallback_coords.size(0) == expected_count:
            return fallback_coords.float()
        raise ValueError(
            f"Cannot map {label_coords.size(0)} {group_name} label atoms to {expected_count} {group_name} graph nodes."
        )

    def _maybe_add_r_init_noise(self, r_init: torch.Tensor, r_true: torch.Tensor, pdb_id: str) -> torch.Tensor:
        mode = self.r_init_noise_apply_to
        if mode == "never" or self.r_init_noise_std <= 0:
            return r_init

        init_mse = torch.mean((r_init.float() - r_true.float()) ** 2).item()
        if mode == "when_equal" and init_mse > self.r_init_noise_min_mse:
            return r_init
        if mode not in {"always", "when_equal"}:
            raise ValueError(
                "Unsupported data_loading.r_init_noise_apply_to: "
                f"{self.r_init_noise_apply_to}. Use 'never', 'when_equal', or 'always'."
            )

        seed_text = f"{self.r_init_noise_seed}:{pdb_id}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 31)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(r_init.shape, generator=generator, dtype=r_init.dtype) * self.r_init_noise_std
        return r_init + noise

    def _empty_phy_geo_attrs(self, data: HeteroData) -> None:
        data.bond_indices = empty_index(2)
        data.ref_bond_lengths = torch.empty(0, dtype=torch.float32)
        data.angle_indices = empty_index(3)
        data.ref_angles = torch.empty(0, dtype=torch.float32)
        data.dihedral_indices = empty_index(4)
        data.true_dihedrals = torch.empty(0, dtype=torch.float32)
        data.vdw_indices = empty_index(2)
        data.vdw_radii = torch.empty(0, dtype=torch.float32)
        data.electro_indices = empty_index(2)
        data.partial_charges = torch.empty(0, dtype=torch.float32)
        data.dipole_vectors = torch.empty((0, 3), dtype=torch.float32)
        data.hbond_indices = empty_index(3)
        data.pi_pi_ring_pair_indices = []

    def _attach_phy_geo_attrs(
        self,
        data: HeteroData,
        labels: Dict[str, Any],
        backbone: Dict[str, Any],
        sidechain: Dict[str, Any],
        drug: Dict[str, Any],
    ) -> None:
        if not self.use_phy_geo_labels:
            self._empty_phy_geo_attrs(data)
            return

        if self.phy_geo_label_mode == "raw":
            data.bond_indices = ensure_tensor(labels.get("bond_indices"), torch.long).view(-1, 2)
            data.ref_bond_lengths = ensure_tensor(labels.get("ref_bond_lengths"), torch.float32).view(-1)
            data.angle_indices = ensure_tensor(labels.get("angle_indices"), torch.long).view(-1, 3)
            data.ref_angles = ensure_tensor(labels.get("ref_angles"), torch.float32).view(-1)
            data.dihedral_indices = ensure_tensor(labels.get("dihedral_indices"), torch.long).view(-1, 4)
            data.true_dihedrals = ensure_tensor(labels.get("true_dihedrals"), torch.float32).view(-1)
            data.vdw_indices = ensure_tensor(labels.get("vdw_indices"), torch.long).view(-1, 2)
            data.vdw_radii = ensure_tensor(labels.get("vdw_radii"), torch.float32).view(-1)
            data.electro_indices = ensure_tensor(labels.get("electro_indices"), torch.long).view(-1, 2)
            data.partial_charges = torch.nan_to_num(
                ensure_tensor(labels.get("partial_charges"), torch.float32).view(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            data.dipole_vectors = self._raw_dipole_vectors(labels)
            data.hbond_indices = ensure_tensor(labels.get("hbond_indices"), torch.long).view(-1, 3)
            data.pi_pi_ring_pair_indices = self._raw_pi_pi_pairs(labels)
            return

        if self.phy_geo_label_mode != "mapped":
            raise ValueError(f"Unsupported data_loading.phy_geo_label_mode: {self.phy_geo_label_mode}")

        raw_to_model, model_to_raw = self._build_model_atom_index_mapping(labels, backbone, sidechain, drug)
        if model_to_raw.numel() == 0:
            self._empty_phy_geo_attrs(data)
            return

        def remap_indices(raw_indices: torch.Tensor, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
            raw_indices = ensure_tensor(raw_indices, torch.long).view(-1, width)
            if raw_indices.numel() == 0:
                return empty_index(width), torch.zeros(0, dtype=torch.bool)
            in_range = (raw_indices >= 0) & (raw_indices < raw_to_model.numel())
            mapped = torch.full_like(raw_indices, -1)
            if in_range.any():
                mapped[in_range] = raw_to_model[raw_indices[in_range]]
            valid = (mapped >= 0).all(dim=1)
            if not self.phy_geo_drop_unmapped and not valid.all():
                bad_count = int((~valid).sum().item())
                raise ValueError(f"Found {bad_count} unmapped phy_geo label rows.")
            return mapped[valid].long(), valid

        def filter_values(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            values = ensure_tensor(values, torch.float32).view(-1)
            if values.numel() == valid.numel():
                return values[valid]
            if values.numel() == 0:
                return torch.empty(0, dtype=torch.float32)
            if not self.phy_geo_drop_unmapped:
                raise ValueError(f"Cannot filter values of length {values.numel()} with mask length {valid.numel()}.")
            return torch.empty(0, dtype=torch.float32)

        data.bond_indices, bond_valid = remap_indices(labels.get("bond_indices"), 2)
        data.ref_bond_lengths = filter_values(labels.get("ref_bond_lengths"), bond_valid)
        data.angle_indices, angle_valid = remap_indices(labels.get("angle_indices"), 3)
        data.ref_angles = filter_values(labels.get("ref_angles"), angle_valid)
        data.dihedral_indices, dihedral_valid = remap_indices(labels.get("dihedral_indices"), 4)
        data.true_dihedrals = filter_values(labels.get("true_dihedrals"), dihedral_valid)
        data.vdw_indices, _ = remap_indices(labels.get("vdw_indices"), 2)
        data.electro_indices, _ = remap_indices(labels.get("electro_indices"), 2)
        data.hbond_indices, _ = remap_indices(labels.get("hbond_indices"), 3)

        raw_vdw = ensure_tensor(labels.get("vdw_radii"), torch.float32).view(-1)
        raw_charges = torch.nan_to_num(
            ensure_tensor(labels.get("partial_charges"), torch.float32).view(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        raw_dipoles = self._raw_dipole_vectors(labels)
        data.vdw_radii = self._select_model_values(raw_vdw, model_to_raw)
        data.partial_charges = self._select_model_values(raw_charges, model_to_raw)
        data.dipole_vectors = self._select_model_vectors(raw_dipoles, model_to_raw)
        data.pi_pi_ring_pair_indices = self._remap_pi_pi_pairs(self._raw_pi_pi_pairs(labels), raw_to_model)

    def _select_model_values(self, raw_values: torch.Tensor, model_to_raw: torch.Tensor) -> torch.Tensor:
        if model_to_raw.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        values = torch.zeros(model_to_raw.numel(), dtype=torch.float32)
        valid = (model_to_raw >= 0) & (model_to_raw < raw_values.numel())
        if valid.any():
            values[valid] = raw_values[model_to_raw[valid]]
        return values

    def _select_model_vectors(self, raw_vectors: torch.Tensor, model_to_raw: torch.Tensor) -> torch.Tensor:
        if model_to_raw.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        values = torch.zeros((model_to_raw.numel(), 3), dtype=torch.float32)
        valid = (model_to_raw >= 0) & (model_to_raw < raw_vectors.size(0))
        if valid.any():
            values[valid] = raw_vectors[model_to_raw[valid]]
        return values

    def _raw_dipole_vectors(self, labels: Dict[str, Any]) -> torch.Tensor:
        raw = labels.get("dipole_vectors")
        if raw is None:
            raw = labels.get("dipoles")
        dipoles = ensure_tensor(raw, torch.float32)
        if dipoles.numel() > 0:
            return torch.nan_to_num(dipoles.view(-1, 3), nan=0.0, posinf=0.0, neginf=0.0)

        charges = ensure_tensor(labels.get("partial_charges"), torch.float32).view(-1)
        coords = ensure_tensor(labels.get("r_true"), torch.float32).view(-1, 3)
        if charges.numel() == 0 or coords.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        if charges.numel() != coords.size(0):
            return torch.empty((0, 3), dtype=torch.float32)
        bond_indices = ensure_tensor(labels.get("bond_indices"), torch.long).view(-1, 2)
        return self._dipoles_from_charges_and_bonds(coords, charges, bond_indices)

    def _dipoles_from_charges_and_bonds(
        self,
        coords: torch.Tensor,
        charges: torch.Tensor,
        bond_indices: torch.Tensor,
    ) -> torch.Tensor:
        dipoles = torch.zeros((coords.size(0), 3), dtype=torch.float32)
        if bond_indices.numel() == 0:
            return dipoles
        valid = (
            (bond_indices[:, 0] >= 0)
            & (bond_indices[:, 0] < coords.size(0))
            & (bond_indices[:, 1] >= 0)
            & (bond_indices[:, 1] < coords.size(0))
        )
        bond_indices = bond_indices[valid]
        for i, j in bond_indices.tolist():
            direction = coords[j] - coords[i]
            norm = torch.linalg.norm(direction).clamp_min(1e-6)
            contribution = (charges[j] - charges[i]) * (direction / norm)
            dipoles[i] += contribution
            dipoles[j] -= contribution
        return torch.nan_to_num(dipoles, nan=0.0, posinf=0.0, neginf=0.0)

    def _raw_pi_pi_pairs(self, labels: Dict[str, Any]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        raw_pairs = labels.get("pi_pi_ring_pair_indices")
        if raw_pairs is None:
            raw_pairs = labels.get("pi_pi", [])
        pairs = []
        for pair in raw_pairs or []:
            if len(pair) != 2:
                continue
            ring_a = ensure_tensor(pair[0], torch.long).view(-1)
            ring_b = ensure_tensor(pair[1], torch.long).view(-1)
            if ring_a.numel() >= 3 and ring_b.numel() >= 3:
                pairs.append((ring_a, ring_b))
        return pairs

    def _remap_pi_pi_pairs(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        raw_to_model: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        remapped = []
        for ring_a, ring_b in pairs:
            valid_a = (ring_a >= 0) & (ring_a < raw_to_model.numel())
            valid_b = (ring_b >= 0) & (ring_b < raw_to_model.numel())
            mapped_a = raw_to_model[ring_a[valid_a]]
            mapped_b = raw_to_model[ring_b[valid_b]]
            mapped_a = mapped_a[mapped_a >= 0].long()
            mapped_b = mapped_b[mapped_b >= 0].long()
            if mapped_a.numel() >= 3 and mapped_b.numel() >= 3:
                remapped.append((mapped_a, mapped_b))
        return remapped

    def _build_model_atom_index_mapping(
        self,
        labels: Dict[str, Any],
        backbone: Dict[str, Any],
        sidechain: Dict[str, Any],
        drug: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        group_ids = ensure_tensor(labels.get("atom_group_ids"), torch.long).view(-1)
        if group_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        raw_coords = ensure_tensor(labels.get("r_true"), torch.float32).view(-1, 3)
        n_bb, n_sc, n_drug = backbone["num_nodes"], sidechain["num_nodes"], drug["num_nodes"]
        model_parts = []

        bb_raw = torch.nonzero(group_ids == 0, as_tuple=False).view(-1)
        model_parts.append(self._select_backbone_ca_indices(bb_raw, n_bb))

        sc_raw = torch.nonzero(group_ids == 1, as_tuple=False).view(-1)
        model_parts.append(
            self._select_group_indices_by_count_or_nearest(
                raw_coords,
                sc_raw,
                sidechain["pos"],
                n_sc,
            )
        )

        drug_raw = torch.nonzero(group_ids == 2, as_tuple=False).view(-1)
        model_parts.append(
            self._select_group_indices_by_count_or_nearest(
                raw_coords,
                drug_raw,
                drug["pos"],
                n_drug,
            )
        )

        model_to_raw = torch.cat(model_parts, dim=0) if model_parts else torch.empty(0, dtype=torch.long)
        raw_to_model = torch.full((group_ids.numel(),), -1, dtype=torch.long)
        valid = model_to_raw >= 0
        if valid.any():
            raw_to_model[model_to_raw[valid]] = torch.arange(model_to_raw.numel(), dtype=torch.long)[valid]
        return raw_to_model, model_to_raw

    def _select_backbone_ca_indices(self, bb_raw: torch.Tensor, n_bb: int) -> torch.Tensor:
        if n_bb == 0:
            return torch.empty(0, dtype=torch.long)
        if bb_raw.numel() == n_bb:
            return bb_raw.long()

        stride = self.lj_backbone_atom_stride
        offset = self.lj_backbone_ca_atom_offset
        if stride > 0 and bb_raw.numel() >= offset + stride * (n_bb - 1) + 1:
            selected = bb_raw[offset : offset + stride * n_bb : stride]
            if selected.numel() == n_bb:
                return selected.long()

        if bb_raw.numel() % n_bb == 0:
            dynamic_stride = bb_raw.numel() // n_bb
            dynamic_offset = min(offset, dynamic_stride - 1)
            selected = bb_raw[dynamic_offset::dynamic_stride][:n_bb]
            if selected.numel() == n_bb:
                return selected.long()

        return torch.full((n_bb,), -1, dtype=torch.long)

    def _select_group_indices_by_count_or_nearest(
        self,
        raw_coords: torch.Tensor,
        group_raw_indices: torch.Tensor,
        model_coords: torch.Tensor,
        expected_count: int,
    ) -> torch.Tensor:
        if expected_count == 0:
            return torch.empty(0, dtype=torch.long)
        if group_raw_indices.numel() == expected_count:
            return group_raw_indices.long()
        if group_raw_indices.numel() == 0 or model_coords.numel() == 0:
            return torch.full((expected_count,), -1, dtype=torch.long)

        group_raw_indices = group_raw_indices.long()
        raw_group_coords = raw_coords[group_raw_indices].float()
        model_coords = model_coords.float()
        distances = torch.cdist(model_coords, raw_group_coords)
        min_distances, nearest = torch.min(distances, dim=1)
        selected = group_raw_indices[nearest]
        selected[min_distances > self.phy_geo_nearest_max_distance] = -1
        return selected.long()

    def _attach_lj_matrices(self, data: HeteroData, pdb_id: str, n_bb: int, n_sc: int, n_drug: int) -> None:
        if not self.load_lj_matrices:
            data._lj_bs = torch.zeros((n_bb, n_sc), dtype=torch.float32)
            data._lj_bd = torch.zeros((n_bb, n_drug), dtype=torch.float32)
            data._lj_sd = torch.zeros((n_sc, n_drug), dtype=torch.float32)
            return

        lj_base = self.lj_dir / pdb_id
        data._lj_bs = self._load_lj(lj_base / "bs.npy", (n_bb, n_sc), reduce_backbone=True)
        data._lj_bd = self._load_lj(lj_base / "bd.npy", (n_bb, n_drug), reduce_backbone=True)
        data._lj_sd = self._load_lj(lj_base / "sd.npy", (n_sc, n_drug), reduce_backbone=False)

    def _load_lj(self, path: Path, expected_shape: Tuple[int, int], reduce_backbone: bool) -> torch.Tensor:
        rows, cols = expected_shape
        if rows == 0 or cols == 0:
            return torch.zeros(expected_shape, dtype=torch.float32)

        if not path.is_file():
            if self.allow_missing_lj:
                return torch.zeros(expected_shape, dtype=torch.float32)
            raise FileNotFoundError(f"LJ matrix not found: {path}")

        matrix = torch.from_numpy(np.load(path)).float()
        if reduce_backbone:
            matrix = self._reduce_backbone_lj_rows(matrix, rows)

        if matrix.shape != expected_shape:
            if self.allow_missing_lj:
                return torch.zeros(expected_shape, dtype=torch.float32)
            raise ValueError(f"{path} has shape {tuple(matrix.shape)}, expected {expected_shape}.")

        return matrix

    def _reduce_backbone_lj_rows(self, matrix: torch.Tensor, n_bb: int) -> torch.Tensor:
        if matrix.size(0) == n_bb:
            return matrix

        stride = self.lj_backbone_atom_stride
        offset = self.lj_backbone_ca_atom_offset
        if stride > 0 and matrix.size(0) >= offset + stride * (n_bb - 1) + 1:
            selected = matrix[offset : offset + stride * n_bb : stride]
            if selected.size(0) == n_bb:
                return selected

        if matrix.size(0) % n_bb == 0:
            dynamic_stride = matrix.size(0) // n_bb
            dynamic_offset = min(offset, dynamic_stride - 1)
            selected = matrix[dynamic_offset::dynamic_stride][:n_bb]
            if selected.size(0) == n_bb:
                return selected

        return matrix


def custom_hetero_collate_fn(data_list: List[HeteroData]) -> Optional[Batch]:
    data_list = [data for data in data_list if data is not None]
    if not data_list:
        return None

    lj_bs_list = [getattr(data, "_lj_bs", None) for data in data_list]
    lj_bd_list = [getattr(data, "_lj_bd", None) for data in data_list]
    lj_sd_list = [getattr(data, "_lj_sd", None) for data in data_list]
    pi_pi_pair_lists = [getattr(data, "pi_pi_ring_pair_indices", []) for data in data_list]
    for data in data_list:
        if hasattr(data, "pi_pi_ring_pair_indices"):
            delattr(data, "pi_pi_ring_pair_indices")

    batch = Batch.from_data_list(data_list)

    node_counts = [data["backbone"].num_nodes + data["sidechain"].num_nodes + data["drug"].num_nodes for data in data_list]
    node_offsets = np.cumsum([0] + node_counts)

    def collect_indices(attr_name: str, width: int) -> torch.Tensor:
        parts = []
        for i, data in enumerate(data_list):
            value = getattr(data, attr_name, None)
            if torch.is_tensor(value) and value.numel() > 0:
                parts.append(value.long() + int(node_offsets[i]))
        return torch.cat(parts, dim=0) if parts else empty_index(width)

    def collect_values(attr_name: str) -> torch.Tensor:
        parts = []
        for data in data_list:
            value = getattr(data, attr_name, None)
            if torch.is_tensor(value) and value.numel() > 0:
                parts.append(value.float().view(-1))
        return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.float32)

    def collect_vectors(attr_name: str) -> torch.Tensor:
        parts = []
        for data in data_list:
            value = getattr(data, attr_name, None)
            if torch.is_tensor(value) and value.numel() > 0:
                parts.append(value.float().view(-1, 3))
        return torch.cat(parts, dim=0) if parts else torch.empty((0, 3), dtype=torch.float32)

    def collect_pi_pi_pairs() -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pairs = []
        for i, pi_pi_pairs in enumerate(pi_pi_pair_lists):
            offset = int(node_offsets[i])
            for ring_a, ring_b in pi_pi_pairs:
                ring_a = ring_a.long() + offset
                ring_b = ring_b.long() + offset
                if ring_a.numel() >= 3 and ring_b.numel() >= 3:
                    pairs.append((ring_a, ring_b))
        return pairs

    batch.phy_geo_labels = {
        "bond": {
            "indices": collect_indices("bond_indices", 2),
            "values": collect_values("ref_bond_lengths"),
        },
        "angle": {
            "indices": collect_indices("angle_indices", 3),
            "values": collect_values("ref_angles"),
        },
        "dihedral": {
            "indices": collect_indices("dihedral_indices", 4),
            "values": collect_values("true_dihedrals"),
        },
        "vdw": {
            "indices": collect_indices("vdw_indices", 2),
            "values": collect_values("vdw_radii"),
        },
        "electro": {
            "indices": collect_indices("electro_indices", 2),
        },
        "hbond": {
            "indices": collect_indices("hbond_indices", 3),
        },
        "partial_charges": collect_values("partial_charges"),
        "dipole_vectors": collect_vectors("dipole_vectors"),
        "pi_pi": collect_pi_pi_pairs(),
    }

    for attr_name in [
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
        "dipole_vectors",
        "hbond_indices",
    ]:
        if hasattr(batch, attr_name):
            delattr(batch, attr_name)
        if attr_name in batch:
            del batch[attr_name]

    batch.lj_bs_list = lj_bs_list
    batch.lj_bd_list = lj_bd_list
    batch.lj_sd_list = lj_sd_list

    return batch


def resolve_split_file(config: Dict[str, Any], split: str) -> Path:
    base_dir = Path(config["project_base_dir"])
    data_cfg = config["data_loading"]

    normalized = split.lower()
    if normalized == "train":
        return resolve_project_path(base_dir, data_cfg["train_split_file"])
    if normalized in {"val", "valid", "validation"}:
        return resolve_project_path(base_dir, data_cfg["val_split_file"])
    if normalized == "test" and "test_split_file" in data_cfg:
        return resolve_project_path(base_dir, data_cfg["test_split_file"])

    for test_set in data_cfg.get("test_sets", []):
        if str(test_set.get("name", "")).lower() == normalized:
            return resolve_project_path(base_dir, test_set["path"])

    raise KeyError(f"Unknown split or test set: {split}")


def get_data_loader(config: Dict[str, Any], split: str, train_cfg: Optional[Dict[str, Any]] = None) -> DataLoader:
    require_config_keys("data_loading", config["data_loading"], REQUIRED_DATA_LOADING_KEYS)
    loader_cfg = train_cfg or config.get("training", {})
    if "batch_size" not in loader_cfg:
        raise KeyError("Missing data loader config key: batch_size")
    if "num_workers" not in loader_cfg and "num_workers" not in config.get("training", {}):
        raise KeyError("Missing data loader config key: num_workers")

    data_cfg = config["data_loading"]
    split_file = resolve_split_file(config, split)
    pdb_ids, affinity_by_id = read_split_table(split_file)
    dataset = ProteinLigandGraphDataset(pdb_ids, config, affinity_by_id=affinity_by_id)

    return DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=(split.lower() == "train" and bool(data_cfg["shuffle_train"])),
        num_workers=int(loader_cfg.get("num_workers", config["training"]["num_workers"])),
        pin_memory=bool(data_cfg["pin_memory"]),
        drop_last=bool(data_cfg["drop_last"]),
        collate_fn=custom_hetero_collate_fn,
    )
