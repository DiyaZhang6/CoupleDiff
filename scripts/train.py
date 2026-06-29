#!/usr/bin/env python
# /home/zdy/Project2/scripts/train.py

import datetime
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from torch_cluster import radius_graph
except ImportError:
    radius_graph = None

from data import get_data_loader
from models.loss import TotalLoss
from models.model import DynaModel


def resolve_project_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]["training_log"]
    base_dir = Path(config["project_base_dir"])
    log_dir = resolve_project_path(base_dir, log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    use_timestamp = bool(log_cfg.get("use_timestamp_in_log_name", True))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = log_cfg["log_base_name"]
    log_file = log_dir / f"{log_name}_{timestamp}.log" if use_timestamp else log_dir / f"{log_name}.log"

    logger = logging.getLogger("training")
    logger.setLevel(getattr(logging, str(log_cfg.get("log_level", "INFO")).upper()))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Training log: %s", log_file)
    return logger


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def move_phy_geo_labels(phy_geo_labels: Dict[str, Any], device: torch.device, config: Dict[str, Any]) -> Dict[str, Any]:
    loss_params = config.get("total_loss", {}).get("params", {})
    vdw_on_cpu = loss_params.get("vdw", {}).get("compute_device") == "cpu"
    electro_on_cpu = loss_params.get("electrostatic", {}).get("compute_device") == "cpu"

    moved = {}
    for key, value in phy_geo_labels.items():
        if key == "vdw" and vdw_on_cpu:
            moved[key] = value
        elif key == "electro" and electro_on_cpu:
            moved[key] = value
        elif key == "dipole_vectors" and electro_on_cpu:
            moved[key] = value
        elif key == "partial_charges" and electro_on_cpu:
            moved[key] = value
        else:
            moved[key] = move_to_device(value, device)
    return moved


def prepare_batch(batch: Any, device: torch.device, train_cfg: Dict[str, Any], config: Dict[str, Any]) -> Any:
    phy_geo_labels = getattr(batch, "phy_geo_labels", None)
    if hasattr(batch, "phy_geo_labels"):
        delattr(batch, "phy_geo_labels")

    batch = batch.to(device)

    batch.lj_bs_list = move_to_device(getattr(batch, "lj_bs_list", []), device)
    batch.lj_bd_list = move_to_device(getattr(batch, "lj_bd_list", []), device)
    batch.lj_sd_list = move_to_device(getattr(batch, "lj_sd_list", []), device)

    if phy_geo_labels is not None:
        batch.phy_geo_labels = move_phy_geo_labels(phy_geo_labels, device, config)

    if bool(train_cfg["build_global_graph_in_train"]):
        if radius_graph is None:
            raise RuntimeError("torch_cluster.radius_graph is required when build_global_graph_in_train=true.")

        all_pos = torch.cat(
            [batch["backbone"].pos, batch["sidechain"].pos, batch["drug"].pos],
            dim=0,
        )
        all_batch = torch.cat(
            [batch["backbone"].batch, batch["sidechain"].batch, batch["drug"].batch],
            dim=0,
        )
        r_max = float(config["decoder_params"]["rbf"]["d_max"])
        batch.global_edge_index = radius_graph(all_pos, r=r_max, batch=all_batch, loop=False)
        max_global_edges = train_cfg.get("max_global_edges")
        if max_global_edges is not None:
            max_global_edges = int(max_global_edges)
            num_edges = int(batch.global_edge_index.size(1))
            if max_global_edges > 0 and num_edges > max_global_edges:
                perm = torch.randperm(num_edges, device=batch.global_edge_index.device)[:max_global_edges]
                batch.global_edge_index = batch.global_edge_index[:, perm]

    return batch


def batch_summary(batch: Any) -> Dict[str, Any]:
    keys = []
    try:
        keys = list(batch.keys())
    except Exception:
        pass

    def edge_count(edge_type):
        src = edge_type[0]
        try:
            if edge_type in batch.edge_types:
                return int(batch[edge_type].edge_index.size(1))
        except Exception:
            pass
        try:
            if src in batch.node_types and "edge_index" in batch[src]:
                return int(batch[src].edge_index.size(1))
        except Exception:
            pass
        return -1

    return {
        "keys": keys,
        "backbone": int(batch["backbone"].num_nodes),
        "sidechain": int(batch["sidechain"].num_nodes),
        "drug": int(batch["drug"].num_nodes),
        "total_nodes": int(batch.r_init.size(0)) if hasattr(batch, "r_init") else -1,
        "bb_edges": edge_count(("backbone", "backbone")),
        "sc_edges": edge_count(("sidechain", "sidechain")),
        "drug_edges": edge_count(("drug", "drug")),
        "global_edges": int(batch.global_edge_index.size(1)) if hasattr(batch, "global_edge_index") else -1,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_item(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def checkpoint_metric_tag(metric_name: str, metric_value: Optional[float]) -> str:
    if metric_value is None or not np.isfinite(metric_value):
        return f"{metric_name}_na"
    value = f"{metric_value:.6e}".replace("+", "").replace("-", "m")
    return f"{metric_name}_{value}"


def decoder_coord_scale_abs_max(model: torch.nn.Module) -> float:
    values = []
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", []) if decoder is not None else []
    for layer in layers:
        raw = getattr(layer, "coord_scale_raw", None)
        max_value = getattr(layer, "coord_scale_max", None)
        if raw is not None and max_value is not None:
            scale = torch.tanh(raw.detach()).abs().max() * float(max_value)
            values.append(float(scale.cpu()))
    direct_raw = getattr(decoder, "direct_delta_scale_raw", None) if decoder is not None else None
    direct_max = getattr(decoder, "direct_delta_scale_max", None) if decoder is not None else None
    if direct_raw is not None and direct_max is not None:
        direct_scale = torch.tanh(direct_raw.detach()).abs().max() * float(direct_max)
        values.append(float(direct_scale.cpu()))
    return max(values) if values else 0.0


def is_decoder_coord_param(name: str) -> bool:
    if not name.startswith("decoder."):
        return False
    return any(
        token in name
        for token in [
            ".coord_scale_raw",
            ".phi_m.",
            ".psi.",
            ".direct_delta_head.",
            ".direct_delta_scale_raw",
        ]
    )


def build_optimizer(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> AdamW:
    base_lr = float(train_cfg["learning_rate"])
    base_weight_decay = float(train_cfg["weight_decay"])
    coord_lr_mult = float(train_cfg["coord_head_lr_multiplier"])
    coord_weight_decay = float(train_cfg["coord_head_weight_decay"])

    base_params = []
    coord_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_decoder_coord_param(name):
            coord_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {
            "params": base_params,
            "lr": base_lr,
            "weight_decay": base_weight_decay,
            "name": "base",
        },
        {
            "params": coord_params,
            "lr": base_lr * coord_lr_mult,
            "weight_decay": coord_weight_decay,
            "name": "decoder_coord_head",
        },
    ]
    return AdamW(param_groups)


def get_config_path() -> Path:
    env_path = os.environ.get("PROJECT2_CONFIG")
    return Path(env_path) if env_path else PROJECT_ROOT / "config.yaml"


def train() -> None:
    config_path = get_config_path()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    base_dir = Path(config["project_base_dir"])

    set_seed(int(train_cfg["seed"]))

    if "cuda_alloc_conf" in train_cfg and train_cfg["cuda_alloc_conf"] is not None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(train_cfg["cuda_alloc_conf"])

    logger = setup_logger(config)
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_cfg["use_amp"]) and device.type == "cuda"

    logger.info("Config: %s", config_path)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))
    logger.info("AMP enabled: %s", use_amp)

    logger.info("Initializing data loaders...")
    train_loader = get_data_loader(config, "train", train_cfg)
    val_loader = get_data_loader(config, "val", train_cfg)

    model = DynaModel(config).to(device)
    optimizer = build_optimizer(model, train_cfg)
    for group in optimizer.param_groups:
        logger.info(
            "Optimizer group '%s': params=%d lr=%.8e weight_decay=%.8e",
            group.get("name", "unnamed"),
            sum(p.numel() for p in group["params"]),
            group["lr"],
            group["weight_decay"],
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    scaler = GradScaler(enabled=use_amp)
    criterion = TotalLoss(config).to(device)
    checkpoint_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    start_epoch = 0

    resume_checkpoint = train_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        resume_path = resolve_project_path(base_dir, str(resume_checkpoint))
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if bool(train_cfg.get("resume_optimizer", True)) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if bool(train_cfg.get("resume_scheduler", True)) and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if bool(train_cfg.get("resume_scaler", True)) and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        logger.info("Resumed checkpoint: %s", resume_path)
        logger.info("Continuing from epoch %d to %d.", start_epoch + 1, int(train_cfg["epochs"]))

    max_train_batches = train_cfg.get("max_train_batches")
    max_val_batches = train_cfg.get("max_val_batches")
    log_every = int(train_cfg["log_every_n_steps"])
    grad_clip = train_cfg.get("gradient_clip_norm")
    use_diffusion_refinement = bool(train_cfg["use_diffusion_refinement"])

    logger.info("Starting training for %d epoch(s).", int(train_cfg["epochs"]))

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            if batch is None:
                continue

            stage = "start"
            try:
                stage = "prepare_batch"
                batch = prepare_batch(batch, device, train_cfg, config)
                stage = "zero_grad"
                optimizer.zero_grad(set_to_none=True)

                if batch_idx < 3 and device.type == "cuda":
                    logger.info(
                        "Forward prep batch %d summary=%s cuda_alloc=%.2fGiB cuda_reserved=%.2fGiB",
                        batch_idx,
                        batch_summary(batch),
                        torch.cuda.memory_allocated(device) / (1024 ** 3),
                        torch.cuda.memory_reserved(device) / (1024 ** 3),
                    )

                stage = "model_forward"
                with autocast(enabled=use_amp):
                    outputs = model(batch, use_diffusion_refinement=use_diffusion_refinement)

                stage = "loss_forward"
                outputs["true_coords"] = batch.r_true
                loss_res = criterion(outputs, batch)
                total_loss = loss_res["total_loss"]
                if not torch.isfinite(total_loss):
                    logger.warning(
                        "Non-finite loss at epoch %d batch %d; skipping batch. PDB %s",
                        epoch + 1,
                        batch_idx,
                        getattr(batch, "pdb_id", ["unknown"]),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if not total_loss.requires_grad:
                    loss_value = tensor_item(total_loss)
                    train_loss_sum += loss_value
                    train_steps += 1

                    if batch_idx % log_every == 0:
                        init_mse = torch.mean((batch.r_init.float() - batch.r_true.float()) ** 2)
                        pred_init_mse = torch.mean(
                            (outputs["pred_coords"].detach().float() - batch.r_init.float()) ** 2
                        )
                        logger.info(
                            "Epoch %d [%d/%d] | Loss %.8e | Struct %.8e | Noise %.8e | InitMSE %.8e | PredInitMSE %.8e | CoordScale %.8e | Phy %.8e | Bond %.8e | Angle %.8e | Dihedral %.8e | VdW %.8e | Electro %.8e | HBond %.8e | PiPi %.8e | PDB %s | no grad",
                            epoch + 1,
                            batch_idx,
                            len(train_loader),
                            loss_value,
                            tensor_item(loss_res.get("L_structure", 0.0)),
                            tensor_item(loss_res.get("L_noise", 0.0)),
                            tensor_item(init_mse),
                            tensor_item(pred_init_mse),
                            decoder_coord_scale_abs_max(model),
                            tensor_item(loss_res.get("L_phy_geo", 0.0)),
                            tensor_item(loss_res.get("L_bond", 0.0)),
                            tensor_item(loss_res.get("L_angle", 0.0)),
                            tensor_item(loss_res.get("L_dihedral", 0.0)),
                            tensor_item(loss_res.get("L_vdW", 0.0)),
                            tensor_item(loss_res.get("L_electro", 0.0)),
                            tensor_item(loss_res.get("L_hbond", 0.0)),
                            tensor_item(loss_res.get("L_pi_pi", 0.0)),
                            getattr(batch, "pdb_id", ["unknown"]),
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                stage = "backward"
                scaler.scale(total_loss).backward()
                stage = "unscale_grad"
                scaler.unscale_(optimizer)
                has_nonfinite_grad = False
                bad_grad_names = []
                for param in model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        has_nonfinite_grad = True
                        break
                if has_nonfinite_grad:
                    for name, param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            bad_grad_names.append(name)
                            if len(bad_grad_names) >= 8:
                                break
                if has_nonfinite_grad:
                    logger.warning(
                        "Non-finite gradient at epoch %d batch %d; skipping optimizer step. PDB %s | bad params: %s",
                        epoch + 1,
                        batch_idx,
                        getattr(batch, "pdb_id", ["unknown"]),
                        bad_grad_names,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                if grad_clip is not None:
                    stage = "grad_clip"
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                stage = "optimizer_step"
                scaler.step(optimizer)
                stage = "scaler_update"
                scaler.update()

                loss_value = tensor_item(total_loss)
                train_loss_sum += loss_value
                train_steps += 1

                if batch_idx % log_every == 0:
                    init_mse = torch.mean((batch.r_init.float() - batch.r_true.float()) ** 2)
                    pred_init_mse = torch.mean(
                        (outputs["pred_coords"].detach().float() - batch.r_init.float()) ** 2
                    )
                    logger.info(
                        "Epoch %d [%d/%d] | Loss %.8e | Struct %.8e | Noise %.8e | InitMSE %.8e | PredInitMSE %.8e | CoordScale %.8e | Phy %.8e | Bond %.8e | Angle %.8e | Dihedral %.8e | VdW %.8e | Electro %.8e | HBond %.8e | PiPi %.8e | PDB %s",
                        epoch + 1,
                        batch_idx,
                        len(train_loader),
                        loss_value,
                        tensor_item(loss_res.get("L_structure", 0.0)),
                        tensor_item(loss_res.get("L_noise", 0.0)),
                        tensor_item(init_mse),
                        tensor_item(pred_init_mse),
                        decoder_coord_scale_abs_max(model),
                        tensor_item(loss_res.get("L_phy_geo", 0.0)),
                        tensor_item(loss_res.get("L_bond", 0.0)),
                        tensor_item(loss_res.get("L_angle", 0.0)),
                        tensor_item(loss_res.get("L_dihedral", 0.0)),
                        tensor_item(loss_res.get("L_vdW", 0.0)),
                        tensor_item(loss_res.get("L_electro", 0.0)),
                        tensor_item(loss_res.get("L_hbond", 0.0)),
                        tensor_item(loss_res.get("L_pi_pi", 0.0)),
                        getattr(batch, "pdb_id", ["unknown"]),
                    )

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "OOM at epoch %d batch %d during %s; skipping batch. PDB %s",
                        epoch + 1,
                        batch_idx,
                        stage,
                        getattr(batch, "pdb_id", ["unknown"]),
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise

        avg_train = train_loss_sum / train_steps if train_steps else 0.0
        logger.info("Epoch %d train average loss: %.8e", epoch + 1, avg_train)

        avg_val = None
        if (epoch + 1) % int(train_cfg["validate_every_n_epochs"]) == 0:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_idx, val_batch in enumerate(val_loader):
                    if max_val_batches is not None and val_idx >= int(max_val_batches):
                        break
                    if val_batch is None:
                        continue

                    val_batch = prepare_batch(val_batch, device, train_cfg, config)
                    with autocast(enabled=use_amp):
                        val_outputs = model(val_batch, use_diffusion_refinement=use_diffusion_refinement)

                    val_outputs["true_coords"] = val_batch.r_true
                    val_loss = criterion(val_outputs, val_batch)["total_loss"]

                    val_loss_sum += tensor_item(val_loss)
                    val_steps += 1

            avg_val = val_loss_sum / val_steps if val_steps else 0.0
            logger.info("Epoch %d validation average loss: %.8e", epoch + 1, avg_val)

        scheduler.step()

        checkpoint_every = int(train_cfg["checkpoint_every_n_epochs"])
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_dir = resolve_project_path(base_dir, train_cfg["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            metric_name = "valloss" if avg_val is not None else "trainloss"
            metric_value = avg_val if avg_val is not None else avg_train
            metric_tag = checkpoint_metric_tag(metric_name, metric_value)
            ckpt_path = ckpt_dir / f"model_{checkpoint_run_id}_epoch_{epoch + 1:03d}_{metric_tag}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    train()
