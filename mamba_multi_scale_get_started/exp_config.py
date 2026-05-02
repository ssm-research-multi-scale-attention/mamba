"""OmegaConf loading, imports, seeds, Lightning accelerator mapping."""
from __future__ import annotations

import importlib
import os
import random
from typing import Any, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def package_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: str, cli_overrides: List[str]) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(cli_overrides))
    OmegaConf.resolve(cfg)
    return cfg


def import_class(path: str) -> Any:
    """
    Resolve a dotted class path such as ``mamba_heads_lit.MambaHeadsLitModule``.

    Leading/trailing whitespace is stripped.
    """
    path = path.strip()
    if not path:
        raise ValueError("Empty model.lit_class path")
    mod_name, sep, cls_name = path.rpartition(".")
    if not sep:
        raise ValueError(f"Need 'package.module.ClassName', got: {path}")
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, cls_name)
    except AttributeError as e:
        raise ImportError(f"Module {mod_name} has no class {cls_name}") from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_devices(devices_cfg: Any):
    """Return int device count or list of IDs for Lightning Trainer."""
    if devices_cfg is None:
        return 1
    cont = OmegaConf.to_container(devices_cfg, resolve=True) if OmegaConf.is_config(devices_cfg) else devices_cfg
    if isinstance(cont, (list, tuple)):
        return list(cont)
    return int(cont)


def trainer_accelerator_and_devices(device_cfg: str, devices_cfg: Any) -> tuple[str, Any]:
    d = str(device_cfg).strip().lower()
    devices = _normalize_devices(devices_cfg)
    if d == "cpu":
        return "cpu", devices
    if d in ("cuda", "gpu"):
        return "gpu", devices
    return "auto", devices


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def lit_init_extra_from_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Optional static kwargs from ``cfg.model.lit_init`` (merged by the experiment runner)."""
    extra_raw = cfg.model.get("lit_init")
    if extra_raw is None:
        return {}
    extra = OmegaConf.to_container(extra_raw, resolve=True) or {}
    if not isinstance(extra, dict):
        raise TypeError(f"cfg.model.lit_init must resolve to a dict, got {type(extra)}")
    return extra
