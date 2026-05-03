"""Run a training job from YAML: dataloaders + any Lightning module + Trainer."""
from __future__ import annotations

import inspect
import os
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from exp_config import (
    import_class,
    lit_init_extra_from_cfg,
    resolve_device,
    set_seed,
    trainer_accelerator_and_devices,
)
from hf_text_dataloaders import build_text_classification_dataloaders
from lightning_callbacks import TrainLossEpochCallback
from metrics_export import save_metrics_csv
from run_timing import collect_run_timing_metadata, timed_phase
from plotting import save_training_artifacts

RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"


def _plotly_include_js(cfg: DictConfig) -> str | bool:
    """``cdn`` (smaller HTML) or ``inline`` / true (embedded JS, better offline in browser)."""
    raw = str(OmegaConf.select(cfg, "logging.plotly_include_js", default="cdn")).strip().lower()
    if raw in ("inline", "embed", "true", "1", "yes", "on"):
        return True
    return "cdn"


def _apply_torch_train_env(cfg: DictConfig) -> None:
    """Global torch training knobs (see ``train.float32_matmul_precision`` in config)."""
    raw = OmegaConf.select(cfg, "train.float32_matmul_precision", default="high")
    if raw is None or str(raw).strip().lower() in ("none", "null", "false", ""):
        return
    torch.set_float32_matmul_precision(str(raw))


def dump_resolved_config(cfg: DictConfig, output_dir: str, filename: str = RESOLVED_CONFIG_FILENAME) -> str:
    """Write fully resolved OmegaConf to YAML under ``output_dir``; return path written."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    OmegaConf.save(cfg, path)
    return path


def _tensorboard_enabled(cfg: DictConfig) -> bool:
    raw = OmegaConf.select(cfg, "logging.tensorboard", default=True)
    if raw is None or str(raw).strip().lower() in ("false", "0", "no", "off"):
        return False
    return bool(raw)


def _build_tensorboard_logger(cfg: DictConfig, output_dir: str) -> TensorBoardLogger:
    """Events live under ``output_dir / tensorboard_subdir / version_*``."""
    sub = str(OmegaConf.select(cfg, "logging.tensorboard_subdir", default="tensorboard"))
    return TensorBoardLogger(
        save_dir=output_dir,
        name=sub,
        default_hp_metric=False,
    )


def _tokenizer_from_cfg(cfg: DictConfig):
    tok_id = cfg.data.get("tokenizer")
    if tok_id is None or str(tok_id).lower() in ("none", "null"):
        raise ValueError("cfg.data.tokenizer is required for hf_text_dataloaders")
    return AutoTokenizer.from_pretrained(str(tok_id))


def _infer_lit_kw(cfg: DictConfig, tokenizer: Any | None, lit_cls: type) -> dict[str, Any]:
    """Pass tokenizer / vocab_size into ``__init__`` when the Lightning module accepts them."""
    inferred: dict[str, Any] = {}
    sig = inspect.signature(lit_cls.__init__)
    params = sig.parameters
    if "tokenizer" in params and tokenizer is not None:
        inferred["tokenizer"] = tokenizer
    if "vocab_size" in params and tokenizer is not None:
        inferred["vocab_size"] = tokenizer.vocab_size
    extra = lit_init_extra_from_cfg(cfg)
    # Inferred tokenizer-derived args win over static lit_init duplicates
    return {**extra, **inferred}


def build_lightning_module(cfg: DictConfig, tokenizer: Any | None):
    cls = import_class(str(cfg.model.lit_class))
    kw = _infer_lit_kw(cfg, tokenizer, cls)
    try:
        return cls(cfg, **kw)
    except TypeError as e:
        sig = getattr(cls.__init__, "__signature__", None) or inspect.signature(cls.__init__)
        raise TypeError(f"{cls.__name__}{sig} incompatible with inferred kwargs keys={list(kw)!r}") from e


def run_experiment(cfg: DictConfig) -> None:
    _apply_torch_train_env(cfg)
    seed = int(cfg.experiment.seed)
    pl.seed_everything(seed, workers=True)
    set_seed(seed)

    device_hint = resolve_device(str(cfg.device))
    print("Resolved config:\n", OmegaConf.to_yaml(cfg))
    print(f"Device hint (data loaders): {device_hint}")

    out_dir = str(cfg.logging.output_dir)
    snap_path = dump_resolved_config(cfg, out_dir)
    print(f"Saved resolved config snapshot: {snap_path}")

    print("Loading tokenizer and HF dataset loaders...")
    tokenizer = _tokenizer_from_cfg(cfg)
    train_loader, eval_loader = build_text_classification_dataloaders(cfg, tokenizer, device_hint)

    print(f"Building Lightning module: {cfg.model.lit_class} ...")
    lit = build_lightning_module(cfg, tokenizer)
    total_params = sum(p.numel() for p in lit.parameters())
    trainable_params = sum(p.numel() for p in lit.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    accel, devices_arg = trainer_accelerator_and_devices(
        str(cfg.device),
        cfg.train.get("devices", 1),
    )
    pr = cfg.train.get("precision", 32)
    if isinstance(pr, (int, float)) and not isinstance(pr, bool):
        precision: int | str = int(pr)
    else:
        precision = str(pr)

    use_tb = _tensorboard_enabled(cfg)
    callbacks: list[Callback] = [TrainLossEpochCallback()]
    logger: bool | TensorBoardLogger = False
    if use_tb:
        tb = _build_tensorboard_logger(cfg, out_dir)
        logger = tb
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        print(f"TensorBoard: logdir={tb.log_dir}")

    log_every = int(OmegaConf.select(cfg, "logging.tensorboard_log_every_n_steps", default=50))
    if log_every < 1:
        log_every = 50

    loss_tracker = callbacks[0]
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=accel,
        devices=devices_arg,
        precision=precision,  # type: ignore[arg-type]
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every if use_tb else 50,
        enable_checkpointing=False,
    )

    _, train_sec, train_time_meth = timed_phase(
        lambda: trainer.fit(lit, train_dataloaders=train_loader),
    )
    _, test_sec, test_time_meth = timed_phase(
        lambda: trainer.test(lit, dataloaders=eval_loader),
    )

    cm = getattr(lit, "_last_cm", None)
    strict_cm = getattr(cfg.logging, "require_confusion_matrix", False)
    if strict_cm and cm is None:
        raise RuntimeError("logging.require_confusion_matrix=true but module set no _last_cm after test.")

    if OmegaConf.select(cfg, "logging.save_metrics", default=True):
        run_timings: dict[str, Any] = {
            "seconds_train": train_sec,
            "seconds_test": test_sec,
            "seconds_train_plus_test": train_sec + test_sec,
            "train_timing_method": train_time_meth,
            "test_timing_method": test_time_meth,
        }
        run_timings.update(collect_run_timing_metadata())
        save_metrics_csv(
            out_dir,
            cfg=cfg,
            train_losses=loss_tracker.epoch_train_losses,
            callback_metrics=dict(trainer.callback_metrics),
            total_params=total_params,
            trainable_params=trainable_params,
            confusion_matrix=cm,
            run_timings=run_timings,
        )

    if cfg.logging.save_plots:
        save_training_artifacts(
            out_dir,
            loss_tracker.epoch_train_losses,
            cm,
            include_plotlyjs=_plotly_include_js(cfg),
        )
