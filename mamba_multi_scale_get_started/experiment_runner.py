"""Run a training job from YAML: dataloaders + any Lightning module + Trainer."""
from __future__ import annotations

import inspect
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
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
from plotting import save_training_artifacts


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
    seed = int(cfg.experiment.seed)
    pl.seed_everything(seed, workers=True)
    set_seed(seed)

    device_hint = resolve_device(str(cfg.device))
    print("Resolved config:\n", OmegaConf.to_yaml(cfg))
    print(f"Device hint (data loaders): {device_hint}")

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

    loss_tracker = TrainLossEpochCallback()
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=accel,
        devices=devices_arg,
        precision=precision,  # type: ignore[arg-type]
        callbacks=[loss_tracker],
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(lit, train_dataloaders=train_loader)
    trainer.test(lit, dataloaders=eval_loader)

    cm = getattr(lit, "_last_cm", None)
    strict_cm = getattr(cfg.logging, "require_confusion_matrix", False)
    if strict_cm and cm is None:
        raise RuntimeError("logging.require_confusion_matrix=true but module set no _last_cm after test.")

    if cfg.logging.save_plots:
        out_dir = str(cfg.logging.output_dir)
        save_training_artifacts(out_dir, loss_tracker.epoch_train_losses, cm)
