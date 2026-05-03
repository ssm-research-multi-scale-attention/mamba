"""Single Lightning module: HF ``(input_ids, labels)`` batches + pluggable sequence backbone."""
from __future__ import annotations

from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sequence_backbones import build_sequence_backbone


class SequenceClassifierLitModule(pl.LightningModule):
    """Backbone from ``cfg.model.backbone`` (mamba | multiscale_mamba | multiscale_mamba_attention | lstm | gru); see ``sequence_backbones.build_sequence_backbone``."""

    def __init__(self, cfg: DictConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_sequence_backbone(cfg, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self._test_preds: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._last_cm: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _tb_experiment(self):
        logger = self.logger
        if logger is None:
            return None
        return getattr(logger, "experiment", None)

    def on_fit_start(self) -> None:
        exp = self._tb_experiment()
        if exp is None:
            return
        step = int(self.trainer.global_step) if self.trainer is not None else 0
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = str(OmegaConf.select(self.cfg, "model.backbone", default="mamba"))
        try:
            exp.add_scalar("model/total_params", float(total), step)
            exp.add_scalar("model/trainable_params", float(trainable), step)
            exp.add_text("model/backbone", backbone, step)
            exp.add_text("config/resolved_yaml", OmegaConf.to_yaml(self.cfg), step)
        except Exception:
            pass

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        preds = logits.detach().argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log(
            "train/acc_batch",
            acc,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=x.size(0),
        )
        return loss

    def on_test_start(self) -> None:
        self._test_preds.clear()
        self._test_targets.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        preds = logits.argmax(dim=-1)
        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        preds = torch.cat(self._test_preds).numpy()
        targets = torch.cat(self._test_targets).numpy()
        acc = accuracy_score(targets, preds)
        report = classification_report(targets, preds, digits=4)
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)
        self._last_cm = confusion_matrix(targets, preds)

        self.log("test/accuracy", float(acc))
        try:
            f1w = float(f1_score(targets, preds, average="weighted", zero_division=0))
            f1m = float(f1_score(targets, preds, average="macro", zero_division=0))
            self.log("test/f1_weighted", f1w)
            self.log("test/f1_macro", f1m)
        except Exception:
            pass

        exp = self._tb_experiment()
        if exp is not None and self.trainer is not None:
            step = int(self.trainer.global_step)
            try:
                exp.add_text("test/classification_report", report, step)
            except Exception:
                pass

    def configure_optimizers(self):
        cfg = self.cfg
        params = self.parameters()
        lr = float(cfg.train.lr)
        wd = float(cfg.train.weight_decay)
        name = str(cfg.train.optimizer).lower()
        if name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")
