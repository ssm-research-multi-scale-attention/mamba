"""Lightning module for Mamba multi-head text classification."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mamba_heads_models import MambaHeadsClassifier


def mamba_kwargs_from_cfg(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.model.mamba, resolve=True)  # type: ignore[return-value]


class MambaHeadsLitModule(pl.LightningModule):
    """Example module: expects HuggingFace-style ``(input_ids, labels)`` batches."""

    def __init__(self, cfg: DictConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        layer_headdims = [int(h) for h in list(cfg.model.layer_headdims)]
        mamba_kw = mamba_kwargs_from_cfg(cfg)
        self.backbone = MambaHeadsClassifier(
            d_model=int(cfg.model.d_model),
            layer_headdims=layer_headdims,
            vocab_size=vocab_size,
            num_classes=int(cfg.model.num_classes),
            **mamba_kw,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self._test_preds: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._last_cm: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

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
        return loss

    def on_test_start(self) -> None:
        self._test_preds.clear()
        self._test_targets.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
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
