"""Reusable PyTorch Lightning callbacks."""
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TrainLossEpochCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.epoch_train_losses: list[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.epoch_train_losses.append(float(v))
