"""Persist run metrics to CSV under the experiment output directory."""
from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


def _to_scalar(v: Any) -> float | int | str | bool:
    if v is None:
        return ""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, np.generic):
        return float(v)
    if hasattr(v, "detach"):
        return float(v.detach().cpu().item())
    if hasattr(v, "item"):
        return float(v.item())
    return str(v)


def save_metrics_csv(
    output_dir: str,
    *,
    cfg: DictConfig,
    train_losses: list[float],
    callback_metrics: dict[str, Any],
    total_params: int | None = None,
    trainable_params: int | None = None,
    confusion_matrix: np.ndarray | None = None,
    run_timings: dict[str, Any] | None = None,
    filename: str = "metrics.csv",
) -> str:
    """
    Write ``metric,value`` rows: experiment metadata, per-epoch train loss, Lightning
    ``callback_metrics`` after ``trainer.test()`` (e.g. test/accuracy), optional CM cells,
    and optional ``run_timings`` (e.g. seconds_train / seconds_test for GPU wall comparison).
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    rows: list[tuple[str, Any]] = []

    rows.append(("experiment_name", str(cfg.experiment.name)))
    rows.append(("seed", int(cfg.experiment.seed)))
    rows.append(("model_lit_class", str(cfg.model.lit_class)))
    rows.append(("model_backbone", str(OmegaConf.select(cfg, "model.backbone", default="mamba"))))
    rows.append(("train_epochs", int(cfg.train.epochs)))

    if run_timings:
        for key in sorted(run_timings.keys()):
            rows.append((str(key), _to_scalar(run_timings[key])))

    if total_params is not None:
        rows.append(("model_total_params", int(total_params)))
    if trainable_params is not None:
        rows.append(("model_trainable_params", int(trainable_params)))

    for i, loss in enumerate(train_losses, start=1):
        rows.append((f"train_loss_epoch_{i}", float(loss)))

    for key in sorted(callback_metrics.keys()):
        val = callback_metrics[key]
        if val is None:
            continue
        rows.append((str(key), _to_scalar(val)))

    if confusion_matrix is not None:
        cm = np.asarray(confusion_matrix)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                rows.append((f"confusion_matrix_{i}_{j}", int(cm[i, j])))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for name, val in rows:
            w.writerow([name, val])

    print(f"Saved: {path}")
    return path
