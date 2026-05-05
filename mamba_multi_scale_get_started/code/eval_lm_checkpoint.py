#!/usr/bin/env python3
"""Evaluate a trained LM checkpoint on a configured char-level dataset."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.tiny_shakespeare import CharVocab, build_char_lm_dataset
from exp_config import load_config
from models.language_models import build_lm

PROJECT_ROOT = _CODE.parent


def lm_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def _resolve_device(raw: str) -> torch.device:
    s = str(raw).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total_n = 0
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = lm_token_cross_entropy(logits, y)
        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.argument("overrides", nargs=-1)
def main(config_path: str, overrides: tuple[str, ...]) -> None:
    cfg = load_config(config_path, list(overrides))
    checkpoint_path = Path(str(cfg.eval.checkpoint_path)).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = (PROJECT_ROOT / str(cfg.logging.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path_cfg = OmegaConf.select(cfg, "eval.vocab_path", default=None)
    vocab = None
    if vocab_path_cfg:
        vocab_path = Path(str(vocab_path_cfg)).resolve()
        vocab = CharVocab.from_json(json.loads(vocab_path.read_text(encoding="utf-8")))
    else:
        default_vocab_path = checkpoint_path.parent / "vocab.json"
        if default_vocab_path.is_file():
            vocab = CharVocab.from_json(json.loads(default_vocab_path.read_text(encoding="utf-8")))

    dataset = str(OmegaConf.select(cfg, "data.dataset", default="tiny_shakespeare")).strip().lower()
    data_dir = PROJECT_ROOT / str(cfg.data.data_dir)
    sampling = str(OmegaConf.select(cfg, "data.sampling", default="sequential")).strip().lower()
    steps_per_epoch = int(OmegaConf.select(cfg, "data.steps_per_epoch", default=1000))

    train_loader, val_loader, data_vocab = build_char_lm_dataset(
        dataset=dataset,
        data_dir=data_dir,
        block_size=int(cfg.data.block_size),
        batch_size=int(cfg.loader.batch_size),
        train_ratio=float(cfg.data.train_ratio),
        num_workers=int(cfg.loader.num_workers),
        vocab=vocab,
        sampling=sampling,
        steps_per_epoch=steps_per_epoch,
        seed=int(cfg.experiment.seed),
    )
    vocab = vocab or data_vocab
    cfg.model.vocab_size = len(vocab)

    device = _resolve_device(str(OmegaConf.select(cfg, "device", default="auto")))
    model = build_lm(cfg).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    evaluate_train = bool(OmegaConf.select(cfg, "eval.evaluate_train", default=True))
    rows: list[dict[str, float | str]] = []
    if evaluate_train:
        train_loss = _eval_epoch(model, train_loader, device)
        rows.append(
            {
                "split": "train",
                "loss": train_loss,
                "perplexity": math.exp(train_loss),
                "bpc": train_loss / math.log(2.0),
            }
        )
    else:
        train_loss = float("nan")
    val_loss = _eval_epoch(model, val_loader, device)
    rows.append({"split": "val", "loss": val_loss, "perplexity": math.exp(val_loss), "bpc": val_loss / math.log(2.0)})

    metrics_path = out_dir / "checkpoint_eval_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "loss", "perplexity", "bpc"])
        w.writeheader()
        w.writerows(rows)

    print(f"checkpoint={checkpoint_path}")
    print(f"dataset={dataset}")
    if evaluate_train:
        print(f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    else:
        print(f"val_loss={val_loss:.6f}")
    print(f"metrics_csv={metrics_path}")


if __name__ == "__main__":
    main()
