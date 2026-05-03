"""Sequence classifiers over token ids: Mamba2 stack, LSTM, GRU (+ mean pool + linear head)."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, headdim: int, **mamba_kwargs):
        super().__init__()
        self.block = Mamba2(d_model=d_model, headdim=headdim, **mamba_kwargs)

    def forward(self, x):
        return x + self.block(x)


class MambaHeadsClassifier(nn.Module):
    """Token Mamba tower + mean pool + classifier."""

    def __init__(
        self,
        d_model: int,
        layer_headdims: Sequence[int],
        vocab_size: int,
        num_classes: int,
        **mamba_kwargs,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in layer_headdims]
        )
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out = self.seq(emb)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


class RnnClassifier(nn.Module):
    """Embedding → stacked LSTM or GRU → mean pool → LayerNorm + linear."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        *,
        cell: str,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        cell_l = str(cell).strip().lower()
        if cell_l not in ("lstm", "gru"):
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_kw = dict(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        if cell_l == "lstm":
            self.rnn = nn.LSTM(**rnn_kw)
        else:
            self.rnn = nn.GRU(**rnn_kw)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


def build_sequence_backbone(cfg: DictConfig, vocab_size: int) -> nn.Module:
    """
    Dispatch on ``cfg.model.backbone`` (default ``mamba``): ``mamba`` | ``lstm`` | ``gru``.

    - **mamba**: ``model.d_model``, ``model.layer_headdims``, ``model.mamba`` kwargs.
    - **lstm** / **gru**: ``model.embed_dim`` (else ``model.d_model``), ``model.rnn`` (hidden_size, num_layers, …).
    """
    kind = str(OmegaConf.select(cfg, "model.backbone", default="mamba")).strip().lower()
    if kind == "mamba":
        layer_headdims = [int(h) for h in list(cfg.model.layer_headdims)]
        mamba_kw: dict[str, Any] = OmegaConf.to_container(cfg.model.mamba, resolve=True)  # type: ignore[assignment]
        return MambaHeadsClassifier(
            d_model=int(cfg.model.d_model),
            layer_headdims=layer_headdims,
            vocab_size=vocab_size,
            num_classes=int(cfg.model.num_classes),
            **mamba_kw,
        )
    if kind in ("lstm", "gru"):
        rnn_kw: dict[str, Any] = OmegaConf.to_container(cfg.model.rnn, resolve=True)  # type: ignore[assignment]
        embed_dim = int(OmegaConf.select(cfg, "model.embed_dim", default=cfg.model.d_model))
        return RnnClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=int(cfg.model.num_classes),
            cell=kind,
            **rnn_kw,
        )
    raise ValueError(
        f"Unknown model.backbone={kind!r}. Supported: mamba, lstm, gru. "
        "Extend sequence_backbones.build_sequence_backbone for new architectures."
    )
