"""Mamba backbone and classifier tower (pure torch.nn)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

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
