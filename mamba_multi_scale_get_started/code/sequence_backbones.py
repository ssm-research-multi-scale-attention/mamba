"""Sequence classifiers over token ids: Mamba2 stack, LSTM, GRU (+ mean pool + linear head)."""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2


def masked_token_mean_pool(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    padding_idx: int | None,
) -> torch.Tensor:
    """
    Mean over time for non-padding positions only.

    ``hidden``: (B, L, D). ``input_ids``: (B, L) token ids; positions equal to ``padding_idx``
    are excluded (must match ``nn.Embedding(..., padding_idx=...)``).
    """
    if padding_idx is None:
        return hidden.mean(dim=1)
    valid = (input_ids != padding_idx).to(dtype=hidden.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (hidden * valid).sum(dim=1) / denom


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, headdim: int, **mamba_kwargs):
        super().__init__()
        timescale_init = str(mamba_kwargs.pop("timescale_init", "single")).strip().lower()
        timescale_splits = mamba_kwargs.pop("timescale_splits", None)
        self.block = Mamba2(d_model=d_model, headdim=headdim, **mamba_kwargs)
        self.timescale_groups: list[dict[str, Any]] = []
        self._maybe_apply_split_timescale_init(timescale_init, timescale_splits)

    def forward(self, x):
        return x + self.block(x)

    @staticmethod
    def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
        # Stable inverse softplus: y = x + log(-expm1(-x)).
        return x + torch.log(-torch.expm1(-x))

    @staticmethod
    def _sample_log_uniform(
        n: int, lo: float, hi: float, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        u = torch.rand((n,), device=device, dtype=dtype)
        return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))

    def _validate_and_normalize_splits(
        self, splits_raw: Any, nheads: int
    ) -> tuple[list[dict[str, Any]], list[int]]:
        if not isinstance(splits_raw, list) or len(splits_raw) == 0:
            raise ValueError("timescale_splits must be a non-empty list when timescale_init='split_heads'.")
        names_seen: set[str] = set()
        total_fraction = 0.0
        normalized: list[dict[str, Any]] = []
        for idx, split in enumerate(splits_raw):
            if not isinstance(split, dict):
                raise ValueError(f"timescale_splits[{idx}] must be a mapping.")
            name = str(split.get("name", f"group_{idx}")).strip()
            if not name:
                raise ValueError(f"timescale_splits[{idx}].name must be non-empty.")
            if name in names_seen:
                raise ValueError(f"Duplicate timescale_splits name: {name!r}.")
            names_seen.add(name)
            if "fraction" not in split:
                raise ValueError(f"timescale_splits[{idx}].fraction is required.")
            fraction = float(split["fraction"])
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(f"timescale_splits[{idx}].fraction must be in [0, 1], got {fraction}.")
            total_fraction += fraction
            if "A_init_range" not in split:
                raise ValueError(f"timescale_splits[{idx}].A_init_range is required.")
            a_init_range = split["A_init_range"]
            if not isinstance(a_init_range, (list, tuple)) or len(a_init_range) != 2:
                raise ValueError(f"timescale_splits[{idx}].A_init_range must be [min, max].")
            a_lo = float(a_init_range[0])
            a_hi = float(a_init_range[1])
            if not (a_lo > 0.0 and a_hi > 0.0 and a_hi > a_lo):
                raise ValueError(
                    f"timescale_splits[{idx}].A_init_range must satisfy 0 < min < max, got {a_init_range!r}."
                )
            if "dt_min" not in split or "dt_max" not in split:
                raise ValueError(f"timescale_splits[{idx}] must contain dt_min and dt_max.")
            dt_min = float(split["dt_min"])
            dt_max = float(split["dt_max"])
            if not (dt_min > 0.0 and dt_max > dt_min):
                raise ValueError(
                    f"timescale_splits[{idx}] must satisfy 0 < dt_min < dt_max, got dt_min={dt_min}, dt_max={dt_max}."
                )
            normalized.append(
                {
                    "name": name,
                    "fraction": fraction,
                    "A_init_range": (a_lo, a_hi),
                    "dt_min": dt_min,
                    "dt_max": dt_max,
                }
            )

        if total_fraction <= 0.0:
            raise ValueError("timescale_splits fractions must sum to > 0.")
        if total_fraction > 1.0 + 1e-8:
            raise ValueError(f"timescale_splits fractions must sum to <= 1.0, got {total_fraction}.")

        counts: list[int] = []
        allocated = 0
        for split in normalized[:-1]:
            c = int(nheads * float(split["fraction"]))
            counts.append(c)
            allocated += c
        counts.append(max(0, nheads - allocated))
        return normalized, counts

    def _maybe_apply_split_timescale_init(self, timescale_init: str, timescale_splits: Any) -> None:
        if timescale_init == "single":
            return
        if timescale_init != "split_heads":
            raise ValueError(
                f"Unsupported model.mamba.timescale_init={timescale_init!r}. Use 'single' or 'split_heads'."
            )
        if not hasattr(self.block, "A_log") or not hasattr(self.block, "dt_bias"):
            raise ValueError("Mamba2 block must expose A_log and dt_bias for split_heads timescale init.")
        nheads = int(self.block.A_log.numel())
        if int(self.block.dt_bias.numel()) != nheads:
            raise ValueError("Expected A_log and dt_bias to have matching [nheads] shape.")
        splits, counts = self._validate_and_normalize_splits(timescale_splits, nheads)
        device = self.block.A_log.device
        dtype = self.block.A_log.dtype
        offset = 0
        group_meta: list[dict[str, Any]] = []
        with torch.no_grad():
            for split, count in zip(splits, counts):
                start = offset
                end = start + count
                if count > 0:
                    a_lo, a_hi = split["A_init_range"]
                    a_vals = self._sample_log_uniform(count, a_lo, a_hi, device=device, dtype=dtype)
                    dt_vals = self._sample_log_uniform(
                        count, float(split["dt_min"]), float(split["dt_max"]), device=device, dtype=dtype
                    )
                    self.block.A_log[start:end].copy_(torch.log(a_vals))
                    self.block.dt_bias[start:end].copy_(self._inv_softplus(dt_vals))
                group_meta.append(
                    {
                        "name": split["name"],
                        "start": start,
                        "end": end,
                        "count": count,
                        "fraction": float(split["fraction"]),
                    }
                )
                offset = end
        self.timescale_groups = group_meta


class MambaHeadsClassifier(nn.Module):
    """Token Mamba tower + masked mean pool (non-padding tokens) + classifier."""

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
        pooled = masked_token_mean_pool(out, x, self.embed.padding_idx)
        return self.classifier(pooled)


class RnnClassifier(nn.Module):
    """Embedding → stacked LSTM or GRU → masked mean pool → LayerNorm + linear."""

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
        pooled = masked_token_mean_pool(out, x, self.embed.padding_idx)
        return self.classifier(pooled)


class MultiScaleMambaClassifier(nn.Module):
    """
    Shared embedding → fast Mamba over full length → slow Mamba over stridden length
    → masked mean pool per branch → concat, sum, or gated fusion → LayerNorm + linear.
    """

    def __init__(
        self,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        slow_layer_headdims: Sequence[int],
        vocab_size: int,
        num_classes: int,
        slow_stride: int = 4,
        fusion: str = "concat",
        **mamba_kwargs: Any,
    ):
        super().__init__()
        fusion_l = str(fusion).strip().lower()
        if fusion_l not in ("concat", "sum", "gated"):
            raise ValueError(f"fusion must be 'concat', 'sum', or 'gated', got {fusion!r}")
        self.slow_stride = int(slow_stride)
        self.fusion = fusion_l
        fusion_dim = 2 * d_model if fusion_l == "concat" else d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fast_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in fast_layer_headdims]
        )
        self.slow_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in slow_layer_headdims]
        )
        self.gate = (
            nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))
            if fusion_l == "gated"
            else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast_out = self.fast_seq(emb)
        fast_pooled = masked_token_mean_pool(fast_out, x, self.embed.padding_idx)

        slow_x = x[:, :: self.slow_stride]
        slow_emb = emb[:, :: self.slow_stride, :]
        slow_out = self.slow_seq(slow_emb)
        slow_pooled = masked_token_mean_pool(slow_out, slow_x, self.embed.padding_idx)

        if self.fusion == "concat":
            fused = torch.cat([fast_pooled, slow_pooled], dim=-1)
        elif self.fusion == "sum":
            fused = fast_pooled + slow_pooled
        else:
            gmod = self.gate
            if gmod is None:
                raise RuntimeError("gated fusion requires gate module")
            gate = torch.sigmoid(gmod(torch.cat([fast_pooled, slow_pooled], dim=-1)))
            fused = fast_pooled + gate * slow_pooled
        return self.classifier(fused)


class MultiScaleMambaAttentionClassifier(nn.Module):
    """
    Shared embedding → fast Mamba (full L) → slow Mamba (stridden L) → cross-attention
    (queries=fast, keys/values=slow) → sequence-level fusion → masked mean pool → classifier.
    """

    def __init__(
        self,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        slow_layer_headdims: Sequence[int],
        vocab_size: int,
        num_classes: int,
        slow_stride: int = 4,
        attention_heads: int = 4,
        fusion: str = "residual",
        dropout: float = 0.0,
        **mamba_kwargs: Any,
    ):
        super().__init__()
        ss = int(slow_stride)
        if ss < 1:
            raise ValueError(f"slow_stride must be >= 1, got {slow_stride!r}")
        self.slow_stride = ss
        fusion_l = str(fusion).strip().lower()
        if fusion_l not in ("residual", "concat", "gated"):
            raise ValueError(f"fusion must be 'residual', 'concat', or 'gated', got {fusion!r}")
        self.fusion = fusion_l
        ah = int(attention_heads)
        if d_model % ah != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by attention_heads ({ah})")

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fast_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in fast_layer_headdims]
        )
        self.slow_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in slow_layer_headdims]
        )
        self.fast_norm = nn.LayerNorm(d_model)
        self.slow_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=ah, batch_first=True
        )
        self.dropout = nn.Dropout(float(dropout))
        self.fusion_proj = nn.Linear(2 * d_model, d_model) if fusion_l == "concat" else None
        self.gate = (
            nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))
            if fusion_l == "gated"
            else None
        )
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast_out = self.fast_seq(emb)

        slow_x = x[:, :: self.slow_stride]
        slow_emb = emb[:, :: self.slow_stride, :]
        slow_out = self.slow_seq(slow_emb)

        fast_normed = self.fast_norm(fast_out)
        slow_normed = self.slow_norm(slow_out)
        slow_key_padding_mask = slow_x == self.embed.padding_idx

        attn_out, _ = self.cross_attn(
            query=fast_normed,
            key=slow_normed,
            value=slow_normed,
            key_padding_mask=slow_key_padding_mask,
            need_weights=False,
        )

        if self.fusion == "residual":
            fused_seq = fast_out + self.dropout(attn_out)
        elif self.fusion == "concat":
            proj = self.fusion_proj
            if proj is None:
                raise RuntimeError("concat fusion requires fusion_proj")
            fused_seq = proj(torch.cat([fast_out, self.dropout(attn_out)], dim=-1))
        else:
            gmod = self.gate
            if gmod is None:
                raise RuntimeError("gated fusion requires gate module")
            gate = torch.sigmoid(gmod(torch.cat([fast_out, attn_out], dim=-1)))
            fused_seq = fast_out + gate * self.dropout(attn_out)

        pooled = masked_token_mean_pool(fused_seq, x, self.embed.padding_idx)
        return self.classifier(pooled)


def build_sequence_backbone(cfg: DictConfig, vocab_size: int) -> nn.Module:
    """
    Dispatch on ``cfg.model.backbone`` (default ``mamba``): ``mamba`` | ``multiscale_mamba`` |
    ``multiscale_mamba_attention`` | ``lstm`` | ``gru``.

    - **mamba**: ``model.d_model``, ``model.layer_headdims``, ``model.mamba`` kwargs.
    - **multiscale_mamba**: ``model.multiscale`` (``slow_stride``, ``fusion``: concat|sum|gated,
      ``fast_layer_headdims``, ``slow_layer_headdims``); headdim lists default to ``model.layer_headdims`` when omitted.
    - **multiscale_mamba_attention**: same ``multiscale`` keys plus ``attention_heads``, ``dropout``;
      ``fusion``: residual|concat|gated; fast headdims default to ``model.layer_headdims``; slow defaults to ``[32, 32]``;
      ``attention_heads`` default 4; ``fusion`` default ``residual``; ``dropout`` default ``0.0``.
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
    if kind == "multiscale_mamba":
        mamba_kw = OmegaConf.to_container(cfg.model.mamba, resolve=True)  # type: ignore[assignment]
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        slow_hd = OmegaConf.select(cfg, "model.multiscale.slow_layer_headdims", default=None)
        fallback_raw = OmegaConf.select(cfg, "model.layer_headdims", default=None)
        if fast_hd is None or slow_hd is None:
            if fallback_raw is None:
                raise ValueError(
                    "multiscale_mamba: specify model.multiscale.fast_layer_headdims and "
                    "slow_layer_headdims, or set model.layer_headdims as fallback."
                )
            fallback_hd = [int(h) for h in list(fallback_raw)]
        else:
            fallback_hd = []
        fast_layer_headdims = (
            [int(h) for h in list(fast_hd)] if fast_hd is not None else list(fallback_hd)
        )
        slow_layer_headdims = (
            [int(h) for h in list(slow_hd)] if slow_hd is not None else list(fallback_hd)
        )
        return MultiScaleMambaClassifier(
            d_model=int(cfg.model.d_model),
            fast_layer_headdims=fast_layer_headdims,
            slow_layer_headdims=slow_layer_headdims,
            vocab_size=vocab_size,
            num_classes=int(cfg.model.num_classes),
            slow_stride=int(OmegaConf.select(cfg, "model.multiscale.slow_stride", default=4)),
            fusion=str(OmegaConf.select(cfg, "model.multiscale.fusion", default="concat")),
            **mamba_kw,
        )
    if kind == "multiscale_mamba_attention":
        mamba_kw = OmegaConf.to_container(cfg.model.mamba, resolve=True)  # type: ignore[assignment]
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        if fast_hd is not None:
            fast_layer_headdims = [int(h) for h in list(fast_hd)]
        else:
            fb = OmegaConf.select(cfg, "model.layer_headdims", default=None)
            if fb is None:
                raise ValueError(
                    "multiscale_mamba_attention: set model.multiscale.fast_layer_headdims "
                    "or model.layer_headdims."
                )
            fast_layer_headdims = [int(h) for h in list(fb)]
        slow_hd = OmegaConf.select(cfg, "model.multiscale.slow_layer_headdims", default=None)
        if slow_hd is not None:
            slow_layer_headdims = [int(h) for h in list(slow_hd)]
        else:
            slow_layer_headdims = [32, 32]
        return MultiScaleMambaAttentionClassifier(
            d_model=int(cfg.model.d_model),
            fast_layer_headdims=fast_layer_headdims,
            slow_layer_headdims=slow_layer_headdims,
            vocab_size=vocab_size,
            num_classes=int(cfg.model.num_classes),
            slow_stride=int(OmegaConf.select(cfg, "model.multiscale.slow_stride", default=4)),
            attention_heads=int(OmegaConf.select(cfg, "model.multiscale.attention_heads", default=4)),
            fusion=str(OmegaConf.select(cfg, "model.multiscale.fusion", default="residual")),
            dropout=float(OmegaConf.select(cfg, "model.multiscale.dropout", default=0.0)),
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
        f"Unknown model.backbone={kind!r}. Supported: mamba, multiscale_mamba, "
        "multiscale_mamba_attention, lstm, gru. "
        "Extend sequence_backbones.build_sequence_backbone for new architectures."
    )
