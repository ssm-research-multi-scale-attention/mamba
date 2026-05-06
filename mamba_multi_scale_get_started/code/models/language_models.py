"""Character-level language models: LSTM, Mamba2, multi-scale Mamba2."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_backbones import MambaLayer


class LstmLanguageModel(nn.Module):
    """Embedding → LSTM → logits over vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            d_model,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.lm_head(out)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class Mamba2LanguageModel(nn.Module):
    """Embedding → Mamba2 residual stack → LayerNorm → LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        layer_headdims: Sequence[int],
        **mamba_kwargs: Any,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in layer_headdims]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.embed(x))
        return self.lm_head(self.out_norm(h))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class MultiScaleMamba2LanguageModel(nn.Module):
    """Fast Mamba on full sequence + slow Mamba on stridden sequence, upsample, fuse, LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        slow_layer_headdims: Sequence[int],
        stride: int = 4,
        fusion: str = "sum",
        **mamba_kwargs: Any,
    ):
        super().__init__()
        fusion_l = str(fusion).strip().lower()
        if fusion_l not in ("sum", "concat", "gated"):
            raise ValueError(f"fusion must be sum, concat, or gated, got {fusion!r}")
        st = int(stride)
        if st < 1:
            raise ValueError(f"stride must be >= 1, got {stride!r}")
        self.stride = st
        self.fusion = fusion_l

        self.embed = nn.Embedding(vocab_size, d_model)
        self.fast_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in fast_layer_headdims]
        )
        self.slow_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in slow_layer_headdims]
        )
        self.fusion_proj = nn.Linear(2 * d_model, d_model) if fusion_l == "concat" else None
        self.gate = (
            nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))
            if fusion_l == "gated"
            else None
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast = self.fast_seq(emb)
        slow_emb = emb[:, :: self.stride, :]
        slow = self.slow_seq(slow_emb)
        L = emb.size(1)
        slow_up = slow.repeat_interleave(self.stride, dim=1)[:, :L, :]

        if self.fusion == "sum":
            fused = fast + slow_up
        elif self.fusion == "concat":
            proj = self.fusion_proj
            if proj is None:
                raise RuntimeError("concat fusion requires fusion_proj")
            fused = proj(torch.cat([fast, slow_up], dim=-1))
        else:
            gmod = self.gate
            if gmod is None:
                raise RuntimeError("gated fusion requires gate module")
            gate = torch.sigmoid(gmod(torch.cat([fast, slow_up], dim=-1)))
            fused = fast + gate * slow_up

        return self.lm_head(self.out_norm(fused))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class MultiScaleMamba2AttentionLanguageModel(nn.Module):
    """
    Fast Mamba (full ``L``) + slow Mamba (``L_slow = ceil(L / stride)``) + cross-attention
    (query=fast, key/value=slow) + sequence fusion → LayerNorm → LM head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        slow_layer_headdims: Sequence[int],
        stride: int = 4,
        attention_heads: int = 4,
        fusion: str = "residual",
        dropout: float = 0.0,
        **mamba_kwargs: Any,
    ):
        super().__init__()
        st = int(stride)
        if st < 1:
            raise ValueError(f"stride must be >= 1, got {stride!r}")
        self.stride = st
        fusion_l = str(fusion).strip().lower()
        if fusion_l not in ("residual", "concat", "gated"):
            raise ValueError(f"fusion must be residual, concat, or gated, got {fusion!r}")
        self.fusion = fusion_l
        ah = int(attention_heads)
        if d_model % ah != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by attention_heads ({ah})")

        self.embed = nn.Embedding(vocab_size, d_model)
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
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast_out = self.fast_seq(emb)
        slow_emb = emb[:, :: self.stride, :]
        slow_out = self.slow_seq(slow_emb)

        fast_normed = self.fast_norm(fast_out)
        slow_normed = self.slow_norm(slow_out)
        attn_out, _ = self.cross_attn(
            query=fast_normed,
            key=slow_normed,
            value=slow_normed,
            key_padding_mask=None,
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

        return self.lm_head(self.out_norm(fused_seq))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class MultiAxisMamba2LanguageModel(nn.Module):
    """
    Fast Mamba on full sequence + N stridden Mamba branches.

    Each branch output is upsampled back to fast length and fused with one of:
    - residual_gated_sum: fast + sum_i sigmoid(g_i([fast, slow_i])) * slow_i
    - residual_sum: fast + sum_i slow_i
    - concat_project: Linear([fast, slow_1, ..., slow_N])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        branches: Sequence[Mapping[str, Any]],
        fusion: str = "residual_gated_sum",
        gate_type: str = "vector",
        **mamba_kwargs: Any,
    ):
        super().__init__()
        if not branches:
            raise ValueError("multiscale.branches must be a non-empty list.")
        fusion_l = str(fusion).strip().lower()
        valid_fusions = {"residual_gated_sum", "residual_sum", "concat_project"}
        if fusion_l not in valid_fusions:
            raise ValueError(
                f"multiscale.fusion must be one of {sorted(valid_fusions)}, got {fusion!r}."
            )
        gate_type_l = str(gate_type).strip().lower()
        valid_gate_types = {"vector", "scalar"}
        if gate_type_l not in valid_gate_types:
            raise ValueError(
                f"multiscale.gate_type must be one of {sorted(valid_gate_types)}, got {gate_type!r}."
            )

        self.fusion = fusion_l
        self.gate_type = gate_type_l
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fast_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in fast_layer_headdims]
        )

        self.branch_names: list[str] = []
        self.branch_strides: list[int] = []
        self.slow_seqs = nn.ModuleList()
        self.gate_projs = nn.ModuleList()

        gate_out_dim = d_model if self.gate_type == "vector" else 1
        for branch in branches:
            name = str(branch["name"])
            stride = int(branch["stride"])
            layer_headdims = [int(h) for h in list(branch["layer_headdims"])]
            self.branch_names.append(name)
            self.branch_strides.append(stride)
            self.slow_seqs.append(
                nn.Sequential(
                    *[
                        MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs)
                        for h in layer_headdims
                    ]
                )
            )
            if self.fusion == "residual_gated_sum":
                self.gate_projs.append(nn.Linear(2 * d_model, gate_out_dim))

        self.concat_proj = (
            nn.Linear((1 + len(self.slow_seqs)) * d_model, d_model)
            if self.fusion == "concat_project"
            else None
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast = self.fast_seq(emb)
        seq_len = emb.size(1)

        slow_ups: list[torch.Tensor] = []
        for stride, slow_seq in zip(self.branch_strides, self.slow_seqs):
            slow_emb = emb[:, ::stride, :]
            slow = slow_seq(slow_emb)
            slow_up = slow.repeat_interleave(stride, dim=1)
            if slow_up.size(1) < seq_len:
                pad_len = seq_len - slow_up.size(1)
                slow_up = F.pad(slow_up, (0, 0, 0, pad_len))
            slow_ups.append(slow_up[:, :seq_len, :])

        if self.fusion == "residual_sum":
            fused = fast
            for slow_up in slow_ups:
                fused = fused + slow_up
        elif self.fusion == "concat_project":
            proj = self.concat_proj
            if proj is None:
                raise RuntimeError("concat_project fusion requires concat_proj")
            fused = proj(torch.cat([fast, *slow_ups], dim=-1))
        else:
            fused = fast
            for gate_proj, slow_up in zip(self.gate_projs, slow_ups):
                gate = torch.sigmoid(gate_proj(torch.cat([fast, slow_up], dim=-1)))
                fused = fused + gate * slow_up

        return self.lm_head(self.out_norm(fused))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build_lm(cfg) -> nn.Module:
    """Build LM from ``cfg.model`` (OmegaConf / dict-like)."""
    from omegaconf import OmegaConf

    backbone = str(OmegaConf.select(cfg, "model.backbone", default="mamba2_lm")).strip().lower()
    vocab = int(cfg.model.vocab_size)
    d_model = int(cfg.model.d_model)
    mamba_raw = OmegaConf.select(cfg, "model.mamba", default=None)
    mamba_kw: dict[str, Any] = (
        OmegaConf.to_container(mamba_raw, resolve=True) if mamba_raw is not None else {}
    )  # type: ignore[assignment]

    if backbone == "lstm_lm":
        hidden = int(OmegaConf.select(cfg, "model.hidden_size", default=d_model))
        nl = int(OmegaConf.select(cfg, "model.num_layers", default=2))
        do = float(OmegaConf.select(cfg, "model.dropout", default=0.0))
        return LstmLanguageModel(vocab, d_model, hidden, nl, do)

    if backbone == "mamba2_lm":
        headdims = [int(h) for h in list(cfg.model.layer_headdims)]
        return Mamba2LanguageModel(vocab, d_model, headdims, **mamba_kw)

    if backbone == "multiscale_mamba2_lm":
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        slow_hd = OmegaConf.select(cfg, "model.multiscale.slow_layer_headdims", default=None)
        fb = OmegaConf.select(cfg, "model.layer_headdims", default=None)
        if fast_hd is not None:
            fast_layer_headdims = [int(h) for h in list(fast_hd)]
        elif fb is not None:
            fast_layer_headdims = [int(h) for h in list(fb)]
        else:
            raise ValueError("multiscale_mamba2_lm: set model.multiscale.fast_layer_headdims or model.layer_headdims")
        if slow_hd is not None:
            slow_layer_headdims = [int(h) for h in list(slow_hd)]
        else:
            slow_layer_headdims = [32, 32]
        stride = int(OmegaConf.select(cfg, "model.multiscale.stride", default=4))
        fusion = str(OmegaConf.select(cfg, "model.multiscale.fusion", default="sum"))
        return MultiScaleMamba2LanguageModel(
            vocab,
            d_model,
            fast_layer_headdims,
            slow_layer_headdims,
            stride=stride,
            fusion=fusion,
            **mamba_kw,
        )

    if backbone == "multiscale_mamba2_attention_lm":
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        slow_hd = OmegaConf.select(cfg, "model.multiscale.slow_layer_headdims", default=None)
        fb = OmegaConf.select(cfg, "model.layer_headdims", default=None)
        if fast_hd is not None:
            fast_layer_headdims = [int(h) for h in list(fast_hd)]
        elif fb is not None:
            fast_layer_headdims = [int(h) for h in list(fb)]
        else:
            raise ValueError(
                "multiscale_mamba2_attention_lm: set model.multiscale.fast_layer_headdims "
                "or model.layer_headdims."
            )
        if slow_hd is not None:
            slow_layer_headdims = [int(h) for h in list(slow_hd)]
        else:
            slow_layer_headdims = [32, 32]
        stride = int(OmegaConf.select(cfg, "model.multiscale.stride", default=4))
        fusion = str(OmegaConf.select(cfg, "model.multiscale.fusion", default="residual"))
        attn_heads = int(OmegaConf.select(cfg, "model.multiscale.attention_heads", default=4))
        drop = float(OmegaConf.select(cfg, "model.multiscale.dropout", default=0.0))
        return MultiScaleMamba2AttentionLanguageModel(
            vocab,
            d_model,
            fast_layer_headdims,
            slow_layer_headdims,
            stride=stride,
            attention_heads=attn_heads,
            fusion=fusion,
            dropout=drop,
            **mamba_kw,
        )

    if backbone == "multiscale_mamba2_multiaxis_lm":
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        fb = OmegaConf.select(cfg, "model.layer_headdims", default=None)
        if fast_hd is not None:
            fast_layer_headdims = [int(h) for h in list(fast_hd)]
        elif fb is not None:
            fast_layer_headdims = [int(h) for h in list(fb)]
        else:
            raise ValueError(
                "multiscale_mamba2_multiaxis_lm: set model.multiscale.fast_layer_headdims "
                "or model.layer_headdims."
            )
        if not fast_layer_headdims:
            raise ValueError("multiscale_mamba2_multiaxis_lm: fast headdims must be non-empty.")

        branches_raw = OmegaConf.select(cfg, "model.multiscale.branches", default=None)
        if branches_raw is None:
            raise ValueError("multiscale_mamba2_multiaxis_lm: model.multiscale.branches is required.")
        branches_list = OmegaConf.to_container(branches_raw, resolve=True)
        if not isinstance(branches_list, list) or len(branches_list) == 0:
            raise ValueError("multiscale_mamba2_multiaxis_lm: multiscale.branches must be a non-empty list.")

        names_seen: set[str] = set()
        normalized_branches: list[dict[str, Any]] = []
        for idx, br in enumerate(branches_list):
            if not isinstance(br, dict):
                raise ValueError(f"multiscale.branches[{idx}] must be a mapping.")
            name = str(br.get("name", "")).strip()
            if not name:
                raise ValueError(f"multiscale.branches[{idx}].name must be a non-empty string.")
            if name in names_seen:
                raise ValueError(f"multiscale.branches contains duplicate name {name!r}.")
            names_seen.add(name)

            stride_raw = br.get("stride", None)
            if stride_raw is None:
                raise ValueError(f"multiscale.branches[{idx}].stride is required.")
            stride = int(stride_raw)
            if stride <= 1:
                raise ValueError(
                    f"multiscale.branches[{idx}].stride must be > 1 for slow branches, got {stride}."
                )

            layer_headdims_raw = br.get("layer_headdims", None)
            if not isinstance(layer_headdims_raw, list) or len(layer_headdims_raw) == 0:
                raise ValueError(f"multiscale.branches[{idx}].layer_headdims must be a non-empty list.")
            layer_headdims = [int(h) for h in layer_headdims_raw]

            normalized_branches.append(
                {
                    "name": name,
                    "stride": stride,
                    "layer_headdims": layer_headdims,
                }
            )

        fusion = str(OmegaConf.select(cfg, "model.multiscale.fusion", default="residual_gated_sum"))
        gate_type = str(OmegaConf.select(cfg, "model.multiscale.gate_type", default="vector"))
        return MultiAxisMamba2LanguageModel(
            vocab_size=vocab,
            d_model=d_model,
            fast_layer_headdims=fast_layer_headdims,
            branches=normalized_branches,
            fusion=fusion,
            gate_type=gate_type,
            **mamba_kw,
        )

    raise ValueError(f"Unknown model.backbone for LM: {backbone!r}")
