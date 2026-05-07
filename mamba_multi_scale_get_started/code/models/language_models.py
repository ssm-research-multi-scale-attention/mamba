"""Character-level language models: LSTM, Mamba2, multi-scale Mamba2, Transformer LM."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_backbones import MambaLayer


class TransformerLanguageModel(nn.Module):
    """
    Causal decoder-only Transformer: token + learned position embeddings,
    stacked causal self-attention blocks, final LayerNorm, LM head.
    """

    @staticmethod
    def _causal_self_attn_mask(
        seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )

    class _Block(nn.Module):
        def __init__(
            self,
            d_model: int,
            n_heads: int,
            mlp_ratio: float,
            dropout: float,
            norm_first: bool,
        ):
            super().__init__()
            self.norm_first = bool(norm_first)
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            dmlp = int(mlp_ratio * d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, dmlp),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dmlp, d_model),
                nn.Dropout(dropout),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            if self.norm_first:
                z = self.norm1(x)
                attn_out, _ = self.self_attn(z, z, z, attn_mask=attn_mask, need_weights=False)
                x = x + self.dropout(attn_out)
                x = x + self.mlp(self.norm2(x))
                return x
            attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.mlp(x))
            return x

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        norm_first: bool = True,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size!r}")
        if n_layers <= 0:
            raise ValueError(f"transformer.n_layers must be > 0, got {n_layers!r}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}); "
                "set transformer.n_heads so d_model % n_heads == 0."
            )
        if max_seq_len < 1:
            raise ValueError(f"transformer.max_seq_len must be >= 1, got {max_seq_len!r}")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.norm_first = bool(norm_first)

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            [
                TransformerLanguageModel._Block(
                    d_model,
                    n_heads,
                    mlp_ratio,
                    float(dropout),
                    norm_first,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"expected input_ids of shape [B, T], got shape {tuple(x.shape)}")
        _b, t = x.shape
        if t > self.max_seq_len:
            raise ValueError(
                f"sequence length ({t}) exceeds transformer.max_seq_len ({self.max_seq_len}); "
                "increase model.transformer.max_seq_len or shorten sequences."
            )
        pos = torch.arange(t, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))
        attn_mask = self._causal_self_attn_mask(t, device=h.device, dtype=h.dtype)
        for blk in self.blocks:
            h = blk(h, attn_mask)
        return self.lm_head(self.out_norm(h))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


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
        fast_A_init_range: tuple[float, float] = (0.1, 4.0),
        slow_A_init_range: tuple[float, float] = (4.0, 32.0),
        fast_dt_min: float = 0.0001,
        fast_dt_max: float = 0.03,
        slow_dt_min: float = 0.003,
        slow_dt_max: float = 0.3,
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
            *[MambaLayer(d_model=d_model, headdim=h, A_init_range=fast_A_init_range, dt_min=fast_dt_min, dt_max=fast_dt_max, **mamba_kwargs) for h in fast_layer_headdims]
        )
        self.slow_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, A_init_range=slow_A_init_range, dt_min=slow_dt_min, dt_max=slow_dt_max, **mamba_kwargs) for h in slow_layer_headdims]
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


class MultiScaleMamba2FiLMLanguageModel(nn.Module):
    """Fast Mamba residual path modulated by slow branch via FiLM."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        slow_layer_headdims: Sequence[int],
        stride: int = 4,
        fusion: str = "film",
        film_init_zero: bool = True,
        **mamba_kwargs: Any,
    ):
        super().__init__()
        st = int(stride)
        if st < 1:
            raise ValueError(f"stride must be >= 1, got {stride!r}")
        fusion_l = str(fusion).strip().lower()
        if fusion_l != "film":
            raise ValueError(f"multiscale_mamba2_film_lm requires multiscale.fusion='film', got {fusion!r}.")
        if not fast_layer_headdims:
            raise ValueError("multiscale_mamba2_film_lm: fast_layer_headdims must be non-empty.")
        if not slow_layer_headdims:
            raise ValueError("multiscale_mamba2_film_lm: slow_layer_headdims must be non-empty.")

        self.stride = st
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fast_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in fast_layer_headdims]
        )
        self.slow_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in slow_layer_headdims]
        )
        self.film_proj = nn.Linear(d_model, 2 * d_model)
        if bool(film_init_zero):
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast = self.fast_seq(emb)
        slow_emb = emb[:, :: self.stride, :]
        slow = self.slow_seq(slow_emb)
        seq_len = emb.size(1)
        slow_up = slow.repeat_interleave(self.stride, dim=1)
        if slow_up.size(1) < seq_len:
            pad_len = seq_len - slow_up.size(1)
            slow_up = F.pad(slow_up, (0, 0, 0, pad_len))
        slow_up = slow_up[:, :seq_len, :]

        gamma, beta = self.film_proj(slow_up).chunk(2, dim=-1)
        fused = fast * (1.0 + gamma) + beta
        return self.lm_head(self.out_norm(fused))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class MultiScaleMamba2WindowSummaryLanguageModel(nn.Module):
    """Fast full-seq Mamba + causal window-summary slow streams with residual fusion."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        fast_layer_headdims: Sequence[int],
        slow_layer_headdims: Sequence[int],
        window_size: int = 8,
        offsets: Sequence[int] = (0,),
        summary_mode: str = "mean",
        causal_mode: str = "completed",
        fusion: str = "window_residual_gated",
        gate_type: str = "vector",
        diagnostics: bool = False,
        **mamba_kwargs: Any,
    ):
        super().__init__()
        ws = int(window_size)
        if ws <= 1:
            raise ValueError(f"multiscale.window_size must be > 1, got {window_size!r}.")
        offs = [int(o) for o in list(offsets)]
        if not offs:
            raise ValueError("multiscale.offsets must be a non-empty list.")
        if len(set(offs)) != len(offs):
            raise ValueError(f"multiscale.offsets must be unique, got {offs!r}.")
        bad_offsets = [o for o in offs if o < 0 or o >= ws]
        if bad_offsets:
            raise ValueError(
                f"multiscale.offsets must satisfy 0 <= offset < window_size ({ws}), got {bad_offsets!r}."
            )
        sm = str(summary_mode).strip().lower()
        if sm not in ("mean", "last"):
            raise ValueError(f"multiscale.summary_mode must be 'mean' or 'last', got {summary_mode!r}.")
        cm = str(causal_mode).strip().lower()
        if cm not in ("completed", "current"):
            raise ValueError(f"multiscale.causal_mode must be 'completed' or 'current', got {causal_mode!r}.")
        fusion_l = str(fusion).strip().lower()
        if fusion_l not in ("window_residual_gated", "window_residual_sum", "window_fast_only"):
            raise ValueError(
                "multiscale.fusion must be 'window_residual_gated', 'window_residual_sum', or "
                "'window_fast_only', "
                f"got {fusion!r}."
            )
        gate_type_l = str(gate_type).strip().lower()
        if gate_type_l not in ("vector", "scalar"):
            raise ValueError(f"multiscale.gate_type must be 'vector' or 'scalar', got {gate_type!r}.")
        if not fast_layer_headdims:
            raise ValueError("multiscale_mamba2_window_summary_lm: fast_layer_headdims must be non-empty.")
        if not slow_layer_headdims:
            raise ValueError("multiscale_mamba2_window_summary_lm: slow_layer_headdims must be non-empty.")

        self.window_size = ws
        self.offsets = offs
        self.summary_mode = sm
        self.causal_mode = cm
        self.fusion = fusion_l
        self.gate_type = gate_type_l
        self.diagnostics = bool(diagnostics)
        self.last_window_debug: dict[str, float | int | list[dict[str, int]] | list[float]] = {}
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fast_seq = nn.Sequential(
            *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in fast_layer_headdims]
        )
        self.slow_seqs = nn.ModuleList(
            [
                nn.Sequential(
                    *[MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in slow_layer_headdims]
                )
                for _ in self.offsets
            ]
        )
        gate_out_dim = d_model if self.gate_type == "vector" else 1
        self.gate_projs = nn.ModuleList(
            [nn.Linear(2 * d_model, gate_out_dim) for _ in self.offsets]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _build_stream_summaries(
        self, emb: torch.Tensor, seq_len: int, offset: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = emb.device
        batch = emb.size(0)
        d_model = emb.size(-1)
        starts = torch.arange(offset, seq_len, self.window_size, device=device)
        if starts.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return None, torch.full((seq_len,), -1, dtype=torch.long, device=device), empty, empty

        ends_inclusive = torch.minimum(starts + self.window_size - 1, torch.full_like(starts, seq_len - 1))
        ends_exclusive = ends_inclusive + 1
        summaries: list[torch.Tensor] = []
        for s, e in zip(starts.tolist(), ends_inclusive.tolist()):
            w = emb[:, s : e + 1, :]
            if self.summary_mode == "mean":
                summaries.append(w.mean(dim=1))
            else:
                summaries.append(w[:, -1, :])
        summary_seq = torch.stack(summaries, dim=1) if summaries else None

        token_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        if self.causal_mode == "completed":
            valid = ends_exclusive.unsqueeze(0) <= (token_idx + 1)
        else:
            valid = starts.unsqueeze(0) <= token_idx
        last_idx = valid.to(torch.long).sum(dim=1) - 1
        if summary_seq is not None:
            valid_mask = last_idx >= 0
            if valid_mask.any():
                chosen_tokens = torch.arange(seq_len, device=device)[valid_mask]
                if self.causal_mode == "completed":
                    chosen_ends_ex = ends_exclusive[last_idx[valid_mask]]
                    if not bool(torch.all(chosen_ends_ex <= (chosen_tokens + 1))):
                        raise RuntimeError(
                            "Causal window mapping violated in completed mode: found end_exclusive > token+1."
                        )
                else:
                    chosen_starts = starts[last_idx[valid_mask]]
                    if not bool(torch.all(chosen_starts <= chosen_tokens)):
                        raise RuntimeError(
                            "Window mapping violated in current mode: found start > token index."
                        )

        if summary_seq is None:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return None, torch.full((seq_len,), -1, dtype=torch.long, device=device), empty, empty
        return summary_seq, last_idx, starts, ends_inclusive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        fast_out = self.fast_seq(emb)
        seq_len = emb.size(1)
        batch = emb.size(0)
        d_model = emb.size(-1)

        slow_ups: list[torch.Tensor] = []
        stream_maps: list[dict[str, int]] = []
        for offset, slow_seq in zip(self.offsets, self.slow_seqs):
            summary_seq, last_idx, starts, ends_inclusive = self._build_stream_summaries(
                emb, seq_len, offset
            )
            slow_up = emb.new_zeros((batch, seq_len, d_model))
            if summary_seq is not None:
                slow_out = slow_seq(summary_seq)
                valid = last_idx >= 0
                if valid.any():
                    gather_idx = last_idx[valid]
                    slow_up[:, valid, :] = slow_out[:, gather_idx, :]
                if self.diagnostics and batch > 0 and starts.numel() > 0:
                    sample_tokens = list(range(min(41, seq_len)))
                    for t in sample_tokens:
                        wi = int(last_idx[t].item())
                        if wi >= 0:
                            stream_maps.append(
                                {
                                    "offset": int(offset),
                                    "token": int(t),
                                    "window_index": wi,
                                    "window_start": int(starts[wi].item()),
                                    "window_end": int(ends_inclusive[wi].item()),
                                }
                            )
            slow_ups.append(slow_up)

        gate_vals: list[torch.Tensor] = []
        if self.fusion == "window_fast_only":
            fused = fast_out
        elif self.fusion == "window_residual_sum":
            stacked = torch.stack(slow_ups, dim=0)
            fused = fast_out + stacked.mean(dim=0)
        else:
            fused = fast_out
            for gate_proj, slow_up in zip(self.gate_projs, slow_ups):
                gate = torch.sigmoid(gate_proj(torch.cat([fast_out, slow_up], dim=-1)))
                gate_vals.append(gate)
                fused = fused + gate * slow_up

        logits = self.lm_head(self.out_norm(fused))
        if self.diagnostics:
            with torch.no_grad():
                slow_cat = torch.stack(slow_ups, dim=0) if slow_ups else None
                slow_nonzero_frac = 0.0
                slow_norm_mean = 0.0
                slow_norm_std = 0.0
                if slow_cat is not None:
                    slow_abs = slow_cat.abs().sum(dim=-1)
                    slow_nonzero_frac = float((slow_abs > 0).float().mean().item())
                    slow_norm = slow_cat.norm(dim=-1)
                    slow_norm_mean = float(slow_norm.mean().item())
                    slow_norm_std = float(slow_norm.std(unbiased=False).item())
                fast_norm = fast_out.norm(dim=-1)
                gate_mean = 0.0
                gate_std = 0.0
                gate_sample: list[float] = []
                if gate_vals:
                    gate_cat = torch.cat([g.reshape(-1) for g in gate_vals], dim=0)
                    gate_mean = float(gate_cat.mean().item())
                    gate_std = float(gate_cat.std(unbiased=False).item())
                    gate_sample = [float(v) for v in gate_cat[:8].tolist()]
                self.last_window_debug = {
                    "seq_len": int(seq_len),
                    "window_size": int(self.window_size),
                    "offsets": [int(o) for o in self.offsets],
                    "summary_mode": str(self.summary_mode),
                    "causal_mode": str(self.causal_mode),
                    "fusion": str(self.fusion),
                    "slow_up_nonzero_fraction": slow_nonzero_frac,
                    "fast_norm_mean": float(fast_norm.mean().item()),
                    "fast_norm_std": float(fast_norm.std(unbiased=False).item()),
                    "slow_norm_mean": slow_norm_mean,
                    "slow_norm_std": slow_norm_std,
                    "gate_mean": gate_mean,
                    "gate_std": gate_std,
                    "gate_sample": gate_sample,
                    "nan_in_fast": int(torch.isnan(fast_out).any().item()),
                    "nan_in_logits": int(torch.isnan(logits).any().item()),
                    "mapping_sample": stream_maps[:120],
                }
        return logits

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
        fast_A_init_range = OmegaConf.select(cfg, "model.multiscale.fast_A_init_range", default=None)
        slow_A_init_range = OmegaConf.select(cfg, "model.multiscale.slow_A_init_range", default=None)
        fast_dt_min = OmegaConf.select(cfg, "model.multiscale.fast_dt_min", default=None)
        fast_dt_max = OmegaConf.select(cfg, "model.multiscale.fast_dt_max", default=None)
        slow_dt_min = OmegaConf.select(cfg, "model.multiscale.slow_dt_min", default=None)
        slow_dt_max = OmegaConf.select(cfg, "model.multiscale.slow_dt_max", default=None)
        return MultiScaleMamba2LanguageModel(
            vocab,
            d_model,
            fast_layer_headdims,
            slow_layer_headdims,
            stride=stride,
            fusion=fusion,
            fast_A_init_range=fast_A_init_range,
            slow_A_init_range=slow_A_init_range,
            fast_dt_min=fast_dt_min,
            fast_dt_max=fast_dt_max,
            slow_dt_min=slow_dt_min,
            slow_dt_max=slow_dt_max,
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

    if backbone == "multiscale_mamba2_film_lm":
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        slow_hd = OmegaConf.select(cfg, "model.multiscale.slow_layer_headdims", default=None)
        fb = OmegaConf.select(cfg, "model.layer_headdims", default=None)
        if fast_hd is not None:
            fast_layer_headdims = [int(h) for h in list(fast_hd)]
        elif fb is not None:
            fast_layer_headdims = [int(h) for h in list(fb)]
        else:
            raise ValueError(
                "multiscale_mamba2_film_lm: set model.multiscale.fast_layer_headdims or model.layer_headdims."
            )
        if not fast_layer_headdims:
            raise ValueError("multiscale_mamba2_film_lm: fast headdims must be non-empty.")
        if slow_hd is not None:
            slow_layer_headdims = [int(h) for h in list(slow_hd)]
        else:
            raise ValueError("multiscale_mamba2_film_lm: model.multiscale.slow_layer_headdims is required.")
        if not slow_layer_headdims:
            raise ValueError("multiscale_mamba2_film_lm: slow headdims must be non-empty.")
        stride = int(OmegaConf.select(cfg, "model.multiscale.stride", default=4))
        fusion = str(OmegaConf.select(cfg, "model.multiscale.fusion", default="film"))
        film_init_zero = bool(OmegaConf.select(cfg, "model.multiscale.film_init_zero", default=True))
        return MultiScaleMamba2FiLMLanguageModel(
            vocab_size=vocab,
            d_model=d_model,
            fast_layer_headdims=fast_layer_headdims,
            slow_layer_headdims=slow_layer_headdims,
            stride=stride,
            fusion=fusion,
            film_init_zero=film_init_zero,
            **mamba_kw,
        )

    if backbone == "multiscale_mamba2_window_summary_lm":
        fast_hd = OmegaConf.select(cfg, "model.multiscale.fast_layer_headdims", default=None)
        slow_hd = OmegaConf.select(cfg, "model.multiscale.slow_layer_headdims", default=None)
        fb = OmegaConf.select(cfg, "model.layer_headdims", default=None)
        if fast_hd is not None:
            fast_layer_headdims = [int(h) for h in list(fast_hd)]
        elif fb is not None:
            fast_layer_headdims = [int(h) for h in list(fb)]
        else:
            raise ValueError(
                "multiscale_mamba2_window_summary_lm: set model.multiscale.fast_layer_headdims "
                "or model.layer_headdims."
            )
        if not fast_layer_headdims:
            raise ValueError("multiscale_mamba2_window_summary_lm: fast headdims must be non-empty.")
        if slow_hd is None:
            raise ValueError("multiscale_mamba2_window_summary_lm: model.multiscale.slow_layer_headdims is required.")
        slow_layer_headdims = [int(h) for h in list(slow_hd)]
        if not slow_layer_headdims:
            raise ValueError("multiscale_mamba2_window_summary_lm: slow headdims must be non-empty.")
        window_size = int(OmegaConf.select(cfg, "model.multiscale.window_size", default=8))
        offsets_raw = OmegaConf.select(cfg, "model.multiscale.offsets", default=[0])
        offsets = [int(o) for o in list(offsets_raw)]
        summary_mode = str(OmegaConf.select(cfg, "model.multiscale.summary_mode", default="mean"))
        causal_mode = str(OmegaConf.select(cfg, "model.multiscale.causal_mode", default="completed"))
        fusion = str(
            OmegaConf.select(cfg, "model.multiscale.fusion", default="window_residual_gated")
        )
        gate_type = str(OmegaConf.select(cfg, "model.multiscale.gate_type", default="vector"))
        diagnostics = bool(OmegaConf.select(cfg, "model.multiscale.diagnostics", default=False))
        return MultiScaleMamba2WindowSummaryLanguageModel(
            vocab_size=vocab,
            d_model=d_model,
            fast_layer_headdims=fast_layer_headdims,
            slow_layer_headdims=slow_layer_headdims,
            window_size=window_size,
            offsets=offsets,
            summary_mode=summary_mode,
            causal_mode=causal_mode,
            fusion=fusion,
            gate_type=gate_type,
            diagnostics=diagnostics,
            **mamba_kw,
        )

    if backbone == "transformer_lm":
        tr_raw = OmegaConf.select(cfg, "model.transformer", default=None)
        if tr_raw is None:
            raise ValueError("transformer_lm requires model.transformer with n_layers, n_heads, etc.")
        tr = OmegaConf.to_container(tr_raw, resolve=True)
        if not isinstance(tr, dict):
            raise ValueError("model.transformer must be a mapping (e.g. n_layers, n_heads).")
        for key in ("n_layers", "n_heads", "max_seq_len"):
            if key not in tr:
                raise ValueError(f"transformer_lm: model.transformer.{key} is required.")
        n_layers = int(tr["n_layers"])
        n_heads = int(tr["n_heads"])
        mlp_ratio = float(tr.get("mlp_ratio", 4.0))
        dropout = float(tr.get("dropout", 0.0))
        max_seq_len = int(tr["max_seq_len"])
        norm_first = bool(tr.get("norm_first", True))
        return TransformerLanguageModel(
            vocab_size=vocab,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            norm_first=norm_first,
        )

    raise ValueError(f"Unknown model.backbone for LM: {backbone!r}")
