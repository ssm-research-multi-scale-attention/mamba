#!/usr/bin/env python3
"""
CUDA forward profiling (no training): latency vs batch for selected LMs.

Uses torch.cuda.Event timing, torch.inference_mode(), model.eval(), and
torch.cuda.synchronize() before reading elapsed time.

MS-gated (MultiScaleMamba2LanguageModel) segments mirror forward():
  embed → fast_seq → (stride slice + slow_seq) → (repeat_interleave + fusion) →
  out_norm + lm_head.

Per repeat iteration we run TWO forwards: one timed end-to-end (`total_fwd`),
one timed step-by-step (segment row). Means are comparable; sums of segment
means can slightly exceed total due to scheduler / cache effects.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from exp_config import load_config
from models.language_models import (
    MultiScaleMamba2LanguageModel,
    Mamba2LanguageModel,
    build_lm,
)

PROJECT_ROOT = _CODE.parent

DEFAULT_CONFIGS: dict[str, str] = {
    "mamba2_depth4": "configs/TinyShakespeare/tiny_shakespeare_mamba2_depth4.yaml",
    "ms_gated_right_init": "configs/TinyShakespeare/tiny_shakespeare_ms_gated_stride2_right_init.yaml",
    "ms_gated_right_init_equal_param": (
        "configs/TinyShakespeare/tiny_shakespeare_ms_gated_stride2_right_init_equal_param.yaml"
    ),
    "mamba2_split_ts_slow_veryslow": (
        "configs/TinyShakespeare/tiny_shakespeare_mamba2_split_ts_slow_veryslow.yaml"
    ),
}


def _maybe_sync_mqar_vocab(cfg: DictConfig) -> None:
    data_name = str(OmegaConf.select(cfg, "data.dataset", default="")).strip().lower()
    if data_name != "mqar":
        return
    v = OmegaConf.select(cfg, "data.vocab_size", default=None)
    if v is not None and cfg.model is not None:
        cfg.model.vocab_size = int(v)


def _conv_dim(block: nn.Module) -> int:
    return int(block.d_inner) + 2 * int(block.ngroups) * int(block.d_state)


@dataclass(frozen=True)
class ModelDigest:
    name: str
    backbone: str
    d_model: int
    vocab_size: int
    depth: str
    headdims: str
    d_state: int
    expand: int
    conv_dim_single: str
    n_params: int


def summarize(model: nn.Module, name: str) -> ModelDigest:
    if isinstance(model, Mamba2LanguageModel):
        m0 = model.blocks[0].block
        conv = _conv_dim(m0)
        hds = [int(model.blocks[i].block.headdim) for i in range(len(model.blocks))]
        ds = int(m0.d_state)
        exp = int(m0.expand)
        depth = str(len(model.blocks))
        conv_s = str(conv)
        h_repr = str(hds)
    elif isinstance(model, MultiScaleMamba2LanguageModel):
        f0 = model.fast_seq[0].block
        conv_f = _conv_dim(f0)
        h_fast = [int(model.fast_seq[i].block.headdim) for i in range(len(model.fast_seq))]
        h_slow = [int(model.slow_seq[i].block.headdim) for i in range(len(model.slow_seq))]
        ds = int(f0.d_state)
        exp = int(f0.expand)
        depth = f"fast:{len(model.fast_seq)} slow:{len(model.slow_seq)}"
        conv_s = f"per_block_fast={conv_f}"
        slow0 = model.slow_seq[0].block
        if conv_f != _conv_dim(slow0):
            conv_s += f" slow={_conv_dim(slow0)}"
        h_repr = f"fast{h_fast} slow{h_slow}"
    else:
        raise TypeError(f"unsupported model type: {type(model)}")

    np_ = sum(p.numel() for p in model.parameters())
    d_model = int(model.embed.embedding_dim)
    vocab = int(model.embed.num_embeddings)

    return ModelDigest(
        name=name,
        backbone=type(model).__name__,
        d_model=d_model,
        vocab_size=vocab,
        depth=depth,
        headdims=h_repr,
        d_state=ds,
        expand=exp,
        conv_dim_single=conv_s,
        n_params=np_,
    )


def _evt_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def forward_ms_gated_segments(
    m: MultiScaleMamba2LanguageModel,
    x: torch.Tensor,
    ev: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
) -> dict[str, float]:
    stride = int(m.stride)
    out: dict[str, float] = {}

    es, ee = ev["embed"]
    es.record()
    emb = m.embed(x)
    ee.record()
    out["embed"] = _evt_elapsed_ms(es, ee)

    fs, fe = ev["fast"]
    fs.record()
    fast = m.fast_seq(emb)
    fe.record()
    out["fast_seq"] = _evt_elapsed_ms(fs, fe)

    ss, se = ev["slow"]
    ss.record()
    slow_emb = emb[:, ::stride, :]
    slow = m.slow_seq(slow_emb)
    se.record()
    out["slow_seq"] = _evt_elapsed_ms(ss, se)

    L = emb.size(1)
    us, ue = ev["fuse"]
    us.record()
    slow_up = slow.repeat_interleave(stride, dim=1)[:, :L, :]
    if m.fusion == "sum":
        fused = fast + slow_up
    elif m.fusion == "concat":
        fused = m.fusion_proj(torch.cat([fast, slow_up], dim=-1))
    elif m.fusion == "gated":
        assert m.gate is not None
        gate = torch.sigmoid(m.gate(torch.cat([fast, slow_up], dim=-1)))
        fused = fast + gate * slow_up
    else:
        raise RuntimeError(f"unknown fusion {m.fusion!r}")
    ue.record()
    out["fusion"] = _evt_elapsed_ms(us, ue)

    hs, he = ev["head"]
    hs.record()
    logits = m.lm_head(m.out_norm(fused))
    he.record()
    out["out_norm_lm_head"] = _evt_elapsed_ms(hs, he)

    return out


def _make_event_pairs(names: tuple[str, ...]) -> dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]:
    return {n: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for n in names}


def _stats_ms(samples: list[float]) -> tuple[float, float, float]:
    a = np.asarray(samples, dtype=np.float64)
    return float(np.mean(a)), float(np.percentile(a, 50)), float(np.percentile(a, 95))


def profile_ms_gated(
    m: MultiScaleMamba2LanguageModel,
    batch: int,
    seq_len: int,
    vocab: int,
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, list[float]], list[float]]:
    device = next(m.parameters()).device
    x = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)

    totals: list[float] = []
    seg_times: dict[str, list[float]] = {
        "embed": [],
        "fast_seq": [],
        "slow_seq": [],
        "fusion": [],
        "out_norm_lm_head": [],
    }

    ev_total_s = torch.cuda.Event(enable_timing=True)
    ev_total_e = torch.cuda.Event(enable_timing=True)
    ev_seg = _make_event_pairs(("embed", "fast", "slow", "fuse", "head"))

    with torch.inference_mode():
        for _ in range(warmup):
            _ = m(x)
            torch.cuda.synchronize()

        for _ in range(repeats):
            ev_total_s.record()
            logits = m(x)
            ev_total_e.record()
            _ = logits.sum()
            totals.append(_evt_elapsed_ms(ev_total_s, ev_total_e))

            segs = forward_ms_gated_segments(m, x, ev_seg)
            for k in seg_times:
                seg_times[k].append(segs[k])

    return seg_times, totals


def profile_plain(
    model: nn.Module,
    batch: int,
    seq_len: int,
    vocab: int,
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    device = next(model.parameters()).device
    x = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)
    es = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    totals: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            torch.cuda.synchronize()
        for _ in range(repeats):
            es.record()
            logits = model(x)
            ee.record()
            _ = logits.sum()
            totals.append(_evt_elapsed_ms(es, ee))
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwd", type=str, default=str(PROJECT_ROOT))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--cuda-device", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.cwd).resolve()

    if not torch.cuda.is_available():
        print("CUDA not available — this script targets GPU timing. Exiting.")
        sys.exit(1)

    dev = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(dev)

    arches = list(DEFAULT_CONFIGS.keys())
    models: dict[str, nn.Module] = {}
    digests: dict[str, ModelDigest] = {}

    for arch in arches:
        cfg = load_config(str(root / DEFAULT_CONFIGS[arch]), [])
        _maybe_sync_mqar_vocab(cfg)
        model = build_lm(cfg).to(dev).eval()
        models[arch] = model
        digests[arch] = summarize(model, arch)

    print("# profile_ms_gated_timing.py")
    print(f"# torch.inference_mode + model.eval() + cuda.Events + synchronize before reads")
    print(f"# device={dev} warmup={args.warmup} repeats={args.repeats} seq_len={args.seq_len}")

    print(
        "\n## Why ms_gated_right_init_equal_param can match ms_gated_right_init throughput\n"
        "## (brief): equal-param narrows conv_dim/headdims but keeps the SAME number of fused\n"
        "## Mamba2 forwards per step (fast 4x full-L + slow 2x L/stride). GPU time is often\n"
        "## dominated by kernel launches & sequence length, so similar wall time is plausible\n"
        "## until memory bandwidth or occupancy diverges strongly.\n"
    )

    for arch in arches:
        d = digests[arch]
        m = models[arch]
        print(f"\n=== {arch} ===")
        print(
            f"  backbone={d.backbone} d_model={d.d_model} vocab={d.vocab_size} "
            f"layers/depth={d.depth} headdims={d.headdims}"
        )
        print(
            f"  d_state={d.d_state} expand={d.expand} conv_dim(one Mamba2 block)={d.conv_dim_single} "
            f"trainable_params={d.n_params}"
        )

        for b in args.batch_sizes:
            if isinstance(m, MultiScaleMamba2LanguageModel):
                segs, totals = profile_ms_gated(
                    m,
                    b,
                    args.seq_len,
                    d.vocab_size,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                mean_t, p50_t, p95_t = _stats_ms(totals)
                thr = (
                    (b * args.seq_len) / (mean_t / 1000.0) if mean_t > 1e-6 else float("nan")
                )
                print(
                    f"  batch={b:>2}: total_fwd mean={mean_t:.4f}ms p50={p50_t:.4f}ms "
                    f"p95={p95_t:.4f}ms (~{thr:.1f} tok/s equiv.)"
                )
                summ = 0.0
                for key in ["embed", "fast_seq", "slow_seq", "fusion", "out_norm_lm_head"]:
                    mu, p50, p95 = _stats_ms(segs[key])
                    summ += mu
                    pct = 100.0 * mu / mean_t if mean_t > 0 else 0.0
                    print(
                        f"    {key:<20} mean={mu:.4f}ms p50={p50:.4f}ms p95={p95:.4f}ms (~{pct:.1f}% of total)"
                    )
                print(
                    f"    segmented_mean_sum={summ:.4f}ms vs total_fwd_mean={mean_t:.4f}ms "
                    f"(second forward path / sync overhead)"
                )
            else:
                totals = profile_plain(
                    m,
                    b,
                    args.seq_len,
                    d.vocab_size,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                mean_t, p50_t, p95_t = _stats_ms(totals)
                thr = (
                    (b * args.seq_len) / (mean_t / 1000.0) if mean_t > 1e-6 else float("nan")
                )
                print(
                    f"  batch={b:>2}: full_forward mean={mean_t:.4f}ms p50={p50_t:.4f}ms "
                    f"p95={p95_t:.4f}ms (~{thr:.1f} tok/s equiv.)"
                )
                print(f"    (MS-gated segment breakdown not applicable)")
    print()


if __name__ == "__main__":
    main()
