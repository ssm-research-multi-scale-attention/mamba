#!/usr/bin/env python3
"""
Grid-search multiscale_mamba2_lm (gated, stride 2, fast/slow right-ish init) near param baselines.

For each candidate:
  - conv_dim = expand*d_model + 2*ngroups*d_state must be ≡ 0 (mod 8) (causal_conv1d).
  - Build LM, count trainable params (Tiny vocab 65 + MQAR vocab 1024).
  - CUDA forward: x = randint(2, 128) on GPU, logits = model(x); catch errors.
  - Prefer (d_model % 8 == 0) and (headdim % 8 == 0) when possible.

Default grid matches user suggestion plus d_state multiples of 8 (SSM width knob).
Does not train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from models.language_models import build_lm


def build_cfg(
    *,
    vocab_override: int,
    d_model: int,
    fast_hd: list[int],
    slow_hd: list[int],
    d_state: int,
    d_conv: int = 4,
    expand: int = 2,
    ngroups: int = 1,
) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "backbone": "multiscale_mamba2_lm",
                "d_model": int(d_model),
                "layer_headdims": list(fast_hd),
                "vocab_size": int(vocab_override),
                "multiscale": {
                    "stride": 2,
                    "fusion": "gated",
                    "fast_layer_headdims": list(fast_hd),
                    "slow_layer_headdims": list(slow_hd),
                    "fast_A_init_range": [0.1, 4.0],
                    "slow_A_init_range": [4.0, 32.0],
                    "fast_dt_min": 0.0001,
                    "fast_dt_max": 0.03,
                    "slow_dt_min": 0.003,
                    "slow_dt_max": 0.3,
                },
                "mamba": {
                    "d_state": int(d_state),
                    "d_conv": int(d_conv),
                    "expand": int(expand),
                    "ngroups": int(ngroups),
                },
            }
        }
    )


def conv_chan_div8(d_model: int, d_state: int, *, expand: int, ngroups: int) -> bool:
    conv_dim = expand * int(d_model) + 2 * int(ngroups) * int(d_state)
    return conv_dim % 8 == 0


def inner_ok(d_model: int, expand: int, headdims: list[int]) -> bool:
    d_inner = expand * int(d_model)
    return all(d_inner % int(h) == 0 for h in headdims)


def cuda_forward_ok(model: torch.nn.Module, vocab: int, device: torch.device) -> tuple[bool, str]:
    model = model.to(device)
    try:
        x = torch.randint(0, vocab, (2, 128), device=device)
        with torch.inference_mode():
            logits = model(x)
        if logits.shape != (2, 128, vocab):
            return False, f"bad shape {tuple(logits.shape)} expected (2,128,{vocab})"
        return True, ""
    except Exception as e:
        return False, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny-vocab", type=int, default=65)
    ap.add_argument("--mqar-vocab", type=int, default=1024)
    ap.add_argument("--tiny-baseline", type=int, default=1_702_401)
    ap.add_argument("--mqar-baseline", type=int, default=2_194_368)
    ap.add_argument("--band-pct", type=float, default=5.0)
    ap.add_argument(
        "--d-states",
        type=int,
        nargs="*",
        default=None,
        help="SSM widths to try (default: 32,40,...,192 step 8).",
    )
    args = ap.parse_args()

    d_states_src = (
        args.d_states if args.d_states is not None else list(range(32, 200, 8))
    )
    d_states = sorted({int(x) for x in d_states_src})

    tiny_lo = args.tiny_baseline * (1.0 - args.band_pct / 100.0)
    tiny_hi = args.tiny_baseline * (1.0 + args.band_pct / 100.0)
    mq_lo = args.mqar_baseline * (1.0 - args.band_pct / 100.0)
    mq_hi = args.mqar_baseline * (1.0 + args.band_pct / 100.0)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")

    d_models = [184, 192, 200, 208, 216, 224]
    headdims = [8, 16, 24, 32]
    expand, ngroups = 2, 1
    fast_n, slow_n = 4, 2

    raw: list[dict[str, object]] = []

    for dm in d_models:
        for h in headdims:
            fast_hd = [h] * fast_n
            slow_hd = [h] * slow_n
            if not inner_ok(dm, expand, fast_hd + slow_hd):
                continue
            for d_state in d_states:
                if not conv_chan_div8(dm, d_state, expand=expand, ngroups=ngroups):
                    continue
                slug = f"dm{dm}_h{h}_ds{d_state}_f{fast_n}s{slow_n}"
                tiny_cfg = build_cfg(vocab_override=args.tiny_vocab, d_model=dm, fast_hd=fast_hd, slow_hd=slow_hd, d_state=d_state)
                mqar_cfg = build_cfg(vocab_override=args.mqar_vocab, d_model=dm, fast_hd=fast_hd, slow_hd=slow_hd, d_state=d_state)

                m_tiny = build_lm(tiny_cfg)
                pt = sum(p.numel() for p in m_tiny.parameters() if p.requires_grad)

                mq = build_lm(mqar_cfg)
                pm = sum(p.numel() for p in mq.parameters() if p.requires_grad)

                fwd_ok = True
                err = ""
                if has_cuda:
                    ok_t, et = cuda_forward_ok(m_tiny, args.tiny_vocab, device)
                    ok_m, em = cuda_forward_ok(mq, args.mqar_vocab, device)
                    fwd_ok = ok_t and ok_m
                    err = (et + (" | " if em else "") + em).strip(" |")

                pct_t = 100.0 * (float(pt) - args.tiny_baseline) / float(args.tiny_baseline)
                pct_m = 100.0 * (float(pm) - args.mqar_baseline) / float(args.mqar_baseline)
                in_band = bool(tiny_lo <= pt <= tiny_hi and mq_lo <= pm <= mq_hi)

                raw.append(
                    {
                        "arch_slug": slug,
                        "d_model": dm,
                        "headdim": h,
                        "fast_hd": tuple(fast_hd),
                        "slow_hd": tuple(slow_hd),
                        "d_state": d_state,
                        "tiny_params": pt,
                        "tiny_pct": pct_t,
                        "mqar_params": pm,
                        "mqar_pct": pct_m,
                        "in_band": in_band,
                        "forward_ok": fwd_ok,
                        "forward_err": err,
                        "max_abs_pct": max(abs(pct_t), abs(pct_m)),
                    }
                )

    band_fwd = [
        r
        for r in raw
        if r["forward_ok"] and bool(r["in_band"])
    ]

    ranked = sorted(
        band_fwd,
        key=lambda r: float(r["max_abs_pct"]),
    )
    fallback = sorted(
        [r for r in raw if r["forward_ok"]],
        key=lambda r: (float(r["max_abs_pct"]), -1 if bool(r["in_band"]) else 0),
    )
    printable = ranked if ranked else fallback

    print("# conv_dim % 8 == 0; CUDA smoke:", "enabled" if has_cuda else "skipped (CPU only)")
    print(f"# baseline tiny={args.tiny_baseline} mqar={args.mqar_baseline} ±{args.band_pct}%")
    print("# columns: arch_name, d_model, headdim, fast_headdims, slow_headdims, tiny_params, delta_pct, mqar_params, delta_pct, forward_ok")

    max_show = min(40, len(printable))
    for r in printable[:max_show]:
        print(
            f"{r['arch_slug']}\t{r['d_model']}\t{r['headdim']}\t{r['fast_hd']}\t{r['slow_hd']}\t"
            f"{r['tiny_params']}\t{r['tiny_pct']:+.5f}\t{r['mqar_params']}\t{r['mqar_pct']:+.5f}\t{r['forward_ok']}"
        )

    failures = [
        r
        for r in raw
        if not r["forward_ok"] and (r["arch_slug"].startswith(("dm192", "dm200", "dm224")))
    ][:6]
    if failures and not printable:
        print("\n# sample forward failures for debugging:")
        for r in failures:
            print(f"# {r['arch_slug']}: {r['forward_err']}")

    non_ok_band = sum(1 for r in raw if bool(r["in_band"]))
    print(f"\n# summary: total_trials={len(raw)} cuda_forward_ok={sum(1 for r in raw if r['forward_ok'])} "
          f"in±{args.band_pct}%_band_any={non_ok_band} in_band_and_forward_ok={len(ranked)}")


if __name__ == "__main__":
    main()
