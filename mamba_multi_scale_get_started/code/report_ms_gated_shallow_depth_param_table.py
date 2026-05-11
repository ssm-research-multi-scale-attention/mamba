#!/usr/bin/env python3
"""Print param counts for MS-gated right-init depth ablations (width fixed like full-size).

Full reference: d_model=256, headdim=32, d_state=32, expand=2, ngroups=1, stride=2, gated.
Depth is (len(fast_layer_headdims), len(slow_layer_headdims)).

No ±5% match exists with depth-only changes: 2/1 is too small, 3/1 is the closest above baseline.
Run from repo root: ``python code/report_ms_gated_shallow_depth_param_table.py``
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

_CODE = Path(__file__).resolve().parent
ROOT = _CODE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from exp_config import load_config
from models.language_models import build_lm


def _count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _params_ms_gated(n_fast: int, n_slow: int, vocab: int) -> int:
    tpl = OmegaConf.load(ROOT / "configs/Text8/text8_ms_gated_right_init.yaml")
    tpl.model.vocab_size = vocab
    hd_f = [32] * n_fast
    hd_s = [32] * n_slow
    tpl.model.layer_headdims = hd_f
    tpl.model.multiscale.fast_layer_headdims = hd_f
    tpl.model.multiscale.slow_layer_headdims = hd_s
    fd, name = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        OmegaConf.save(tpl, name)
        cfg = load_config(name, [])
        cfg.model.vocab_size = vocab
        return _count_trainable(build_lm(cfg))
    finally:
        os.unlink(name)


def _cuda_forward_ok(vocab: int, n_fast: int, n_slow: int) -> str:
    import torch

    if not torch.cuda.is_available():
        return "skipped(no_cuda)"
    tpl = OmegaConf.load(ROOT / "configs/Text8/text8_ms_gated_right_init.yaml")
    tpl.model.vocab_size = vocab
    hd_f = [32] * n_fast
    hd_s = [32] * n_slow
    tpl.model.layer_headdims = hd_f
    tpl.model.multiscale.fast_layer_headdims = hd_f
    tpl.model.multiscale.slow_layer_headdims = hd_s
    fd, name = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        OmegaConf.save(tpl, name)
        cfg = load_config(name, [])
        cfg.model.vocab_size = vocab
        m = build_lm(cfg).to("cuda").eval()
        x = torch.randint(0, vocab, (2, 128), device="cuda")
        with torch.inference_mode():
            y = m(x)
        if tuple(y.shape) != (2, 128, vocab):
            return f"bad_shape{tuple(y.shape)}"
        return "ok"
    except Exception as e:
        return f"fail({e})"
    finally:
        os.unlink(name)


def main() -> None:
    tiny_m2 = ROOT / "configs/TinyShakespeare/tiny_shakespeare_mamba2_depth4.yaml"
    mqar_m2 = ROOT / "configs/MQAR/mqar_mamba2_depth4_len128.yaml"

    def count_path(p: Path) -> int:
        cfg = load_config(str(p), [])
        ds = str(OmegaConf.select(cfg, "data.dataset", default="") or "").lower()
        if ds == "mqar":
            cfg.model.vocab_size = int(cfg.data.vocab_size)
        elif ds == "text8" and int(cfg.model.vocab_size) == 0:
            cfg.model.vocab_size = 27
        return _count_trainable(build_lm(cfg))

    base_tiny = count_path(tiny_m2)
    base_mqar = count_path(mqar_m2)
    text8_m2 = ROOT / "configs/Text8/text8_mamba2_depth4.yaml"
    base_text8 = count_path(text8_m2)

    pairs = [(4, 2), (4, 1), (3, 2), (3, 1), (2, 2), (2, 1), (1, 1)]
    arch = "ms_gated_right_init_depth_only"
    d_model, headdim, d_state = 256, 32, 32

    print(
        "| arch_name | fast_depth | slow_depth | d_model | headdim | d_state | "
        "tiny_params | d_tiny_vs_m2% | text8_params | d_text8_vs_m2% | mqar_params | d_mqar_vs_m2% | forward_ok |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for nf, ns in pairs:
        if ns < 1:
            continue
        t = _params_ms_gated(nf, ns, 65)
        tx = _params_ms_gated(nf, ns, 27)
        mq = _params_ms_gated(nf, ns, 1024)
        dt = 100.0 * (t - base_tiny) / base_tiny
        dx = 100.0 * (tx - base_text8) / base_text8
        dm = 100.0 * (mq - base_mqar) / base_mqar
        ok = _cuda_forward_ok(27, nf, ns)
        print(
            f"| {arch} | {nf} | {ns} | {d_model} | {headdim} | {d_state} | "
            f"{t} | {dt:+.4f}% | {tx} | {dx:+.4f}% | {mq} | {dm:+.4f}% | {ok} |"
        )

    lo, hi = 0.95 * base_tiny, 1.05 * base_tiny
    print()
    print(f"Mamba2 depth4 baselines — Tiny: {base_tiny}, Text8 (vocab 27): {base_text8}, MQAR: {base_mqar}")
    print(f"Tiny ±5% band: [{lo:.0f}, {hi:.0f}]")
    print("Closest depth-only above band: fast/slow 3/1 (~+7.8% Tiny, ~+7.9% Text8). Below: 2/1 (~−16.7% Tiny).")
    print("Chosen configs: 3/1 (not strict equal-param; documented in YAML headers).")


if __name__ == "__main__":
    main()
