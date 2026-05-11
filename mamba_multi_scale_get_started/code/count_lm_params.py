#!/usr/bin/env python3
"""Dry-run: build LM from YAML and print trainable parameter counts (no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from omegaconf import DictConfig, OmegaConf

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from exp_config import load_config
from models.language_models import build_lm

PROJECT_ROOT = _CODE.parent


def _maybe_sync_mqar_vocab(cfg: DictConfig) -> None:
    data_name = str(OmegaConf.select(cfg, "data.dataset", default="")).strip().lower()
    if data_name != "mqar":
        return
    v = OmegaConf.select(cfg, "data.vocab_size", default=None)
    if v is not None:
        if "model" not in cfg or cfg.model is None:
            raise ValueError("MQAR configs must define cfg.model.")
        cfg.model.vocab_size = int(v)


def _maybe_sync_text8_vocab_for_count(cfg: DictConfig) -> None:
    """train_lm sets vocab from the built char table; for dry-run counts use a fixed proxy when unset."""
    data_name = str(OmegaConf.select(cfg, "data.dataset", default="")).strip().lower()
    if data_name != "text8":
        return
    if int(OmegaConf.select(cfg, "model.vocab_size", default=0)) == 0:
        if "model" not in cfg or cfg.model is None:
            raise ValueError("Text8 configs must define cfg.model.")
        cfg.model.vocab_size = 27


def _count_params(cfg: DictConfig, *, cuda_smoke: bool) -> tuple[int, str]:
    _maybe_sync_mqar_vocab(cfg)
    _maybe_sync_text8_vocab_for_count(cfg)
    model = build_lm(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not cuda_smoke:
        return n, ""

    vocab = int(cfg.model.vocab_size)
    if not torch.cuda.is_available():
        return n, "skipped(no_cuda)"

    try:
        m = model.to(torch.device("cuda")).eval()
        x = torch.randint(0, vocab, (2, 128), device="cuda")
        with torch.inference_mode():
            y = m(x)
        ok = tuple(y.shape) == (2, 128, vocab)
        if ok:
            return n, "ok"
        raise RuntimeError(f"bad_shape {tuple(y.shape)} expected (2,128,{vocab})")
    except Exception as e:
        raise RuntimeError(f"CUDA smoke failed ({e})") from e


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "pairs",
    nargs=-1,
    required=True,
    type=str,
)
@click.option(
    "--baseline",
    "baseline_pair",
    type=str,
    default=None,
    help="LABEL=PATH for baseline row; adds delta_vs_<label>_pct column vs that row's params.",
)
@click.option(
    "--cwd",
    type=click.Path(path_type=str),
    default=str(PROJECT_ROOT),
    help="Working directory so relative YAML paths resolve (default: mamba_multi_scale_get_started/).",
)
@click.option(
    "--cuda-smoke",
    is_flag=True,
    default=False,
    help="After counting, run logits = model(randint(B=2,L=128)) on CUDA when available.",
)
def main(pairs: tuple[str, ...], baseline_pair: str | None, cwd: str, cuda_smoke: bool) -> None:
    """
    PAIRS: ARCH=configs/...\n\b
    ARCH is a label; configs path is relative to --cwd unless absolute.

    Examples:

      python count_lm_params.py \\
        mamba2_depth4=configs/TinyShakespeare/tiny_shakespeare_mamba2_depth4.yaml \\
        ms_gated_right_init_equal_param=configs/TinyShakespeare/tiny_shakespeare_ms_gated_stride2_right_init_equal_param.yaml \\
        --baseline mamba2_depth4=configs/TinyShakespeare/tiny_shakespeare_mamba2_depth4.yaml
    """
    root = Path(cwd).resolve()

    def load_pair(pair: str) -> tuple[str, Path]:
        if "=" not in pair:
            raise click.BadParameter(f"expected LABEL=PATH, got {pair!r}")
        label, rel = pair.split("=", 1)
        label, rel = label.strip(), rel.strip()
        if not label or not rel:
            raise click.BadParameter(f"expected LABEL=PATH, got {pair!r}")
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise click.BadArgumentUsage(f"missing config file: {path}")
        return label, path

    rows: list[tuple[str, int, str]] = []
    for p in pairs:
        label, path = load_pair(p)
        cfg = load_config(str(path), [])
        n, sm = _count_params(cfg, cuda_smoke=cuda_smoke)
        rows.append((label, n, sm))

    baseline_label: str | None = None
    baseline_params: int | None = None
    if baseline_pair:
        baseline_label, path = load_pair(baseline_pair)
        cfg_b = load_config(str(path), [])
        baseline_params, _bs = _count_params(cfg_b, cuda_smoke=cuda_smoke)
        if baseline_label not in {r[0] for r in rows}:
            rows.insert(0, (baseline_label, baseline_params, ""))

    if cuda_smoke:
        print("| arch_name | trainable_params | delta_vs_ref_pct | ref | cuda_smoke |")
        print("|---|---:|---:|---|---|")
    else:
        print("| arch_name | trainable_params | delta_vs_ref_pct | ref |")
        print("|---|---:|---:|---|")
    for label, n, sm in rows:
        if baseline_params is None or baseline_params == 0:
            delta_str = ""
        else:
            if label == baseline_label:
                pct = 0.0
            else:
                pct = 100.0 * float(n - baseline_params) / float(baseline_params)
            delta_str = f"{pct:+.4f}%"
        ref_col = baseline_label if baseline_label else ""
        if cuda_smoke:
            print(f"| {label} | {n} | {delta_str} | {ref_col} | {sm or '—'} |")
        else:
            print(f"| {label} | {n} | {delta_str} | {ref_col} |")


if __name__ == "__main__":
    main()
