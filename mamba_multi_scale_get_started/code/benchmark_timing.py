#!/usr/bin/env python3
"""Benchmark LM forward pass latency / throughput (random init, no checkpoint, no dataloader)."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from exp_config import load_config, set_seed  # noqa: E402
from models.language_models import build_lm  # noqa: E402

PROJECT_ROOT = _CODE.parent


def _resolve_lm_device(cfg: DictConfig) -> torch.device:
    raw = str(OmegaConf.select(cfg, "device", default="auto")).strip()
    rl = raw.lower()
    cuda_idx_cfg = int(OmegaConf.select(cfg, "cuda_device", default=0))

    if rl in ("auto", ""):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        dev = torch.device(f"cuda:{cuda_idx_cfg}")
        torch.cuda.set_device(dev)
        return dev

    if rl == "cpu":
        return torch.device("cpu")

    if ":" in raw and raw.lower().startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={raw!r} but CUDA is not available.")
        dev = torch.device(raw)
        torch.cuda.set_device(dev)
        return dev

    if rl in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={raw!r} but CUDA is not available.")
        dev = torch.device(f"cuda:{cuda_idx_cfg}")
        torch.cuda.set_device(dev)
        return dev

    raise ValueError(
        f"Unknown device={raw!r}. Use auto, cpu, cuda, gpu, or cuda:N. "
        f"For a physical index with cuda/gpu, set cuda_device (int), default 0."
    )


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def _meta_row(cfg: DictConfig, config_path: Path) -> dict[str, Any]:
    backbone = str(OmegaConf.select(cfg, "model.backbone", default="")).strip()
    fusion = OmegaConf.select(cfg, "model.multiscale.fusion", default="")
    fusion_s = str(fusion).strip() if fusion is not None else ""
    stride = OmegaConf.select(cfg, "model.multiscale.stride", default="")
    stride_s = str(int(stride)) if stride not in (None, "") else ""
    return {
        "model_name": str(cfg.experiment.name),
        "experiment_name": str(cfg.experiment.name),
        "config_path": str(config_path.resolve().relative_to(PROJECT_ROOT)),
        "backbone": backbone,
        "fusion": fusion_s,
        "stride": stride_s,
    }


def _oom_row(
    base: dict[str, Any],
    block_size: int,
    batch_size_requested: int,
    batch_size_effective: str | int,
    device: torch.device,
    cuda_name: str,
) -> dict[str, Any]:
    row = {**base}
    row.update(
        {
            "block_size": block_size,
            "batch_size": batch_size_effective,
            "batch_size_requested": batch_size_requested,
            "batch_size_effective": batch_size_effective,
            "vocab_size": "",
            "d_model": "",
            "num_parameters_trainable": "",
            "device": str(device),
            "cuda_device_name": cuda_name,
            "forward_latency_ms_mean": "",
            "forward_latency_ms_std": "",
            "forward_latency_ms_p50": "",
            "forward_latency_ms_p95": "",
            "tokens_per_second": "",
            "max_cuda_memory_mb": "",
            "status": "oom",
        }
    )
    return row


def _forward_timings(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[list[float], float]:
    """Returns (latencies_ms, max_cuda_memory_peak_mb). CPU: peak 0."""
    is_cuda = device.type == "cuda"
    latencies: list[float] = []
    peak_mb = 0.0

    with torch.inference_mode():
        model.eval()
        for _ in range(warmup_steps):
            if is_cuda:
                torch.cuda.synchronize()
            _ = model(input_ids)
            if is_cuda:
                torch.cuda.synchronize()

        for _ in range(measure_steps):
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(input_ids)
            if is_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            latencies.append(dt_ms)
            if is_cuda:
                peak_mb = max(peak_mb, torch.cuda.max_memory_allocated(device) / (1024.0**2))

    return latencies, peak_mb


def _run_one_config_batch(
    cfg: DictConfig,
    config_path: Path,
    block_size: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    base_meta: dict[str, Any],
    device: torch.device,
    cuda_name: str,
) -> dict[str, Any]:
    vocab_size = int(cfg.model.vocab_size)
    d_model = int(cfg.model.d_model)
    seed = int(cfg.experiment.seed)
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg2.model.vocab_size = vocab_size

    model = build_lm(cfg2).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    high = max(vocab_size, 1)
    gen_seed = seed + block_size * 10_007 + batch_size * 100_003
    gen = torch.Generator(device=device)
    gen.manual_seed(gen_seed)
    t_ids = torch.randint(0, high, (batch_size, block_size), device=device, dtype=torch.long, generator=gen)

    lat_ms, peak_cuda_mb = _forward_timings(model, t_ids, device, warmup_steps, measure_steps)
    mean_s = statistics.mean(lat_ms) / 1000.0
    std_ms = statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0.0
    srt = sorted(lat_ms)
    tokens_per_s = (batch_size * block_size) / mean_s if mean_s > 0 else float("inf")

    row = {**base_meta}
    row.update(
        {
            "block_size": block_size,
            "batch_size": batch_size,
            "batch_size_requested": batch_size,
            "batch_size_effective": batch_size,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_parameters_trainable": n_params,
            "device": str(device),
            "cuda_device_name": cuda_name,
            "forward_latency_ms_mean": round(statistics.mean(lat_ms), 6),
            "forward_latency_ms_std": round(std_ms, 6),
            "forward_latency_ms_p50": round(_percentile(srt, 50), 6),
            "forward_latency_ms_p95": round(_percentile(srt, 95), 6),
            "tokens_per_second": round(tokens_per_s, 4),
            "max_cuda_memory_mb": round(peak_cuda_mb, 4) if device.type == "cuda" else "",
            "status": "ok",
        }
    )
    return row


def _try_benchmark(
    cfg: DictConfig,
    config_path: Path,
    block_size: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    cuda_name: str,
) -> dict[str, Any]:
    base_meta = _meta_row(cfg, Path(config_path))
    return _run_one_config_batch(
        cfg, Path(config_path), block_size, batch_size, warmup_steps, measure_steps, base_meta, device, cuda_name
    )


FIELDNAMES = [
    "model_name",
    "experiment_name",
    "config_path",
    "backbone",
    "fusion",
    "stride",
    "block_size",
    "batch_size",
    "batch_size_requested",
    "batch_size_effective",
    "vocab_size",
    "d_model",
    "num_parameters_trainable",
    "device",
    "cuda_device_name",
    "forward_latency_ms_mean",
    "forward_latency_ms_std",
    "forward_latency_ms_p50",
    "forward_latency_ms_p95",
    "tokens_per_second",
    "max_cuda_memory_mb",
    "status",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, type=str, help="YAML configs (paths relative ok).")
    p.add_argument("--block-sizes", nargs="+", type=int, required=True)
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--measure-steps", type=int, default=100)
    p.add_argument(
        "--output-csv",
        type=str,
        default="outputs/timing/timing_lm.csv",
        help="Relative to repo root unless absolute.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate output CSV before writing; default append.",
    )
    p.add_argument(
        "--set",
        dest="cfg_overrides",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="OmegaConf CLI override (repeatable), e.g. --set device=cpu",
    )
    args = p.parse_args()

    out_path = Path(args.output_csv)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    append = not args.overwrite and out_path.exists()
    csv_mode = "w" if (args.overwrite or not append) else "a"

    with out_path.open(csv_mode, newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if csv_mode == "w" or not append:
            writer.writeheader()

        for cfg_path_str in args.configs:
            cfg_path = Path(cfg_path_str)
            if not cfg_path.is_absolute():
                cfg_path = (PROJECT_ROOT / cfg_path).resolve()
            cfg = load_config(str(cfg_path), list(args.cfg_overrides or []))
            device = _resolve_lm_device(cfg)
            set_seed(int(cfg.experiment.seed))
            cuda_name = ""
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    cuda_name = str(torch.cuda.get_device_name(device.index or 0))
                except Exception:
                    cuda_name = ""

            base_meta = _meta_row(cfg, cfg_path)

            def _attempt(batch_sz: int) -> dict[str, Any]:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return _try_benchmark(
                    cfg,
                    cfg_path,
                    block_size,
                    batch_sz,
                    args.warmup_steps,
                    args.measure_steps,
                    device,
                    cuda_name,
                )

            for block_size in args.block_sizes:
                for requested_bs in args.batch_sizes:
                    candidates = [requested_bs]
                    if requested_bs == 16:
                        candidates = [16, 8]
                    succeeded = False
                    for cand in candidates:
                        try:
                            row = _attempt(cand)
                            row["batch_size_requested"] = requested_bs
                            row["batch_size_effective"] = cand
                            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
                            out_f.flush()
                            succeeded = True
                            break
                        except RuntimeError as e:
                            if "out of memory" not in str(e).lower():
                                raise
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                    if not succeeded:
                        wr = _oom_row(
                            base_meta,
                            block_size,
                            requested_bs,
                            "",
                            device,
                            cuda_name,
                        )
                        writer.writerow({k: wr.get(k, "") for k in FIELDNAMES})
                        out_f.flush()

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
