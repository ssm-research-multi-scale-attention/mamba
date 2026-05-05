"""Time training vs evaluation phases (CUDA events when available, else wall clock).

LM forward latency / throughput sweeps: see ``benchmark_timing.py``.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def timed_phase(run: Callable[[], T]) -> tuple[T, float, str]:
    """
    Run ``run()`` and return ``(return_value, seconds, method)``.

    - **CUDA**: ``cuda_event`` — elapsed between CUDA events on the default stream (good for
      comparing GPU-bound runs on the same device).
    - **No CUDA**: ``wall_clock`` — ``time.perf_counter`` around the phase (CPU / MPS / etc.).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = run()
        e1.record()
        torch.cuda.synchronize()
        sec = float(e0.elapsed_time(e1)) / 1000.0
        return out, sec, "cuda_event"
    t0 = time.perf_counter()
    out = run()
    return out, float(time.perf_counter() - t0), "wall_clock"


def collect_run_timing_metadata() -> dict[str, str]:
    """Optional device strings for metrics (empty if no CUDA)."""
    meta: dict[str, str] = {}
    if torch.cuda.is_available():
        meta["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        meta["cuda_device_index"] = str(torch.cuda.current_device())
    return meta
