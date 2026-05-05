#!/usr/bin/env python3
"""Run a timing grid: parallel workers each invoke code/benchmark_timing.py (one GPU per worker)."""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from itertools import product
from pathlib import Path

import click
import torch
import yaml
from tqdm import tqdm

# repo root: runners/tiny_shakespeare/py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from benchmark_timing import FIELDNAMES as _BENCH_FIELDNAMES  # noqa: E402


def _worker_gpu_ids() -> list[int]:
    raw = os.environ.get("GRID_GPU_IDS", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    n = torch.cuda.device_count()
    return list(range(max(n, 0)))


def init_worker(gpu_queue: mp.Queue | None) -> None:
    if gpu_queue is None:
        return
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} CUDA_VISIBLE_DEVICES={gpu_id} (benchmark: --set cuda_device=0)")


def _config_path_for_benchmark(config_path: str | Path) -> str:
    """Path string relative to repo root when possible (matches benchmark_timing resolution)."""
    p = Path(config_path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _task_slug(config_path: str | Path, block_size: int, batch_size: int) -> str:
    stem = Path(config_path).stem
    return f"{stem}__b{block_size}__bs{batch_size}"


def run_benchmark(
    config_path: str,
    block_size: int,
    batch_size: int,
    out_csv: Path,
    log_dir: Path,
    warmup_steps: int,
    measure_steps: int,
    use_gpu: bool,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg_arg = _config_path_for_benchmark(config_path)
    try:
        out_rel = str(out_csv.resolve().relative_to(ROOT))
    except ValueError:
        out_rel = str(out_csv.resolve())

    cmd: list[str] = [
        sys.executable,
        str(ROOT / "code" / "benchmark_timing.py"),
        "--configs",
        cfg_arg,
        "--block-sizes",
        str(int(block_size)),
        "--batch-sizes",
        str(int(batch_size)),
        "--warmup-steps",
        str(int(warmup_steps)),
        "--measure-steps",
        str(int(measure_steps)),
        "--output-csv",
        out_rel,
        "--overwrite",
    ]
    if use_gpu:
        cmd.extend(["--set", "cuda_device=0"])
    else:
        cmd.extend(["--set", "device=cpu"])

    log_file = log_dir / "run.log"
    with log_file.open("w", encoding="utf-8") as lf:
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
        )


def call_task(task: partial) -> None:
    task()


def merge_partial_csvs(part_paths: list[Path], merged_path: Path) -> int:
    """Concatenate partial CSVs (same schema as benchmark_timing). Returns row count excluding header."""
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    paths = sorted(p for p in part_paths if p.is_file() and p.stat().st_size > 0)
    if not paths:
        with merged_path.open("w", newline="", encoding="utf-8") as out:
            csv.DictWriter(out, fieldnames=_BENCH_FIELDNAMES).writeheader()
        return 0

    with merged_path.open("w", newline="", encoding="utf-8") as out_f:
        writer: csv.DictWriter | None = None
        fld_final: list[str] = []
        for p in paths:
            with p.open(newline="", encoding="utf-8") as in_f:
                r = csv.DictReader(in_f)
                fld = list(r.fieldnames or _BENCH_FIELDNAMES)
                if writer is None:
                    fld_final = fld
                    writer = csv.DictWriter(out_f, fieldnames=fld_final, extrasaction="ignore")
                    writer.writeheader()
                for row in r:
                    writer.writerow({k: row.get(k, "") for k in fld_final})
                    rows_written += 1
    return rows_written


def run_timing_grid(grid: dict, *, grid_label: str) -> Path:
    configs = grid["configs"]
    block_sizes = grid["block_size"]
    batch_sizes = grid["batch_size"]
    warmup = int(grid.get("warmup_steps", 20))
    measure = int(grid.get("measure_steps", 100))
    merged_rel = grid.get(
        "merged_csv",
        f"outputs/timing/merged_{grid_label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
    )
    merged_path = (ROOT / str(merged_rel)).resolve() if not Path(str(merged_rel)).is_absolute() else Path(merged_rel)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts_dir = ROOT / "outputs" / "timing" / "grid_parts" / f"{grid_label}_{ts}"

    tasks: list[partial] = []
    for config, block_size, batch_sz in product(configs, block_sizes, batch_sizes):
        slug = _task_slug(config, int(block_size), int(batch_sz))
        part_csv = parts_dir / f"{slug}.csv"
        log_dir = ROOT / "logs" / "timing_grid" / f"{grid_label}" / slug
        tasks.append(
            partial(
                run_benchmark,
                str(Path(config).resolve()),
                int(block_size),
                int(batch_sz),
                part_csv,
                log_dir,
                warmup,
                measure,
                torch.cuda.is_available(),
            )
        )

    if not tasks:
        merge_partial_csvs([], merged_path)
        return merged_path

    use_gpu = torch.cuda.is_available()
    gpu_ids = _worker_gpu_ids() if use_gpu else []
    max_workers_env = os.environ.get("GRID_NUM_GPUS", "").strip()
    if max_workers_env:
        cap = int(max_workers_env)
        gpu_ids = gpu_ids[:cap]
    n_workers = min(len(gpu_ids), len(tasks)) if use_gpu else 1

    mp.set_start_method("spawn", force=True)

    if not use_gpu or n_workers < 1:
        for t in tqdm(tasks, desc="Timing grid (CPU)"):
            t()
    else:
        gpu_queue: mp.Queue = mp.Queue()
        for gid in gpu_ids[:n_workers]:
            gpu_queue.put(gid)
        with mp.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(gpu_queue,),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(call_task, tasks),
                total=len(tasks),
                desc="Timing grid",
            ):
                pass

    part_files = list(parts_dir.glob("*.csv"))
    n_rows = merge_partial_csvs(part_files, merged_path)
    print(f"Merged {len(part_files)} partial CSVs → {merged_path} ({n_rows} data rows)")
    return merged_path


@click.command()
@click.option(
    "--grid-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def main(grid_config: Path) -> None:
    with grid_config.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    label = grid_config.stem
    run_timing_grid(data["grid"], grid_label=label)


if __name__ == "__main__":
    main()
