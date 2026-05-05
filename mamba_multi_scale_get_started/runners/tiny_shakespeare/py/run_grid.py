#!/usr/bin/env python3
"""Run a small hyperparameter grid by spawning train_lm.py workers (one GPU each)."""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from functools import partial
from itertools import product
from pathlib import Path

import click
import torch
import yaml
from tqdm import tqdm

# repo root: runners/tiny_shakespeare/py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]


def _worker_gpu_ids() -> list[int]:
    """Physical GPU indices for pool workers (env GRID_GPU_IDS=1,2,3 or unset → 0..device_count-1)."""
    raw = os.environ.get("GRID_GPU_IDS", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    n = torch.cuda.device_count()
    return list(range(max(n, 0)))


def init_worker(gpu_queue: mp.Queue | None) -> None:
    """Pin this worker to a single physical GPU via CUDA_VISIBLE_DEVICES."""
    if gpu_queue is None:
        return
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} CUDA_VISIBLE_DEVICES={gpu_id} (use cuda_device=0 in train_lm)")


def generate_name(config: str | Path, block_size: int, stride: int) -> str:
    cfg_path = Path(config)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    exp_name = data["experiment"]["name"]
    return f"{exp_name}_block_size_{block_size}_stride_{stride}"


def run_training(config_path: str, exp_name: str, block_size: int, stride: int, log_dir: Path, use_gpu: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    overrides = [
        f"data.block_size={block_size}",
        f"experiment.name={exp_name}",
        f"model.multiscale.stride={stride}",
    ]
    if use_gpu:
        # After CUDA_VISIBLE_DEVICES is set to a single GPU, it is always device index 0.
        overrides.append("cuda_device=0")
    else:
        overrides.append("device=cpu")

    cmd = [
        sys.executable,
        str(ROOT / "code" / "train_lm.py"),
        "--config",
        str(Path(config_path).resolve()),
        *overrides,
    ]
    with log_file.open("w", encoding="utf-8") as lf:
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
        )


def call_task(task) -> None:
    task()


def run_grid(grid: dict) -> None:
    configs = grid["configs"]
    block_sizes = grid["block_size"]
    strides = grid["stride"]
    tasks: list[partial] = []
    for config, block_size, stride in product(configs, block_sizes, strides):
        exp_name = generate_name(config, block_size, stride)
        log_path = ROOT / "logs" / "TinyShakespeare" / exp_name
        tasks.append(
            partial(
                run_training,
                str(config),
                exp_name,
                int(block_size),
                int(stride),
                log_path,
                use_gpu=torch.cuda.is_available(),
            )
        )

    if not tasks:
        return

    use_gpu = torch.cuda.is_available()
    gpu_ids = _worker_gpu_ids() if use_gpu else []
    max_workers_env = os.environ.get("GRID_NUM_GPUS", "").strip()
    if max_workers_env:
        cap = int(max_workers_env)
        gpu_ids = gpu_ids[:cap]
    n_workers = min(len(gpu_ids), len(tasks)) if use_gpu else 1

    mp.set_start_method("spawn", force=True)

    if not use_gpu or n_workers < 1:
        for t in tqdm(tasks, desc="Running grid (CPU)"):
            t()
        return

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
            desc="Running grid",
        ):
            pass


@click.command()
@click.option("--grid-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def main(grid_config: Path) -> None:
    with grid_config.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    run_grid(data["grid"])


if __name__ == "__main__":
    main()
