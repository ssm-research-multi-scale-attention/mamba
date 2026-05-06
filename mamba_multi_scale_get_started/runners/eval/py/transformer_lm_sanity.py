#!/usr/bin/env python3
"""
Parallel hyperparameter sanity sweep for TinyShakespeare transformer_lm.

Runs:
  lr in {1e-4, 3e-4, 1e-3}
  dropout in {0.0, 0.1}

Parallelism:
  one process per GPU via CUDA_VISIBLE_DEVICES=<gpu>
  inside each job: device=cuda:0 cuda_device=0

Example:
  python runners/eval/run_transformer_lm_sanity.py --num-gpus 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "code" / "train_lm.py").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {p}")


def fmt_float_for_name(x: str) -> str:
    return x.replace("-", "m").replace(".", "p").replace("+", "").replace("e", "e")


def run_job(
    *,
    root: Path,
    python_bin: str,
    gpu_id: int,
    cfg: str,
    lr: str,
    dropout: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[str, int]:
    name = f"sanity_pm_lr{fmt_float_for_name(lr)}_do{fmt_float_for_name(dropout)}_seed{seed}"
    out_dir = root / "outputs" / "TransformerLMSanity" / name
    log_dir = root / "logs" / "transformer_lm_sanity"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{name}.log"

    cmd = [
        python_bin,
        "code/train_lm.py",
        "--config",
        str(root / cfg),
        f"experiment.name={name}",
        f"logging.output_dir={out_dir}",
        f"train.lr={lr}",
        f"model.transformer.dropout={dropout}",
        f"train.epochs={epochs}",
        f"train.early_stopping.patience={patience}",
        f"experiment.seed={seed}",
        f"loader.batch_size={batch_size}",
        "loader.num_workers=0",
        "cuda_device=0",
        "device=cuda:0",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    header = (
        f"======== {name} ========\n"
        f"gpu={gpu_id} lr={lr} dropout={dropout} seed={seed}\n"
        f"cmd: {' '.join(cmd)}\n\n"
    )

    with log_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()

        rc = proc.wait()
        elapsed = time.time() - start

        f.write(f"\n======== finished {name} rc={rc} elapsed_sec={elapsed:.1f} ========\n")

    return name, rc


def main() -> int:
    root = repo_root()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "8")))
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--config", default="configs/TinyShakespeare/tiny_shakespeare_transformer_param_match.yaml")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "20")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", "5")))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lrs", nargs="+", default=["1e-4", "3e-4", "1e-3"])
    parser.add_argument("--dropouts", nargs="+", default=["0.0", "0.1"])
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    if args.num_gpus <= 0:
        raise SystemExit("--num-gpus must be > 0")

    (root / "outputs" / "TransformerLMSanity").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "transformer_lm_sanity").mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    for lr in args.lrs:
        for dropout in args.dropouts:
            jobs.append(
                {
                    "lr": lr,
                    "dropout": dropout,
                }
            )

    print(f"root: {root}")
    print(f"jobs: {len(jobs)}")
    print(f"num_gpus: {args.num_gpus}")
    print(f"python: {args.python}")

    failures: list[tuple[str, int]] = []

    with ThreadPoolExecutor(max_workers=args.num_gpus) as ex:
        futures = []
        for idx, job in enumerate(jobs):
            gpu_id = idx % args.num_gpus
            futures.append(
                ex.submit(
                    run_job,
                    root=root,
                    python_bin=args.python,
                    gpu_id=gpu_id,
                    cfg=args.config,
                    lr=job["lr"],
                    dropout=job["dropout"],
                    seed=args.seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
            )

        for fut in as_completed(futures):
            name, rc = fut.result()
            if rc == 0:
                print(f"[OK] {name}")
            else:
                print(f"[FAIL rc={rc}] {name}")
                failures.append((name, rc))

    if not args.skip_aggregate:
        csv_out = root / "outputs" / "TransformerLMSanity" / "summary_transformer_lm_sanity.csv"
        agg_cmd = [
            args.python,
            "code/aggregate_transformer_lm_sanity.py",
            "--root",
            str(root / "outputs" / "TransformerLMSanity"),
            "--csv-out",
            str(csv_out),
        ]

        print("======== aggregate ========")
        print(" ".join(agg_cmd))
        rc = subprocess.call(agg_cmd, cwd=root)
        if rc != 0:
            print(f"[WARN] aggregate failed rc={rc}")
        else:
            print(f"Summary: {csv_out}")

    if failures:
        print("\nFailures:")
        for name, rc in failures:
            print(f"  {name}: rc={rc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
