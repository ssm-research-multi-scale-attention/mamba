#!/usr/bin/env python3
"""
MQAR failure-point sweep: Mamba2 split-TS, single-TS baselines (slow/wide), MS gated, Transformer.

Does not touch outputs/ArchEval. Writes runs under outputs/MQARFailureSweep/
and logs under logs/mqar_failure_sweep/.

Example:
  NUM_GPUS=8 python runners/mqar/py/run_mqar_failure_sweep_best.py --num-gpus 8

Only new Mamba2 baseline runs (skip dirs that already finished ok):
  NUM_GPUS=8 python runners/mqar/py/run_mqar_failure_sweep_best.py --num-gpus 8 \\
    --skip-completed --models mamba2_slow_init,mamba2_wide_init

Only missing transformer runs (completed = meta_metrics.csv with status=ok):
  NUM_GPUS=8 python runners/mqar/py/run_mqar_failure_sweep_best.py --num-gpus 8 \\
    --skip-completed --models transformer_param_match

Slow-biased ablations with vocab subset:
  NUM_GPUS=8 python runners/mqar/py/run_mqar_failure_sweep_best.py --num-gpus 8 --skip-completed \\
    --models mamba2_split_ts_slow_default,mamba2_split_ts_slow_veryslow,ms_gated_slow_init \\
    --vocabs 512,768,896,1024,1280
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_MQAR = ROOT / "code" / "train_mqar.py"
ANALYZE = ROOT / "code" / "analyze_mqar_failure_sweep.py"
OUT_SWEEP = ROOT / "outputs" / "MQARFailureSweep"

MODELS: list[tuple[str, Path]] = [
    (
        "mamba2_split_ts_50_50",
        ROOT / "configs" / "MQAR" / "mqar_mamba2_split_ts_50_50_len128.yaml",
    ),
    (
        "ms_gated_right_init",
        ROOT / "configs" / "MQAR" / "mqar_ms_gated_stride2_len128_right_init.yaml",
    ),
    (
        "transformer_param_match",
        ROOT / "configs" / "MQAR" / "mqar_transformer_param_match_len128.yaml",
    ),
    (
        "mamba2_slow_init",
        ROOT / "configs" / "MQAR" / "mqar_mamba2_slow_init_len128.yaml",
    ),
    (
        "mamba2_wide_init",
        ROOT / "configs" / "MQAR" / "mqar_mamba2_wide_init_len128.yaml",
    ),
    (
        "mamba2_split_ts_slow_default",
        ROOT / "configs" / "MQAR" / "mqar_mamba2_split_ts_slow_default_len128.yaml",
    ),
    (
        "mamba2_split_ts_slow_veryslow",
        ROOT / "configs" / "MQAR" / "mqar_mamba2_split_ts_slow_veryslow_len128.yaml",
    ),
    (
        "ms_gated_slow_init",
        ROOT / "configs" / "MQAR" / "mqar_ms_gated_slow_init_len128.yaml",
    ),
]
VOCAB_SIZES = [512, 640, 704, 768, 896, 1024, 1280, 1536, 2048]
SEEDS = [42, 43, 44]
FULL_SWEEP_JOB_COUNT = len(MODELS) * len(VOCAB_SIZES) * len(SEEDS)


def _job_out_dir(exp_name: str) -> Path:
    return OUT_SWEEP / exp_name


def _is_job_completed_ok(exp_name: str) -> bool:
    """True if a prior run finished and wrote meta_metrics with status ok (do not re-run)."""
    meta = _job_out_dir(exp_name) / "meta_metrics.csv"
    if not meta.is_file():
        return False
    try:
        with meta.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return False
    if not rows:
        return False
    return rows[0].get("status", "").strip() == "ok"


def _build_jobs(vocab_sizes: list[int]) -> list[dict[str, str | int]]:
    jobs: list[dict[str, str | int]] = []
    for model_name, cfg_path in MODELS:
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing MQAR config: {cfg_path}")
        for vs in vocab_sizes:
            for seed in SEEDS:
                exp = f"mqar_fail_{model_name}_v{vs}_seed{seed}"
                jobs.append(
                    {
                        "model_name": model_name,
                        "config": str(cfg_path),
                        "vocab_size": vs,
                        "seed": seed,
                        "experiment_name": exp,
                    }
                )
    return jobs


def _job_cmd(job: dict[str, str | int]) -> list[str]:
    exp = str(job["experiment_name"])
    vs = int(job["vocab_size"])
    seed = int(job["seed"])
    return [
        sys.executable,
        str(TRAIN_MQAR),
        "--config",
        str(job["config"]),
        f"experiment.name={exp}",
        f"logging.output_dir=outputs/MQARFailureSweep/{exp}",
        f"data.vocab_size={vs}",
        f"experiment.seed={seed}",
        "data.input_seq_len=128",
        "data.num_kv_pairs=16",
        "data.train_examples=20000",
        "data.val_examples=2000",
        "data.test_examples=2000",
        "data.fixed_examples=true",
        "data.min_query_pos=null",
        "loader.num_workers=0",
        "loader.pin_memory=false",
        "device=cuda:0",
        "cuda_device=0",
    ]


def _run_wave(wave: list[dict[str, str | int]], log_root: Path) -> int:
    """Run jobs in parallel; wave length should be <= num_gpus. Returns 0 if all ok."""
    log_root.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    log_files: list[object] = []
    try:
        for slot, job in enumerate(wave):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
            log_path = log_root / f"{job['experiment_name']}.log"
            logf = log_path.open("w", encoding="utf-8")
            log_files.append(logf)
            procs.append(
                subprocess.Popen(
                    _job_cmd(job),
                    cwd=str(ROOT),
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            )
        rc = 0
        for p, job in zip(procs, wave, strict=True):
            p.wait()
            if p.returncode != 0:
                print(
                    f"[error] job failed rc={p.returncode} experiment={job['experiment_name']}",
                    file=sys.stderr,
                )
                rc = p.returncode or 1
    finally:
        for f in log_files:
            try:
                f.close()
            except OSError:
                pass
    return rc


def main() -> None:
    # Align with shell runners: avoid CPU oversubscription on multi-GPU hosts.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(
        description=(
            "MQAR vocab failure sweep (Mamba2 split + slow/wide single-TS baselines, MS gated, Transformer)."
        )
    )
    ap.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "8")),
        help="Parallel jobs per wave (default: env NUM_GPUS or 8).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and exit (no training).",
    )
    ap.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Do not run analyze_mqar_failure_sweep.py after jobs.",
    )
    ap.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip jobs whose output dir already has meta_metrics.csv with status=ok (no overwrite).",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model_name subset (e.g. transformer_param_match). Default: all.",
    )
    ap.add_argument(
        "--vocabs",
        type=str,
        default="",
        help=(
            "Comma-separated vocab sizes (e.g. 512,768,1024). "
            f"Default: full list ({len(VOCAB_SIZES)} values)."
        ),
    )
    args = ap.parse_args()
    num_gpus = max(1, int(args.num_gpus))
    if args.vocabs.strip():
        vocab_sizes = [int(x.strip()) for x in args.vocabs.split(",") if x.strip()]
        if not vocab_sizes:
            raise ValueError("--vocabs must list at least one integer.")
        for v in vocab_sizes:
            if v < 2:
                raise ValueError(f"Invalid vocab_size={v!r}")
    else:
        vocab_sizes = list(VOCAB_SIZES)
    jobs = _build_jobs(vocab_sizes)
    if args.models.strip():
        allow = {m.strip() for m in args.models.split(",") if m.strip()}
        known = {name for name, _ in MODELS}
        unknown = allow - known
        if unknown:
            raise ValueError(f"--models unknown: {sorted(unknown)}; known: {sorted(known)}")
        jobs = [j for j in jobs if j["model_name"] in allow]

    if args.dry_run:
        todo = [j for j in jobs if not (args.skip_completed and _is_job_completed_ok(j["experiment_name"]))]
        skip_n = len(jobs) - len(todo)
        print(
            f"dry-run: {len(jobs)} jobs in selection "
            f"(full default: {FULL_SWEEP_JOB_COUNT} = {len(MODELS)} models × "
            f"{len(VOCAB_SIZES)} vocabs × {len(SEEDS)} seeds)"
        )
        if args.skip_completed:
            print(f"  with --skip-completed: would run {len(todo)}, skip {skip_n} already ok")
        for j in jobs:
            mark = ""
            if args.skip_completed and _is_job_completed_ok(j["experiment_name"]):
                mark = "  [skip: completed ok]"
            print(
                f"  {j['experiment_name']} model={j['model_name']} "
                f"vocab={j['vocab_size']} seed={j['seed']}{mark}"
            )
        return

    if not TRAIN_MQAR.is_file():
        raise FileNotFoundError(f"train_mqar.py not found: {TRAIN_MQAR}")

    pending = [j for j in jobs if not (args.skip_completed and _is_job_completed_ok(j["experiment_name"]))]
    if args.skip_completed and len(pending) < len(jobs):
        print(
            f"skip-completed: running {len(pending)} jobs, skipping {len(jobs) - len(pending)} with status=ok"
        )
    if not pending:
        print("No pending jobs. Run analyze only (--skip-analyze to skip).")
        if not args.skip_analyze and ANALYZE.is_file():
            subprocess.run(
                [sys.executable, str(ANALYZE), "--root", str(ROOT / "outputs" / "MQARFailureSweep")],
                check=False,
            )
        return

    log_root = ROOT / "logs" / "mqar_failure_sweep"
    failed = False
    for start in range(0, len(pending), num_gpus):
        wave = pending[start : start + num_gpus]
        print(f"========== wave jobs [{start}, {start + len(wave)}) / {len(pending)} ==========")
        rc = _run_wave(wave, log_root)
        if rc != 0:
            failed = True

    if not args.skip_analyze:
        if not ANALYZE.is_file():
            raise FileNotFoundError(str(ANALYZE))
        ar = subprocess.run([sys.executable, str(ANALYZE), "--root", str(ROOT / "outputs" / "MQARFailureSweep")])
        if ar.returncode != 0:
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
