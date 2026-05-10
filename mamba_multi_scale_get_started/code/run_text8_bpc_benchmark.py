#!/usr/bin/env python3
"""Parallel Text8 BPC benchmark launcher (multi-GPU, multi-seed).

Calls ``code/train_lm.py`` with OmegaConf overrides. Important:
- Outputs go to ``logging.output_dir`` (not ``experiment.output_dir`` — train_lm resolves only ``logging.output_dir``).
- Global seed / dataloader RNG: ``experiment.seed`` only (there is no separate ``train.seed`` / ``data.seed``).

Example (dry-run):

  python code/run_text8_bpc_benchmark.py \\
      --registry configs/EvalRegistry/text8_bpc_core.yaml \\
      --seeds 42,43,44 \\
      --num-gpus 8 \\
      --train-max-steps 20000 \\
      --eval-max-batches 1000 \\
      --num-workers 0 \\
      --skip-completed \\
      --dry-run
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from arch_eval_common import lm_meta_ok, parse_int_list_csv

_CODE = Path(__file__).resolve().parent
ROOT = _CODE.parent
TRAIN_LM = _CODE / "train_lm.py"
ANALYZE = _CODE / "analyze_text8_bpc_benchmark.py"
TEXT8_BPC_ROOT = ROOT / "outputs" / "Text8BPC"


def _experiment_stem(arch_name: str, seed: int) -> str:
    return f"text8_{arch_name}_seed{seed}"


def _out_rel(arch_name: str, seed: int) -> str:
    stem = _experiment_stem(arch_name, seed)
    return f"outputs/Text8BPC/{stem}"


def _parse_registry(path: Path) -> list[dict]:
    cfg = OmegaConf.load(path)
    rows = OmegaConf.select(cfg, "architectures", default=None)
    if not rows:
        raise ValueError(f"No architectures in registry: {path}")
    out: list[dict] = []
    for row in OmegaConf.to_container(rows, resolve=True):  # type: ignore[arg-type]
        if not isinstance(row, dict):
            continue
        arch = row.get("arch_name") or row.get("name")
        arch_type = row.get("arch_type") or row.get("type") or ""
        yaml_path = row.get("text8_config") or row.get("lm_config")
        if not arch or not yaml_path:
            raise ValueError(f"Registry entry missing arch_name or text8_config: {row}")
        out.append({"arch_name": str(arch), "arch_type": str(arch_type), "text8_config": str(yaml_path)})
    return out


@dataclass(frozen=True)
class Job:
    arch_name: str
    arch_type: str
    config_relpath: str
    seed: int


def _build_cmd(
    job: Job,
    *,
    train_max_steps: int,
    eval_max_batches: int,
    num_workers: int,
    early_stop_patience: int,
) -> list[str]:
    stem = _experiment_stem(job.arch_name, job.seed)
    cfg_path = ROOT / job.config_relpath
    return [
        sys.executable,
        str(TRAIN_LM),
        "--config",
        str(cfg_path),
        f"experiment.name={stem}",
        f"logging.output_dir={_out_rel(job.arch_name, job.seed)}",
        f"experiment.seed={job.seed}",
        f"train.max_steps={train_max_steps}",
        f"eval.max_batches={eval_max_batches}",
        f"loader.num_workers={num_workers}",
        f"train.early_stopping.patience={early_stop_patience}",
    ]


def _job_completed_ok(job: Job) -> bool:
    out = ROOT / _out_rel(job.arch_name, job.seed)
    return lm_meta_ok(out / "meta_metrics.csv")


def _gpu_worker(gpu_id: int, q: queue.Queue[Job | None], cmd_opts: dict) -> list[tuple[Job, int]]:
    results: list[tuple[Job, int]] = []
    while True:
        job = q.get()
        try:
            if job is None:
                return results
            cmd = _build_cmd(job, **cmd_opts)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"[gpu{gpu_id}] start {job.arch_name} seed={job.seed}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, env=env)
            results.append((job, r.returncode))
            status = "ok" if r.returncode == 0 else f"fail({r.returncode})"
            print(f"[gpu{gpu_id}] done {job.arch_name} seed={job.seed} {status}", flush=True)
        finally:
            q.task_done()


def _print_dry_run(
    *,
    registry: Path,
    arch_rows: list[dict],
    seeds: list[int],
    all_jobs: list[Job],
    skipped: list[Job],
    scheduled: list[Job],
    preview_n: int,
    cmd_opts: dict,
) -> None:
    print(f"registry: {registry}")
    print(f"models ({len(arch_rows)}): {[r['arch_name'] for r in arch_rows]}")
    print(f"seeds: {seeds}")
    print(f"selected_jobs (arch×seed): {len(all_jobs)}")
    print(f"skipped_jobs: {len(skipped)}")
    print(f"scheduled_jobs: {len(scheduled)}")
    by_m: defaultdict[str, int] = defaultdict(int)
    for j in scheduled:
        by_m[j.arch_name] += 1
    print("scheduled breakdown by arch_name:")
    for k in sorted(by_m.keys()):
        print(f"  {k}: {by_m[k]}")
    print("\ncommand preview (first jobs):")
    for j in scheduled[:preview_n]:
        cmd = _build_cmd(j, **cmd_opts)
        print(" ", " \\\n    ".join(cmd))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=str, default="configs/EvalRegistry/text8_bpc_core.yaml")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--train-max-steps", type=int, default=20000)
    p.add_argument("--eval-max-batches", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Forwarded to train.early_stopping.patience (default 0 so training is not truncated before train.max_steps).",
    )
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help="Do not run analyze_text8_bpc_benchmark.py after jobs finish.",
    )
    p.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Run at most N scheduled jobs (after skip-filter), for smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    reg_arg = Path(args.registry)
    registry = reg_arg.resolve() if reg_arg.is_absolute() else (ROOT / reg_arg).resolve()
    seeds = parse_int_list_csv(args.seeds)
    num_gpus = max(1, int(args.num_gpus))

    cmd_opts = dict(
        train_max_steps=int(args.train_max_steps),
        eval_max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
        early_stop_patience=int(args.early_stopping_patience),
    )

    arch_rows = _parse_registry(registry)
    all_jobs: list[Job] = []
    for row in arch_rows:
        for s in seeds:
            all_jobs.append(
                Job(arch_name=row["arch_name"], arch_type=row["arch_type"], config_relpath=row["text8_config"], seed=s)
            )

    skipped: list[Job] = []
    scheduled: list[Job] = []
    for job in all_jobs:
        if args.skip_completed and _job_completed_ok(job):
            skipped.append(job)
        else:
            scheduled.append(job)

    if args.max_jobs is not None:
        scheduled = scheduled[: max(0, int(args.max_jobs))]

    if args.dry_run:
        _print_dry_run(
            registry=registry,
            arch_rows=arch_rows,
            seeds=seeds,
            all_jobs=all_jobs,
            skipped=skipped,
            scheduled=scheduled,
            preview_n=min(3, len(scheduled)),
            cmd_opts=cmd_opts,
        )
        if not args.no_analyze:
            print("\n(analyzer would run next; suppressed in --dry-run)")
        return

    TEXT8_BPC_ROOT.mkdir(parents=True, exist_ok=True)

    jq: queue.Queue[Job | None] = queue.Queue()
    for job in scheduled:
        jq.put(job)
    for _ in range(num_gpus):
        jq.put(None)

    failures: list[tuple[Job, int]] = []
    lock = threading.Lock()

    def wrap(gpu: int):
        nonlocal failures
        rs = _gpu_worker(gpu, jq, cmd_opts=cmd_opts)
        with lock:
            for job, code in rs:
                if code != 0:
                    failures.append((job, code))

    threads = [threading.Thread(target=wrap, args=(g,)) for g in range(num_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failures:
        print("Failures:", failures, file=sys.stderr)
        sys.exit(1)

    if args.no_analyze:
        return

    try:
        reg_for_ana = str(registry.relative_to(ROOT))
    except ValueError:
        reg_for_ana = str(registry)
    ana_cmd = [
        sys.executable,
        str(ANALYZE),
        "--registry",
        reg_for_ana,
        "--outputs-root",
        "outputs/Text8BPC",
    ]
    print("Running analyzer:", " ".join(ana_cmd), flush=True)
    r = subprocess.run(ana_cmd, cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
