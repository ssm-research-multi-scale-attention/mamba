#!/usr/bin/env python3
"""
Grid search for TinyShakespeare transformer_lm (parallel over GPUs).

Example:
  NUM_GPUS=8 python runners/eval/py/run_transformer_lm_search.py --num-gpus 8
  python runners/eval/py/run_transformer_lm_search.py --dry-run --num-gpus 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ARCHITECTURES: list[tuple[str, int, int, int]] = [
    ("d192_l6_h4", 192, 6, 4),
    ("d232_l4_h4", 232, 4, 4),
    ("d256_l4_h4", 256, 4, 4),
    ("d256_l6_h8", 256, 6, 8),
    ("d384_l4_h8", 384, 4, 8),
]

LEARNING_RATES: list[str] = ["7e-4", "1e-3", "1.5e-3", "2e-3"]
DROPOUTS: list[str] = ["0.0", "0.05"]

BASE_CONFIG = "configs/TinyShakespeare/tiny_shakespeare_transformer_param_match.yaml"

MAMBA_DEPTH4_REF = 1.59202970
MAMBA_DEPTH6_REF = 1.53943708


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        tlm = parent / "code" / "train_lm.py"
        cfg = parent / "configs" / "TinyShakespeare" / "tiny_shakespeare_transformer_param_match.yaml"
        if tlm.is_file() and cfg.is_file():
            return parent
    raise RuntimeError(f"Could not find repo root (train_lm + base config) from {p}")


def _fmt_token(s: str) -> str:
    return (
        s.replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
        .lower()
    )


@dataclass(frozen=True)
class Job:
    run_name: str
    d_model: int
    n_layers: int
    n_heads: int
    dropout: str
    lr: str
    seed: int


def build_jobs(*, seed: int) -> list[Job]:
    jobs: list[Job] = []
    for arch, d, nl, nh in ARCHITECTURES:
        if d % nh != 0:
            raise ValueError(f"Invalid arch {arch}: d_model={d} not divisible by n_heads={nh}")
        for lr in LEARNING_RATES:
            for do in DROPOUTS:
                lr_t = _fmt_token(lr)
                do_t = _fmt_token(do)
                name = f"tsrch_{arch}_lr{lr_t}_do{do_t}_s{seed}"
                jobs.append(
                    Job(
                        run_name=name,
                        d_model=d,
                        n_layers=nl,
                        n_heads=nh,
                        dropout=do,
                        lr=lr,
                        seed=seed,
                    )
                )
    return jobs


def run_one_job(
    *,
    root: Path,
    python_bin: str,
    gpu_id: int,
    job: Job,
    epochs: int,
    patience: int,
    batch_size: int,
    base_cfg: str,
    dry_run: bool,
) -> tuple[str, int, str]:
    """Returns (run_name, exit_code, message)."""
    rel_out = Path("outputs") / "TransformerLMSearch" / job.run_name
    out_dir = root / rel_out
    log_dir = root / "logs" / "transformer_lm_search"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.run_name}.log"

    cmd = [
        python_bin,
        "code/train_lm.py",
        "--config",
        str(root / base_cfg),
        f"experiment.name={job.run_name}",
        f"logging.output_dir={rel_out.as_posix()}",
        f"model.d_model={job.d_model}",
        f"model.transformer.n_layers={job.n_layers}",
        f"model.transformer.n_heads={job.n_heads}",
        f"model.transformer.dropout={job.dropout}",
        f"train.lr={job.lr}",
        f"train.epochs={epochs}",
        f"train.early_stopping.patience={patience}",
        f"experiment.seed={job.seed}",
        f"loader.batch_size={batch_size}",
        "loader.num_workers=0",
        "cuda_device=0",
        "device=cuda:0",
    ]

    header = (
        f"=== {job.run_name} CUDA_VISIBLE_DEVICES={gpu_id} ===\n"
        + " ".join(cmd)
        + "\n"
    )

    if dry_run:
        print(header, end="")
        return job.run_name, 0, "dry-run"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(header)
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        code = int(proc.returncode)
        msg = "ok" if code == 0 else f"exit {code}"
        return job.run_name, code, msg


def _print_best_vs_mamba(root: Path) -> None:
    csv_path = root / "outputs" / "TransformerLMSearch" / "summary_transformer_lm_search.csv"
    if not csv_path.is_file():
        print(f"No summary at {csv_path}; skip comparison.", file=sys.stderr)
        return
    import csv

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        bv = row.get("best_val_loss", "").strip()
        if not bv:
            continue
        try:
            scored.append((float(bv), row))
        except ValueError:
            continue
    if not scored:
        print("No successful runs with best_val_loss; skip comparison.")
        return
    scored.sort(key=lambda x: x[0])
    best_loss, best = scored[0]
    bpc = best_loss / __import__("math").log(2)
    ppl = __import__("math").exp(best_loss)

    print()
    print("========== SEARCH: best vs Mamba2 LM reference ==========")
    print(f"Best run:        {best.get('run_name', '')}")
    print(f"  val_loss:      {best_loss:.8f}")
    print(f"  val_bpc:       {bpc:.8f}")
    print(f"  val_ppl:       {ppl:.6f}")
    print(f"  params:        {best.get('params', '')}")
    print(f"  d_model/L/H/do/lr: {best.get('d_model')}/{best.get('n_layers')}/{best.get('n_heads')}/"
          f"{best.get('dropout')}/{best.get('lr')}")
    print(f"  output_dir:    {best.get('output_dir', '')}")
    print()
    print(f"Mamba2 depth4 ref val_loss: {MAMBA_DEPTH4_REF:.8f}  (delta best: {best_loss - MAMBA_DEPTH4_REF:+.8f})")
    print(f"Mamba2 depth6 ref val_loss: {MAMBA_DEPTH6_REF:.8f}  (delta best: {best_loss - MAMBA_DEPTH6_REF:+.8f})")
    print("=========================================================")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-gpus", type=int, default=None, help="Parallel workers (default: $NUM_GPUS or 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--python", type=str, default=sys.executable, help="Python binary for train_lm.py")
    p.add_argument("--config", type=str, default=BASE_CONFIG, help="Base YAML for train_lm")
    p.add_argument("--dry-run", action="store_true", help="Print planned jobs; do not train")
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument("--max-jobs", type=int, default=None, help="Run only first N jobs (debug)")
    args = p.parse_args()

    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = int(os.environ.get("NUM_GPUS", "8"))
    if num_gpus < 1:
        print("--num-gpus must be >= 1", file=sys.stderr)
        return 1

    root = repo_root()
    os.chdir(root)
    print(f"repo_root={root}", flush=True)

    jobs = build_jobs(seed=args.seed)
    if args.max_jobs is not None:
        jobs = jobs[: max(0, args.max_jobs)]

    print(f"Total jobs: {len(jobs)}  num_gpus: {num_gpus}  dry_run={args.dry_run}", flush=True)

    if args.dry_run:
        for i, job in enumerate(jobs):
            gpu = i % num_gpus
            run_one_job(
                root=root,
                python_bin=args.python,
                gpu_id=gpu,
                job=job,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                base_cfg=args.config,
                dry_run=True,
            )
        print(f"(dry-run: {len(jobs)} jobs listed)", flush=True)
        return 0

    failed: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        futs = {}
        for i, job in enumerate(jobs):
            gpu = i % num_gpus
            fut = ex.submit(
                run_one_job,
                root=root,
                python_bin=args.python,
                gpu_id=gpu,
                job=job,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                base_cfg=args.config,
                dry_run=False,
            )
            futs[fut] = job.run_name
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                rn, code, msg = fut.result()
                if code != 0:
                    failed.append((rn, code, msg))
                    print(f"FAILED {rn}: {msg}", flush=True)
                else:
                    print(f"OK {rn}", flush=True)
            except Exception as e:
                failed.append((name, -1, repr(e)))
                print(f"FAILED {name}: {e!r}", flush=True)

    if failed:
        print("\n========== Failed jobs ==========", flush=True)
        for rn, code, msg in failed:
            print(f"  {rn}  code={code}  {msg}", flush=True)
    else:
        print("\nAll jobs finished with exit code 0.", flush=True)

    if not args.skip_aggregate:
        agg = subprocess.run(
            [
                args.python,
                str(root / "code" / "aggregate_transformer_lm_search.py"),
                "--root",
                str(root / "outputs" / "TransformerLMSearch"),
                "--csv-out",
                str(root / "outputs" / "TransformerLMSearch" / "summary_transformer_lm_search.csv"),
                "--print-top",
                "10",
            ],
            cwd=str(root),
        )
        if agg.returncode != 0:
            print(f"aggregate_transformer_lm_search.py exited {agg.returncode}", file=sys.stderr)
        _print_best_vs_mamba(root)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
