#!/usr/bin/env python3
"""
Collect metrics.csv and meta_metrics.csv from TinyShakespeare run dirs (e.g. run_grid.py / run_grid.sh).

Writes two CSVs with one row per epoch-line (metrics) and one row per run (meta).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path

# repo root: outputs/processing/<this.py> -> parents[2]
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT = ROOT / "outputs" / "processing" / "grid_aggregated"

_BLOCK_SIZE_RE = re.compile(r".*_block_size_(\d+)")
_STRIDE_RE = re.compile(r".*_stride_(\d+)$")


def _meta_field_order() -> list[str]:
    """Stable column order for meta (matches train_lm.py when present)."""
    return [
        "experiment_name",
        "output_dir",
        "parsed_block_size",
        "parsed_stride",
        "seed",
        "device",
        "cuda_device_name",
        "epochs_planned",
        "epochs_completed",
        "early_stopped",
        "best_val_loss",
        "num_parameters_trainable",
        "wall_setup_sec",
        "wall_train_phases_sec",
        "wall_eval_phases_sec",
        "wall_training_loop_sec",
        "wall_overhead_sec",
        "finished_at_utc",
        "dataset",
        "data_sampling",
        "debug_initial_train_ce",
        "debug_initial_val_ce",
        "debug_shift_ok",
    ]


def _parse_block_size(run_name: str) -> str:
    m = _BLOCK_SIZE_RE.search(run_name)
    return m.group(1) if m else ""


def _parse_stride(run_name: str) -> str:
    m = _STRIDE_RE.match(run_name)
    return m.group(1) if m else ""


def _iter_run_dirs(outputs_root: Path, name_glob: str) -> list[Path]:
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Outputs root not found: {outputs_root}")
    dirs: list[Path] = []
    for p in sorted(outputs_root.iterdir()):
        if not p.is_dir():
            continue
        if name_glob and not fnmatch(p.name, name_glob):
            continue
        if (p / "meta_metrics.csv").is_file() or (p / "metrics.csv").is_file():
            dirs.append(p)
    return dirs


def aggregate(
    outputs_root: Path,
    name_glob: str,
    out_meta: Path,
    out_metrics: Path,
) -> tuple[int, int]:
    run_dirs = _iter_run_dirs(outputs_root, name_glob)
    meta_rows: list[OrderedDict[str, str]] = []
    metric_rows: list[dict[str, str]] = []

    for d in run_dirs:
        run_name = d.name
        rel_out = str(d.resolve().relative_to(ROOT))
        block = _parse_block_size(run_name)
        stride = _parse_stride(run_name)

        meta_path = d / "meta_metrics.csv"
        if meta_path.is_file():
            with meta_path.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not any((v or "").strip() for v in row.values()):
                        continue
                    od = OrderedDict()
                    od["experiment_name"] = row.get("experiment_name", "") or run_name
                    od["output_dir"] = rel_out
                    od["parsed_block_size"] = block
                    od["parsed_stride"] = stride
                    for k, v in row.items():
                        if k is None or k in ("experiment_name",):
                            continue
                        od[k] = v if v is not None else ""
                    meta_rows.append(od)

        mpath = d / "metrics.csv"
        if mpath.is_file():
            with mpath.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row:
                        continue
                    line = dict(row)
                    line["experiment_name"] = run_name
                    line["output_dir"] = rel_out
                    line["parsed_block_size"] = block
                    line["parsed_stride"] = stride
                    metric_rows.append(line)

    # --- write meta ---
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    all_meta_keys: set[str] = set()
    for od in meta_rows:
        all_meta_keys.update(od.keys())
    base_order = _meta_field_order()
    priority = (
        ["experiment_name", "output_dir", "parsed_block_size", "parsed_stride"]
        + [
            c
            for c in base_order
            if c not in {"experiment_name", "output_dir", "parsed_block_size", "parsed_stride"}
        ]
    )
    meta_fieldnames = [k for k in priority if k in all_meta_keys]
    meta_fieldnames.extend(sorted(all_meta_keys - set(meta_fieldnames)))
    with out_meta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_fieldnames, extrasaction="ignore")
        w.writeheader()
        for od in meta_rows:
            w.writerow({k: od.get(k, "") for k in meta_fieldnames})

    # --- write metrics ---
    m_fields: list[str] = ["experiment_name", "output_dir", "parsed_block_size", "parsed_stride"]
    seen_m: set[str] = set(m_fields)
    for row in metric_rows:
        for k in row:
            if k not in seen_m:
                seen_m.add(k)
                m_fields.append(k)

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=m_fields, extrasaction="ignore")
        w.writeheader()
        for row in metric_rows:
            w.writerow({k: row.get(k, "") for k in m_fields})

    return len(meta_rows), len(metric_rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs" / "TinyShakespeare",
        help="Directory containing per-run subfolders with metrics CSVs.",
    )
    p.add_argument(
        "--name-glob",
        type=str,
        default="*_block_size_*",
        help="fnmatch pattern on run folder name; empty string = all dirs with CSVs.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for aggregated CSVs.",
    )
    p.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Override path for aggregated meta_metrics (default: OUT_DIR/grid_meta_metrics.csv).",
    )
    p.add_argument(
        "--out-metrics",
        type=Path,
        default=None,
        help="Override path for aggregated metrics (default: OUT_DIR/grid_metrics.csv).",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_meta = args.out_meta or (out_dir / "grid_meta_metrics.csv")
    out_metrics = args.out_metrics or (out_dir / "grid_metrics.csv")
    name_glob = (args.name_glob or "").strip()

    n_meta, n_m = aggregate(
        outputs_root=args.outputs_root.resolve(),
        name_glob=name_glob,
        out_meta=out_meta,
        out_metrics=out_metrics,
    )
    print(f"Runs with meta rows: {n_meta}")
    print(f"Metric lines: {n_m}")
    print(f"Wrote {out_meta}")
    print(f"Wrote {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
