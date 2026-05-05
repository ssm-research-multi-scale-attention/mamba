#!/usr/bin/env python3
"""
Aggregate metrics for runs produced by grid_configs/B_stride_sweep.yaml:
  tiny_shakespeare_multiscale_gated_mamba2_random_windows_block_size_1024_stride_{2,4,8,16}

Writes to a dedicated directory (default: outputs/processing/grid_aggregated_B_stride_sweep/)
so runs of aggregate_grid_results.py (default grid_aggregated/) stay untouched.
Re-running this script overwrites only files inside its own default --out-dir.
For a new snapshot, pass e.g. --out-dir outputs/processing/grid_B_stride_$(date -Iminutes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import aggregate_grid_results as agr

# Matches run_grid.generate_name for B_stride_sweep + gated multiscale yaml experiment.name
DEFAULT_NAME_GLOB = (
    "tiny_shakespeare_multiscale_gated_mamba2_random_windows_block_size_1024_stride_*"
)
DEFAULT_OUT = agr.ROOT / "outputs" / "processing" / "grid_aggregated_B_stride_sweep"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=agr.ROOT / "outputs" / "TinyShakespeare",
        help="Directory containing per-run subfolders.",
    )
    p.add_argument(
        "--name-glob",
        type=str,
        default=DEFAULT_NAME_GLOB,
        help="fnmatch on folder name (default matches B_stride_sweep.yml only).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for CSVs (separate from grid_aggregated/).",
    )
    p.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Defaults to OUT_DIR/B_stride_grid_meta_metrics.csv.",
    )
    p.add_argument(
        "--out-metrics",
        type=Path,
        default=None,
        help="Defaults to OUT_DIR/B_stride_grid_metrics.csv.",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_meta = args.out_meta or (out_dir / "B_stride_grid_meta_metrics.csv")
    out_metrics = args.out_metrics or (out_dir / "B_stride_grid_metrics.csv")
    name_glob = (args.name_glob or "").strip()

    n_meta, n_m = agr.aggregate(
        outputs_root=args.outputs_root.resolve(),
        name_glob=name_glob,
        out_meta=out_meta,
        out_metrics=out_metrics,
    )
    print(f"[B_stride_sweep] runs (meta rows): {n_meta}")
    print(f"[B_stride_sweep] metric lines: {n_m}")
    print(f"Wrote {out_meta}")
    print(f"Wrote {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
