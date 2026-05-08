#!/usr/bin/env python3
"""Emit TSV manifest lines for run_arch_eval.sh (validate config paths exist)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from arch_eval_common import (
    DEFAULT_LM_SEEDS,
    DEFAULT_MQAR_SEEDS,
    DEFAULT_TIMING_REPEATS,
    lm_dir_stem,
    parse_int_list_csv,
    timing_csv_relpath,
)

_CODE = Path(__file__).resolve().parent
ROOT = _CODE.parent


def _emit(registry: Path, lm_seeds: list[int], mqar_seeds: list[int], timing_repeats: int, dry_run: bool) -> int:
    reg_path = registry.resolve()
    if not reg_path.is_file():
        print(f"arch_eval_emit_manifest: missing registry {reg_path}", file=sys.stderr)
        return 1

    cfg = OmegaConf.load(reg_path)
    archs = OmegaConf.select(cfg, "architectures", default=None)
    if archs is None:
        print("arch_eval_emit_manifest: no architectures:", reg_path, file=sys.stderr)
        return 1
    mqar_settings = [
        ("easy", "512", "null"),
        ("trans704", "704", "null"),
        ("trans768", "768", "null"),
    ]

    lm_out: list[str] = []
    mqar_out: list[str] = []
    timing_out: list[str] = []

    for a in archs:
        name = str(a.name)
        atype = str(a.type)
        for key in ("lm_config", "mqar_config", "timing_config"):
            rel = Path(str(OmegaConf.select(a, key, default="")))
            ap = ROOT / rel
            if not ap.is_file():
                print(f"arch_eval_emit_manifest: missing file for {name} {key}={rel}", file=sys.stderr)
                return 1

        lm = str(a.lm_config)
        mqar = str(a.mqar_config)
        timing = str(a.timing_config)

        for seed in lm_seeds:
            out_stem = lm_dir_stem(name, seed, lm_seeds)
            lm_out.append(f"lm\t{name}\t{atype}\t{lm}\t{seed}\t{out_stem}")

        for setting, vocab, mnul in mqar_settings:
            for seed in mqar_seeds:
                mqar_out.append(f"mqar\t{name}\t{atype}\t{mqar}\t{setting}\t{seed}\t{vocab}\t{mnul}")

        for repeat in range(timing_repeats):
            tcsv = timing_csv_relpath(name, repeat, timing_repeats)
            timing_out.append(f"timing\t{name}\t{atype}\t{timing}\t{repeat}\t{tcsv}")

    n_arch = len(archs)
    n_lm = len(lm_out)
    n_mqar = len(mqar_out)
    n_timing = len(timing_out)
    total = n_lm + n_mqar + n_timing

    if dry_run:
        print(f"registry_path: {reg_path}")
        print(f"num_architectures: {n_arch}")
        print(f"num_lm_jobs: {n_lm}")
        print(f"num_mqar_jobs: {n_mqar}")
        print(f"num_timing_jobs: {n_timing}")
        print(f"num_total_jobs: {total}")
        print(f"lm_seeds: {','.join(str(s) for s in lm_seeds)}")
        print(f"mqar_seeds: {','.join(str(s) for s in mqar_seeds)}")
        print(f"timing_repeats: {timing_repeats}")
        return 0

    for line in lm_out + mqar_out + timing_out:
        print(line)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("registry", type=Path, nargs="?", default=ROOT / "configs/EvalRegistry/architectures.yaml")
    p.add_argument(
        "--lm-seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LM_SEEDS),
        help=f"Comma-separated LM seeds (default: {','.join(str(x) for x in DEFAULT_LM_SEEDS)}).",
    )
    p.add_argument(
        "--mqar-seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MQAR_SEEDS),
        help=f"Comma-separated MQAR seeds (default: {','.join(str(x) for x in DEFAULT_MQAR_SEEDS)}).",
    )
    p.add_argument(
        "--timing-repeats",
        type=int,
        default=DEFAULT_TIMING_REPEATS,
        help=f"Timing benchmark repeats per architecture (default: {DEFAULT_TIMING_REPEATS}).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts and exit without manifest lines.")
    args = p.parse_args()

    lm_seeds = parse_int_list_csv(args.lm_seeds)
    mqar_seeds = parse_int_list_csv(args.mqar_seeds)
    if not lm_seeds:
        print("arch_eval_emit_manifest: empty --lm-seeds", file=sys.stderr)
        return 1
    if not mqar_seeds:
        print("arch_eval_emit_manifest: empty --mqar-seeds", file=sys.stderr)
        return 1
    if args.timing_repeats < 1 or args.timing_repeats != int(args.timing_repeats):
        print("arch_eval_emit_manifest: --timing-repeats must be positive int", file=sys.stderr)
        return 1

    try:
        return _emit(Path(args.registry), lm_seeds, mqar_seeds, int(args.timing_repeats), args.dry_run)
    except Exception as e:  # noqa: BLE001
        print(f"arch_eval_emit_manifest: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
