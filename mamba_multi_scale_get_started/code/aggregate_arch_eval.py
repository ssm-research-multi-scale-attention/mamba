#!/usr/bin/env python3
"""Merge ArchEval LM / MQAR / timing outputs → summary_arch_eval.csv."""
from __future__ import annotations

import argparse
import csv
import math
import os
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
ARCH_EVAL = ROOT / "outputs" / "ArchEval"


def _sf(s: str) -> float | None:
    t = (s or "").strip()
    if not t:
        return None
    try:
        x = float(t)
        if x != x:
            return None
        return x
    except ValueError:
        return None


def _mq_regime(acc: float | None, em: float | None) -> str:
    if acc is None:
        return "unstable_or_edge"
    ev = em if em is not None else 0.0
    if acc < 0.05 and ev == 0.0:
        return "too_hard"
    if acc > 0.95 and ev > 0.8:
        return "too_easy"
    if (0.1 <= acc <= 0.9) or (0.05 <= ev <= 0.8):
        return "interesting"
    return "unstable_or_edge"


def _read_meta_lm(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows[0] if rows else {}


def _pick(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = _sf(row.get(k, ""))
        if v is not None:
            return v
    return None


def _mq_primary(row: dict[str, str]) -> tuple[float | None, float | None]:
    acc = _pick(
        row,
        (
            "test_acc_ckpt_answer_accuracy",
            "test_loss_ckpt_answer_accuracy",
            "test_answer_accuracy",
        ),
    )
    em = _pick(
        row,
        (
            "test_acc_ckpt_exact_match",
            "test_loss_ckpt_exact_match",
            "test_exact_match",
        ),
    )
    return acc, em


def _read_timing_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


FIELDNAMES = [
    "arch_name",
    "arch_type",
    "task",
    "setting",
    "seed",
    "timing_repeat",
    "params",
    "lm_val_loss",
    "lm_bpc",
    "lm_ppl",
    "mqar_answer_acc",
    "mqar_exact_match",
    "mqar_regime",
    "timing_batch",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "tokens_per_second",
    "max_cuda_memory_mb",
]


def _empty_timing_vals() -> dict[str, str]:
    return {
        "timing_batch": "",
        "latency_ms_mean": "",
        "latency_ms_p50": "",
        "latency_ms_p95": "",
        "tokens_per_second": "",
        "max_cuda_memory_mb": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/EvalRegistry/architectures.yaml",
    )
    ap.add_argument(
        "--lm-seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated LM seeds (default: ARCH_EVAL_LM_SEEDS env or legacy 42). "
            "If unset elsewhere, mqar fallback default is 42,43,44."
        ),
    )
    ap.add_argument(
        "--mqar-seeds",
        type=str,
        default=None,
        help="Comma-separated MQAR seeds (default: ARCH_EVAL_MQAR_SEEDS or 42,43,44).",
    )
    ap.add_argument(
        "--timing-repeats",
        type=int,
        default=None,
        help="Timing repeats per arch (default: ARCH_EVAL_TIMING_REPEATS or 1).",
    )
    ap.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output summary filename/path. Relative paths are written inside outputs/ArchEval/. "
            "Absolute paths are used as-is. Default: outputs/ArchEval/summary_arch_eval.csv"
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting default summary_arch_eval.csv for non-full registries.",
    )
    args = ap.parse_args()

    lm_seed_arg = (
        args.lm_seeds
        if args.lm_seeds is not None
        else os.environ.get("ARCH_EVAL_LM_SEEDS", "").strip() or ",".join(map(str, DEFAULT_LM_SEEDS))
    )
    mq_seed_arg = (
        args.mqar_seeds
        if args.mqar_seeds is not None
        else os.environ.get("ARCH_EVAL_MQAR_SEEDS", "").strip() or ",".join(map(str, DEFAULT_MQAR_SEEDS))
    )
    tr_raw = args.timing_repeats
    if tr_raw is None:
        tr_env = os.environ.get("ARCH_EVAL_TIMING_REPEATS", "").strip()
        tr_raw = int(tr_env) if tr_env else DEFAULT_TIMING_REPEATS
    timing_repeats = max(1, int(tr_raw))

    lm_seeds = parse_int_list_csv(lm_seed_arg)
    mqar_seeds = parse_int_list_csv(mq_seed_arg)
    reg = args.registry.resolve()
    if not reg.is_file():
        print(f"Missing registry {reg}", file=sys.stderr)
        return 1

    default_out = ARCH_EVAL / "summary_arch_eval.csv"
    output_file_provided = args.output_file is not None
    if output_file_provided:
        raw_out = Path(args.output_file)
        outp = raw_out.resolve() if raw_out.is_absolute() else (ARCH_EVAL / raw_out).resolve()
    else:
        outp = default_out.resolve()

    if (
        reg.name != "architectures.yaml"
        and not output_file_provided
        and outp == default_out.resolve()
        and not args.force
    ):
        print(
            "Refusing to overwrite summary_arch_eval.csv for non-full registry. "
            "Pass --output-file or --force.",
            file=sys.stderr,
        )
        return 1

    cfg = OmegaConf.load(reg)
    arch_list = OmegaConf.select(cfg, "architectures", default=[])
    if not arch_list:
        print(f"No architectures in {reg}", file=sys.stderr)
        return 1

    out_rows: list[dict[str, str]] = []

    settings = (
        ("easy", "512", "null"),
        ("trans704", "704", "null"),
        ("trans768", "768", "null"),
    )

    def empty_timing_row() -> dict[str, str]:
        return dict(_empty_timing_vals())

    for arch in arch_list:
        name = str(arch.name)
        atype = str(arch.type)
        bv_ref = ""
        params = ""
        lm_bv_map: dict[int, str] = {}

        for seed in lm_seeds:
            stem = lm_dir_stem(name, seed, lm_seeds)
            lm_dir = ARCH_EVAL / stem
            lm_meta_path = lm_dir / "meta_metrics.csv"
            lm_meta = _read_meta_lm(lm_meta_path)
            bv = lm_meta.get("best_val_loss", "")
            if bv.strip():
                lm_bv_map[seed] = bv
                bv_ref = bv
            lf = _sf(bv)
            lm_bpc = f"{lf / math.log(2):.8f}" if lf is not None else ""
            lm_ppl = f"{math.exp(lf):.8f}" if lf is not None else ""
            params = str(lm_meta.get("num_parameters_trainable", params)).strip()

            lm_row = {
                "arch_name": name,
                "arch_type": atype,
                "task": "lm",
                "setting": "",
                "seed": str(seed),
                "timing_repeat": "",
                "params": params if params else str(lm_meta.get("num_parameters_trainable", "")),
                "lm_val_loss": bv,
                "lm_bpc": lm_bpc,
                "lm_ppl": lm_ppl,
                "mqar_answer_acc": "",
                "mqar_exact_match": "",
                "mqar_regime": "",
                **empty_timing_row(),
            }
            out_rows.append({k: lm_row.get(k, "") for k in FIELDNAMES})
            if not lm_meta_path.is_file():
                print(f"WARN: missing LM output {lm_meta_path}", file=sys.stderr)

        def _bv_fallback_mq(mq_seed: int) -> str:
            bv_m = lm_bv_map.get(mq_seed)
            if bv_m and bv_m.strip():
                return bv_m
            for s in lm_seeds:
                bx = lm_bv_map.get(s, "")
                if bx.strip():
                    return bx
            return bv_ref

        for setting, vocab, mnul in settings:
            for seed in mqar_seeds:
                mq_dir = ARCH_EVAL / f"mqar_{setting}_{name}_seed{seed}"
                mq_meta_path = mq_dir / "meta_metrics.csv"
                bv_last = _bv_fallback_mq(seed)

                mq_lm_bpc = ""
                mq_lm_ppl = ""
                lf2 = _sf(bv_last)
                if lf2 is not None:
                    mq_lm_bpc = f"{lf2 / math.log(2):.8f}"
                    mq_lm_ppl = f"{math.exp(lf2):.8f}"

                if not mq_meta_path.is_file():
                    print(f"WARN: missing MQAR output {mq_meta_path}", file=sys.stderr)
                    acc_s = ""
                    em_s = ""
                    rg = ""
                else:
                    with mq_meta_path.open(newline="", encoding="utf-8") as f:
                        mrows = list(csv.DictReader(f))
                    mmeta = mrows[0] if mrows else {}
                    acc_v, em_v = _mq_primary(mmeta)
                    acc_s = f"{acc_v:.8f}" if acc_v is not None else ""
                    em_s = f"{em_v:.8f}" if em_v is not None else ""
                    rg = _mq_regime(acc_v, em_v)
                    pm = str(mmeta.get("num_parameters_trainable", "")).strip()
                    if pm:
                        params = pm

                out_rows.append(
                    {
                        "arch_name": name,
                        "arch_type": atype,
                        "task": "mqar",
                        "setting": setting,
                        "seed": str(seed),
                        "timing_repeat": "",
                        "params": params,
                        "lm_val_loss": bv_last,
                        "lm_bpc": mq_lm_bpc,
                        "lm_ppl": mq_lm_ppl,
                        "mqar_answer_acc": acc_s,
                        "mqar_exact_match": em_s,
                        "mqar_regime": rg,
                        **empty_timing_row(),
                    }
                )

        for repeat in range(timing_repeats):
            relpath = timing_csv_relpath(name, repeat, timing_repeats)
            tp = (ROOT / relpath).resolve()
            timing_rows_raw = _read_timing_csv(tp)
            bv_t = bv_ref
            t_lm_bpc = ""
            t_lm_ppl = ""
            lft = _sf(bv_t)
            if lft is not None:
                t_lm_bpc = f"{lft / math.log(2):.8f}"
                t_lm_ppl = f"{math.exp(lft):.8f}"

            if not timing_rows_raw:
                print(f"WARN: missing timing {tp}", file=sys.stderr)
                out_rows.append(
                    {
                        "arch_name": name,
                        "arch_type": atype,
                        "task": "timing",
                        "setting": "",
                        "seed": "",
                        "timing_repeat": str(repeat),
                        "params": params,
                        "lm_val_loss": bv_t,
                        "lm_bpc": t_lm_bpc,
                        "lm_ppl": t_lm_ppl,
                        "mqar_answer_acc": "",
                        "mqar_exact_match": "",
                        "mqar_regime": "",
                        **empty_timing_row(),
                    }
                )
            else:
                appended_any = False
                for tr in timing_rows_raw:
                    if str(tr.get("block_size", "")).strip() != "1024":
                        continue
                    appended_any = True
                    bs = str(tr.get("batch_size", "")).strip()
                    lat_m = str(tr.get("forward_latency_ms_mean", "")).strip()
                    lat_p50 = str(tr.get("forward_latency_ms_p50", "")).strip()
                    lat_p95 = str(tr.get("forward_latency_ms_p95", "")).strip()
                    tps = str(tr.get("tokens_per_second", "")).strip()
                    mxm = str(tr.get("max_cuda_memory_mb", "")).strip()

                    row_t = {
                        "arch_name": name,
                        "arch_type": atype,
                        "task": "timing",
                        "setting": "",
                        "seed": "",
                        "timing_repeat": str(repeat),
                        "params": params,
                        "lm_val_loss": bv_t,
                        "lm_bpc": t_lm_bpc,
                        "lm_ppl": t_lm_ppl,
                        "mqar_answer_acc": "",
                        "mqar_exact_match": "",
                        "mqar_regime": "",
                        "timing_batch": bs,
                        "latency_ms_mean": lat_m,
                        "latency_ms_p50": lat_p50,
                        "latency_ms_p95": lat_p95,
                        "tokens_per_second": tps,
                        "max_cuda_memory_mb": mxm,
                    }
                    out_rows.append({k: row_t.get(k, "") for k in FIELDNAMES})
                if not appended_any:
                    print(f"WARN: timing {tp} had no block_size=1024 rows", file=sys.stderr)
                    out_rows.append(
                        {
                            "arch_name": name,
                            "arch_type": atype,
                            "task": "timing",
                            "setting": "",
                            "seed": "",
                            "timing_repeat": str(repeat),
                            "params": params,
                            "lm_val_loss": bv_t,
                            "lm_bpc": t_lm_bpc,
                            "lm_ppl": t_lm_ppl,
                            "mqar_answer_acc": "",
                            "mqar_exact_match": "",
                            "mqar_regime": "",
                            **empty_timing_row(),
                        }
                    )

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"Wrote {outp} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


Read