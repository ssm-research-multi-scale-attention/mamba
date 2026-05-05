#!/usr/bin/env python3
"""Merge ArchEval LM / MQAR / timing outputs → summary_arch_eval.csv."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from omegaconf import OmegaConf

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/EvalRegistry/architectures.yaml",
    )
    args = ap.parse_args()
    reg = args.registry.resolve()
    if not reg.is_file():
        print(f"Missing registry {reg}", file=sys.stderr)
        return 1
    cfg = OmegaConf.load(reg)
    arch_list = OmegaConf.select(cfg, "architectures", default=[])
    if not arch_list:
        print(f"No architectures in {reg}", file=sys.stderr)
        return 1

    out_rows: list[dict[str, str]] = []

    seeds = ("42", "43", "44")
    settings = (
        ("easy", "512", "null"),
        ("trans704", "704", "null"),
        ("trans768", "768", "null"),
    )

    for arch in arch_list:
        name = str(arch.name)
        atype = str(arch.type)
        lm_dir = ARCH_EVAL / f"lm_{name}"
        lm_meta_path = lm_dir / "meta_metrics.csv"
        lm_meta = _read_meta_lm(lm_meta_path)
        bv = lm_meta.get("best_val_loss", "")
        lf = _sf(bv)
        params = str(lm_meta.get("num_parameters_trainable", ""))
        lm_bpc = f"{lf / math.log(2):.8f}" if lf is not None else ""
        lm_ppl = f"{math.exp(lf):.8f}" if lf is not None else ""

        def empty_timing() -> dict[str, str]:
            return {
                "timing_batch": "",
                "latency_ms_mean": "",
                "latency_ms_p50": "",
                "latency_ms_p95": "",
                "tokens_per_second": "",
                "max_cuda_memory_mb": "",
            }

        lm_row = {
            "arch_name": name,
            "arch_type": atype,
            "task": "lm",
            "setting": "",
            "seed": "42",
            "params": params,
            "lm_val_loss": bv,
            "lm_bpc": lm_bpc,
            "lm_ppl": lm_ppl,
            "mqar_answer_acc": "",
            "mqar_exact_match": "",
            "mqar_regime": "",
            **empty_timing(),
        }
        out_rows.append({k: lm_row.get(k, "") for k in FIELDNAMES})

        if not lm_meta_path.is_file():
            print(f"WARN: missing LM output {lm_meta_path}", file=sys.stderr)

        for setting, vocab, mnul in settings:
            for seed in seeds:
                mq_dir = ARCH_EVAL / f"mqar_{setting}_{name}_seed{seed}"
                mq_meta_path = mq_dir / "meta_metrics.csv"
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
                    pm = str(mmeta.get("num_parameters_trainable", params))
                    if pm.strip():
                        params = pm

                out_rows.append(
                    {
                        "arch_name": name,
                        "arch_type": atype,
                        "task": "mqar",
                        "setting": setting,
                        "seed": seed,
                        "params": params,
                        "lm_val_loss": bv,
                        "lm_bpc": lm_bpc,
                        "lm_ppl": lm_ppl,
                        "mqar_answer_acc": acc_s,
                        "mqar_exact_match": em_s,
                        "mqar_regime": rg,
                        **empty_timing(),
                    }
                )

        tp = ARCH_EVAL / f"timing_{name}.csv"
        timing_rows_raw = _read_timing_csv(tp)
        if not timing_rows_raw:
            print(f"WARN: missing timing {tp}", file=sys.stderr)
            out_rows.append(
                {
                    "arch_name": name,
                    "arch_type": atype,
                    "task": "timing",
                    "setting": "",
                    "seed": "",
                    "params": params,
                    "lm_val_loss": bv,
                    "lm_bpc": lm_bpc,
                    "lm_ppl": lm_ppl,
                    "mqar_answer_acc": "",
                    "mqar_exact_match": "",
                    "mqar_regime": "",
                    **empty_timing(),
                }
            )
        else:
            appended = False
            for tr in timing_rows_raw:
                if str(tr.get("block_size", "")).strip() != "1024":
                    continue
                appended = True
                bs = str(tr.get("batch_size", "")).strip()
                lat_m = str(tr.get("forward_latency_ms_mean", "")).strip()
                lat_p50 = str(tr.get("forward_latency_ms_p50", "")).strip()
                lat_p95 = str(tr.get("forward_latency_ms_p95", "")).strip()
                tps = str(tr.get("tokens_per_second", "")).strip()
                mxm = str(tr.get("max_cuda_memory_mb", "")).strip()

                out_rows.append(
                    {
                        "arch_name": name,
                        "arch_type": atype,
                        "task": "timing",
                        "setting": "",
                        "seed": "",
                        "params": params,
                        "lm_val_loss": bv,
                        "lm_bpc": lm_bpc,
                        "lm_ppl": lm_ppl,
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
                )
            if not appended:
                print(f"WARN: timing {tp} had no block_size=1024 rows", file=sys.stderr)
                out_rows.append(
                    {
                        "arch_name": name,
                        "arch_type": atype,
                        "task": "timing",
                        "setting": "",
                        "seed": "",
                        "params": params,
                        "lm_val_loss": bv,
                        "lm_bpc": lm_bpc,
                        "lm_ppl": lm_ppl,
                        "mqar_answer_acc": "",
                        "mqar_exact_match": "",
                        "mqar_regime": "",
                        **empty_timing(),
                    }
                )

    ARCH_EVAL.mkdir(parents=True, exist_ok=True)
    outp = ARCH_EVAL / "summary_arch_eval.csv"
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"Wrote {outp} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
