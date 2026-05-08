#!/usr/bin/env python3
"""Aggregate MQAR failure sweep runs under outputs/MQARFailureSweep/ → CSV + report.

Scans all subdirs with meta_metrics.csv; model_name is parsed from experiment_name
(mqar_fail_<model>_v<vocab>_seed<seed>) with no hardcoded model list.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "MQARFailureSweep"

EXP_RE = re.compile(r"^mqar_fail_(.+)_v(\d+)_seed(\d+)$")

SWEEP_VOCABS = [512, 640, 704, 768, 896, 1024, 1280, 1536, 2048]


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
    """Same thresholds as aggregate_arch_eval._mq_regime."""
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


def _pick_metric(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = _sf(row.get(k, ""))
        if v is not None:
            return v
    return None


def _extract_model_vocab_seed(exp_name: str) -> tuple[str, int, int] | None:
    m = EXP_RE.match(exp_name.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _label_group(mean_em: float, collapse_rate: float) -> str:
    if mean_em >= 0.80:
        return "stable"
    if 0.10 <= mean_em < 0.80:
        return "transition"
    if mean_em < 0.10 or collapse_rate >= 0.5:
        return "collapsed"
    return "unknown"


def _read_meta_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize MQAR failure sweep (per-seed + grouped report).")
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory with mqar_fail_* runs (default: {DEFAULT_ROOT})",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/summary_mqar_failure_sweep_best.csv)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output report path (default: <root>/report_mqar_failure_sweep_best.txt)",
    )
    args = ap.parse_args()
    sweep_root: Path = args.root.resolve()
    summary_out = (args.summary or sweep_root / "summary_mqar_failure_sweep_best.csv").resolve()
    report_out = (args.report or sweep_root / "report_mqar_failure_sweep_best.txt").resolve()

    meta_paths = sorted(sweep_root.glob("**/meta_metrics.csv"))
    per_run: list[dict[str, str]] = []

    fieldnames = [
        "model_name",
        "seed",
        "input_seq_len",
        "num_kv_pairs",
        "vocab_size",
        "params",
        "answer_acc",
        "exact_match",
        "answer_loss",
        "regime",
        "output_dir",
        "status",
    ]

    for mp in meta_paths:
        row = _read_meta_row(mp)
        exp = row.get("experiment_name", "").strip()
        parsed = _extract_model_vocab_seed(exp)
        if parsed is None:
            continue
        model_name, _, _ = parsed

        seed_s = row.get("seed", "").strip()
        try:
            seed_i = int(seed_s)
        except ValueError:
            seed_i = -1

        isl = row.get("input_seq_len", "").strip()
        nkv = row.get("num_kv_pairs", "").strip()
        vs = row.get("vocab_size", "").strip()
        st = row.get("status", "").strip() or ("ok" if row else "missing")

        acc = _pick_metric(
            row,
            (
                "test_loss_ckpt_answer_accuracy",
                "test_answer_accuracy",
                "test_acc_ckpt_answer_accuracy",
            ),
        )
        em = _pick_metric(
            row,
            (
                "test_loss_ckpt_exact_match",
                "test_exact_match",
                "test_acc_ckpt_exact_match",
            ),
        )
        aloss = _pick_metric(
            row,
            (
                "test_loss_ckpt_answer_loss",
                "test_answer_loss",
                "test_acc_ckpt_answer_loss",
            ),
        )
        npar = row.get("num_parameters_trainable", "").strip()
        od = row.get("output_dir", "").strip()
        regime = _mq_regime(acc, em)

        def _fmt(x: float | None) -> str:
            if x is None:
                return ""
            return f"{x:.8f}"

        per_run.append(
            {
                "model_name": model_name,
                "seed": str(seed_i),
                "input_seq_len": isl,
                "num_kv_pairs": nkv,
                "vocab_size": vs,
                "params": npar,
                "answer_acc": _fmt(acc),
                "exact_match": _fmt(em),
                "answer_loss": _fmt(aloss),
                "regime": regime,
                "output_dir": od,
                "status": st,
            }
        )

    per_run.sort(key=lambda r: (r["model_name"], int(r["vocab_size"] or 0), int(r["seed"] or 0)))
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_run:
            w.writerow(r)

    by_mv: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for r in per_run:
        if r["status"] != "ok":
            continue
        try:
            vs = int(r["vocab_size"])
        except ValueError:
            continue
        by_mv[(r["model_name"], vs)].append(r)

    lines: list[str] = []
    lines.append(f"MQAR failure sweep report (root={sweep_root})")
    lines.append(f"Per-run summary: {summary_out}")
    lines.append("")

    for model_name in sorted({m for (m, _) in by_mv}):
        lines.append(f"## model_name={model_name}")
        lines.append("")
        lines.append(
            "Grouped by model_name / vocab_size (seeds aggregated): mean_acc, std_acc, mean_em, std_em, "
            "collapse_rate (fraction regime==too_hard), saturation_rate (fraction regime==too_easy), label."
        )
        lines.append("")
        first_collapsed_global: int | None = None
        last_stable_global: int | None = None
        for vs in SWEEP_VOCABS:
            rows_g = by_mv.get((model_name, vs), [])
            if not rows_g:
                lines.append(f"vocab={vs}\t(no completed ok runs in summary)")
                lines.append("")
                continue
            accs_f = [x for x in (_sf(x["answer_acc"]) for x in rows_g) if x is not None]
            ems_f = [x for x in (_sf(x["exact_match"]) for x in rows_g) if x is not None]
            regs = [x["regime"] for x in rows_g]
            mean_acc = statistics.mean(accs_f) if accs_f else float("nan")
            std_acc = statistics.stdev(accs_f) if len(accs_f) > 1 else 0.0
            mean_em = statistics.mean(ems_f) if ems_f else float("nan")
            std_em = statistics.stdev(ems_f) if len(ems_f) > 1 else 0.0
            collapse_rate = sum(1 for r in regs if r == "too_hard") / max(len(regs), 1)
            sat_rate = sum(1 for r in regs if r == "too_easy") / max(len(regs), 1)
            label = _label_group(mean_em, collapse_rate)
            if label == "collapsed" and first_collapsed_global is None:
                first_collapsed_global = vs
            if label == "stable":
                last_stable_global = vs

            def _ft(a: float) -> str:
                if a != a:
                    return "nan"
                return f"{a:.4f}"

            lines.append(f"vocab_size={vs}")
            lines.append(
                f"  mean_acc={_ft(mean_acc)} std_acc={_ft(std_acc)} "
                f"mean_em={_ft(mean_em)} std_em={_ft(std_em)}"
            )
            lines.append(
                f"  collapse_rate={collapse_rate:.3f} saturation_rate={sat_rate:.3f} label={label}"
            )
            lines.append("")

        lines.append(
            f"estimated_failure_point: first_collapsed_vocab={first_collapsed_global} "
            f"last_stable_vocab={last_stable_global}"
        )
        lines.append("")

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_out} ({len(per_run)} rows)")
    print(f"Wrote {report_out}")


if __name__ == "__main__":
    main()
