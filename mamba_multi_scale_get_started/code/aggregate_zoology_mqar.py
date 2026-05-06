#!/usr/bin/env python3
"""Aggregate Zoology MQAR sanity runs from outputs/ZoologyMQAR/*/results.json + train_config.json."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _dig(d: dict, *keys: str, default: str = "") -> str:
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur) if cur is not None else default


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ZoologyMQAR"),
        help="Directory containing per-run subfolders",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=Path("outputs/ZoologyMQAR/summary_zoology_mqar.csv"),
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/ZoologyMQAR/report_zoology_mqar.txt"),
    )
    args = p.parse_args()

    root = args.output_root.resolve()
    if not root.is_dir():
        print(f"Missing output root: {root}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        results = _read_json(sub / "results.json")
        tcfg = _read_json(sub / "train_config.json")
        data = tcfg.get("data") or {}
        train_cfgs = data.get("train_configs") or [{}]
        tc0 = train_cfgs[0] if train_cfgs else {}
        mf = results.get("metrics_final") or {}
        row = {
            "run_name": sub.name,
            "model": _dig(tcfg, "model", "name"),
            "seed": str(results.get("seed", _dig(tcfg, "seed", default=""))),
            "input_seq_len": str(tc0.get("input_seq_len", "")),
            "num_kv_pairs": str(tc0.get("num_kv_pairs", "")),
            "vocab_size": str(tc0.get("vocab_size", "")),
            "train_examples": str(tc0.get("num_examples", "")),
            "val_test_examples": str((data.get("test_configs") or [{}])[0].get("num_examples", "")),
            "val_loss": str(mf.get("valid/loss", "")),
            "train_loss_last": "",
            "answer_accuracy": str(mf.get("valid/accuracy", "")),
            "exact_match": "",
            "status": str(results.get("status", "")),
            "output_dir": str(sub),
        }
        rows.append(row)

    fieldnames = [
        "run_name",
        "model",
        "seed",
        "input_seq_len",
        "num_kv_pairs",
        "vocab_size",
        "train_examples",
        "val_test_examples",
        "val_loss",
        "train_loss_last",
        "answer_accuracy",
        "exact_match",
        "status",
        "output_dir",
    ]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Short qualitative report
    def acc_for(reg_part: str, model_part: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            if reg_part in r["run_name"] and model_part in r["run_name"]:
                a = r["answer_accuracy"].strip()
                if a:
                    try:
                        out.append(float(a))
                    except ValueError:
                        pass
        return out

    lines = [
        "Zoology upstream MQAR sanity (this repo's wrapper around zoology.train).",
        "",
        "Compared to local ArchEval MQAR (train_mqar.py): same task family (Zoology multiquery_ar),",
        " but training loop / metrics are Zoology's Trainer (valid/accuracy = token accuracy on labels != -100).",
        "Exact numeric match to our runs is not expected; compare qualitative regimes only.",
        "",
        "--- Easy (vocab 512, L=128, kv=16) ---",
    ]
    for mk in ("mha", "mamba2", "mamba"):
        xs = acc_for("easy", mk)
        lines.append(f"  {mk}: n={len(xs)} acc samples: {xs}")

    lines += [
        "",
        "--- Transition 704 ---",
    ]
    for mk in ("mha", "mamba2", "mamba"):
        xs = acc_for("trans704", mk)
        lines.append(f"  {mk}: {xs}")

    lines += [
        "",
        "--- Transition 768 ---",
    ]
    for mk in ("mha", "mamba2", "mamba"):
        xs = acc_for("trans768", mk)
        lines.append(f"  {mk}: {xs}")

    lines += [
        "",
        "--- Hard 1024 (if present) ---",
    ]
    for mk in ("mha", "mamba2", "mamba"):
        xs = acc_for("hard", mk)
        lines.append(f"  {mk}: {xs}")

    lines += [
        "",
        "Qualitative questions:",
        "  - Easy solved? Look for answer_accuracy near 1.0 on mha/mamba2 when status=ok.",
        "  - Transition unstable? Compare acc spread across seeds at trans704/trans768.",
        "  - Hard collapse? Typically answer_accuracy near random (1/vocab for value slice) under difficult settings.",
        "",
        f"CSV: {args.csv_out.resolve()}",
    ]

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv_out} ({len(rows)} rows) and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
