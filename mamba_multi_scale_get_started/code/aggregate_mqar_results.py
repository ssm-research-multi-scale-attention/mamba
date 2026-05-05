#!/usr/bin/env python3
"""Aggregate outputs/MQAR/*/meta_metrics.csv into outputs/MQAR/summary_mqar.csv."""
from __future__ import annotations

import click
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _model_label(row: dict[str, str]) -> str:
    name = str(row.get("experiment_name", "")).lower()

    if "mamba2_depth4" in name:
        return "Mamba2 depth=4"
    if "mamba2_depth6" in name:
        return "Mamba2 depth=6"
    if "ms_gated_stride2" in name:
        return "MS-gated stride=2"
    if "ms_attention_stride4" in name or ("ms_attention" in name and "stride4" in name):
        return "MS-attention stride=4"

    backbone = str(row.get("backbone", ""))
    fusion = str(row.get("fusion", ""))
    stride = row.get("stride", "")

    if backbone == "mamba2_lm":
        return "Mamba2"
    if backbone == "multiscale_mamba2_lm":
        return f"MS-{fusion} stride={stride}"
    if backbone == "multiscale_mamba2_attention_lm":
        return f"MS-attention stride={stride}"

    return backbone or "unknown"


@click.command()
@click.option("--output-file", type=str, default="summary_mqar.csv")
def main(output_file: str) -> int:
    mqar_root = _ROOT / "outputs" / "MQAR"
    if not mqar_root.is_dir():
        print(f"No {mqar_root}", file=sys.stderr)
        return 1

    out_name = Path(output_file).name
    rows_out: list[dict[str, str]] = []
    for d in sorted(mqar_root.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "meta_metrics.csv"
        if not meta.is_file():
            continue
        with meta.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if not any((v or "").strip() for k, v in row.items() if k):
                    continue
                rows_out.append(
                    {
                        "experiment_name": row.get("experiment_name", ""),
                        "model": _model_label(row),
                        "seed": row.get("seed", ""),
                        "lr": row.get("lr", ""),
                        "input_seq_len": row.get("input_seq_len", ""),
                        "num_kv_pairs": row.get("num_kv_pairs", ""),
                        "vocab_size": row.get("vocab_size", ""),
                        "min_query_pos": row.get("min_query_pos", ""),
                        "params": str(row.get("num_parameters_trainable", "")),
                        "best_val_answer_loss": row.get("best_val_answer_loss", ""),
                        "best_val_answer_accuracy": row.get("best_val_answer_accuracy", ""),
                        "test_loss_ckpt_answer_accuracy": row.get(
                            "test_loss_ckpt_answer_accuracy",
                            row.get("test_answer_accuracy", ""),
                        ),
                        "test_acc_ckpt_answer_accuracy": row.get("test_acc_ckpt_answer_accuracy", ""),
                        "test_loss_ckpt_exact_match": row.get(
                            "test_loss_ckpt_exact_match",
                            row.get("test_exact_match", ""),
                        ),
                        "test_acc_ckpt_exact_match": row.get("test_acc_ckpt_exact_match", ""),
                    }
                )

    summary = mqar_root / out_name
    fieldnames = [
        "experiment_name",
        "model",
        "seed",
        "lr",
        "input_seq_len",
        "num_kv_pairs",
        "vocab_size",
        "min_query_pos",
        "params",
        "best_val_answer_loss",
        "best_val_answer_accuracy",
        "test_loss_ckpt_answer_accuracy",
        "test_acc_ckpt_answer_accuracy",
        "test_loss_ckpt_exact_match",
        "test_acc_ckpt_exact_match",
    ]
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {summary} ({len(rows_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
