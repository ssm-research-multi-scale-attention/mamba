#!/usr/bin/env python3
"""Aggregate outputs/TransformerLMSanity/*/meta_metrics.csv + config into one CSV."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]  # mamba_multi_scale_get_started root


def _read_meta(meta_path: Path) -> dict[str, str]:
    if not meta_path.is_file():
        return {}
    with meta_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "outputs/TransformerLMSanity",
        help="Directory with per-run subfolders",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=PROJECT / "outputs/TransformerLMSanity/summary_transformer_lm_sanity.csv",
    )
    args = p.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Missing {root}", file=sys.stderr)
        return 1

    rows_out: list[dict[str, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith(".") or sub.name == "__pycache__":
            continue
        meta_path = sub / "meta_metrics.csv"
        cfg_path = sub / "config_resolved.yaml"
        meta = _read_meta(meta_path)
        if not cfg_path.is_file():
            rows_out.append(
                {
                    "run_name": sub.name,
                    "d_model": "",
                    "n_layers": "",
                    "n_heads": "",
                    "dropout": "",
                    "lr": "",
                    "seed": "",
                    "params": "",
                    "best_val_loss": "",
                    "best_val_bpc": "",
                    "best_val_ppl": "",
                    "epochs_completed": "",
                    "status": "missing_config",
                    "output_dir": str(sub),
                }
            )
            continue
        cfg = OmegaConf.load(cfg_path)
        bv = meta.get("best_val_loss", "").strip()
        try:
            bvf = float(bv) if bv else float("nan")
            bpc = f"{bvf / math.log(2):.8f}"
            bppl = f"{math.exp(bvf):.6f}"
        except ValueError:
            bpc = ""
            bppl = ""

        rows_out.append(
            {
                "run_name": sub.name,
                "d_model": str(OmegaConf.select(cfg, "model.d_model", default="")),
                "n_layers": str(OmegaConf.select(cfg, "model.transformer.n_layers", default="")),
                "n_heads": str(OmegaConf.select(cfg, "model.transformer.n_heads", default="")),
                "dropout": str(OmegaConf.select(cfg, "model.transformer.dropout", default="")),
                "lr": str(OmegaConf.select(cfg, "train.lr", default="")),
                "seed": str(OmegaConf.select(cfg, "experiment.seed", default=meta.get("seed", ""))),
                "params": str(meta.get("num_parameters_trainable", "")),
                "best_val_loss": bv,
                "best_val_bpc": bpc,
                "best_val_ppl": bppl,
                "epochs_completed": str(meta.get("epochs_completed", "")),
                "status": (
                    "ok"
                    if meta_path.is_file() and meta.get("best_val_loss", "").strip()
                    else "missing_meta"
                ),
                "output_dir": str(sub),
            }
        )

    fieldnames = [
        "run_name",
        "d_model",
        "n_layers",
        "n_heads",
        "dropout",
        "lr",
        "seed",
        "params",
        "best_val_loss",
        "best_val_bpc",
        "best_val_ppl",
        "epochs_completed",
        "status",
        "output_dir",
    ]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {args.csv_out} ({len(rows_out)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
