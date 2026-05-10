#!/usr/bin/env python3
"""Aggregate Text8 BPC benchmark runs under outputs/Text8BPC/

Reads each ``*/meta_metrics.csv``, builds per-seed summary and per-architecture scorecards
(student-t 95%% CI uses ``arch_eval_common.mean_std_sem_ci``, same tcrit table as ArchEval).

This analysis assumes Text8 BPC was evaluated with fixed ``train_max_steps`` and ``eval.max_batches``
(see columns in summary rows and in each ``meta_metrics.csv``).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from arch_eval_common import fmt_fin, mean_std_sem_ci

_CODE = Path(__file__).resolve().parent
ROOT = _CODE.parent

_DIR_RE = re.compile(r"^text8_(.+)_seed(\d+)$")


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


def _si(s: str) -> int | None:
    t = (s or "").strip()
    if not t:
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _parse_run_dir(dirname: str) -> tuple[str, int] | None:
    m = _DIR_RE.fullmatch(dirname)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _ci_row(prefix: str, xs: list[float]) -> dict[str, str]:
    st = mean_std_sem_ci(xs)
    blank_ci = st["n_ok"] < 2
    out: dict[str, str] = {
        f"{prefix}_mean": fmt_fin(st["mean"]),
        f"{prefix}_std": "",
        f"{prefix}_sem": "",
        f"{prefix}_ci95_low": "",
        f"{prefix}_ci95_high": "",
    }
    if st["n_ok"] >= 2:
        out[f"{prefix}_std"] = fmt_fin(st["std"])
        out[f"{prefix}_sem"] = fmt_fin(st["sem"])
    if blank_ci:
        return out
    out[f"{prefix}_ci95_low"] = fmt_fin(st["ci95_low"])
    out[f"{prefix}_ci95_high"] = fmt_fin(st["ci95_high"])
    return out


def _read_meta(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: str(v) if v is not None else "" for k, v in rows[0].items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--registry",
        type=str,
        default="configs/EvalRegistry/text8_bpc_core.yaml",
        help="Registry YAML (used for stem in output filenames and optional arch ordering).",
    )
    p.add_argument(
        "--outputs-root",
        type=str,
        default="outputs/Text8BPC",
        help="Directory containing text8_<arch>_seed<N> run folders.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    reg_path = Path(args.registry)
    if not reg_path.is_absolute():
        reg_path = ROOT / reg_path
    stem = reg_path.stem
    out_root = Path(args.outputs_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root = out_root.resolve()

    ordering: list[str] = []
    if reg_path.is_file():
        cfg = OmegaConf.load(reg_path)
        for row in OmegaConf.select(cfg, "architectures", default=[]) or []:
            d = OmegaConf.to_container(row, resolve=True)
            if isinstance(d, dict):
                name = d.get("arch_name") or d.get("name")
                if name:
                    ordering.append(str(name))

    summary_path = out_root / f"summary_{stem}.csv"
    score_path = out_root / f"scorecard_{stem}.csv"
    report_path = out_root / f"report_{stem}.txt"

    summary_rows: list[dict[str, str]] = []
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)

    meta_paths = sorted(out_root.glob("*/meta_metrics.csv"))
    for meta_path in meta_paths:
        parsed = _parse_run_dir(meta_path.parent.name)
        rowd = _read_meta(meta_path)
        if parsed is None or rowd is None:
            continue
        arch_name, seed = parsed
        status = rowd.get("status", "").strip().lower()

        summary_rows.append(
            {
                "arch_name": arch_name,
                "seed": str(seed),
                "params": rowd.get("num_parameters_trainable", ""),
                "best_val_loss": rowd.get("best_val_loss", ""),
                "best_val_bpc": rowd.get("best_val_bpc", ""),
                "final_val_loss": rowd.get("final_val_loss", ""),
                "final_val_bpc": rowd.get("final_val_bpc", ""),
                "train_max_steps": rowd.get("train_max_steps", ""),
                "eval_max_batches": rowd.get("eval_max_batches", ""),
                "status": rowd.get("status", ""),
                "output_dir": str(meta_path.parent.resolve()),
            }
        )

        if status != "ok":
            continue
        b_loss = _sf(rowd.get("best_val_loss", ""))
        f_loss = _sf(rowd.get("final_val_loss", ""))
        b_bpc = _sf(rowd.get("best_val_bpc", ""))
        f_bpc = _sf(rowd.get("final_val_bpc", ""))
        npar = _si(rowd.get("num_parameters_trainable", ""))
        if b_loss is None or f_loss is None or b_bpc is None or f_bpc is None:
            continue
        by_arch[arch_name].append(
            {
                "seed": seed,
                "best_val_loss": b_loss,
                "final_val_loss": f_loss,
                "best_val_bpc": b_bpc,
                "final_val_bpc": f_bpc,
                "params": npar,
            }
        )

    def _sort_key(r: dict[str, str]) -> tuple[str, int]:
        ar = r["arch_name"]
        try:
            sd = int(r["seed"])
        except ValueError:
            sd = 0
        ia = ordering.index(ar) if ar in ordering else 999
        return (ia, ar, sd)

    summary_rows.sort(key=_sort_key)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    s_fields = [
        "arch_name",
        "seed",
        "params",
        "best_val_loss",
        "best_val_bpc",
        "final_val_loss",
        "final_val_bpc",
        "train_max_steps",
        "eval_max_batches",
        "status",
        "output_dir",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in s_fields})

    arch_names = list(ordering) if ordering else sorted(by_arch.keys())
    for a in sorted(by_arch.keys()):
        if a not in arch_names:
            arch_names.append(a)

    score_rows: list[dict[str, str]] = []
    for arch in arch_names:
        rows = by_arch.get(arch, [])
        if not rows:
            score_rows.append({"arch_name": arch, "n_ok": "0"})
            continue
        params_list = [r["params"] for r in rows if r.get("params") is not None]
        params_mean = ""
        if params_list:
            params_mean = str(int(round(statistics.fmean([int(x) for x in params_list]))))

        b_bpcs = [r["best_val_bpc"] for r in rows]
        f_bpcs = [r["final_val_bpc"] for r in rows]
        b_losses = [r["best_val_loss"] for r in rows]
        f_losses = [r["final_val_loss"] for r in rows]

        out: dict[str, str] = {"arch_name": arch, "n_ok": str(len(rows))}
        out["params_mean"] = params_mean
        out.update(_ci_row("best_val_bpc", b_bpcs))
        out.update(_ci_row("final_val_bpc", f_bpcs))
        out.update(_ci_row("best_val_loss", b_losses))
        out.update(_ci_row("final_val_loss", f_losses))
        score_rows.append(out)

    sc_fields = [
        "arch_name",
        "n_ok",
        "params_mean",
        "best_val_bpc_mean",
        "best_val_bpc_std",
        "best_val_bpc_sem",
        "best_val_bpc_ci95_low",
        "best_val_bpc_ci95_high",
        "final_val_bpc_mean",
        "final_val_bpc_std",
        "final_val_bpc_sem",
        "final_val_bpc_ci95_low",
        "final_val_bpc_ci95_high",
        "best_val_loss_mean",
        "best_val_loss_std",
        "best_val_loss_sem",
        "best_val_loss_ci95_low",
        "best_val_loss_ci95_high",
        "final_val_loss_mean",
        "final_val_loss_std",
        "final_val_loss_sem",
        "final_val_loss_ci95_low",
        "final_val_loss_ci95_high",
    ]
    with score_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sc_fields)
        w.writeheader()
        for r in score_rows:
            w.writerow({k: r.get(k, "") for k in sc_fields})

    # Report
    trains = {_si(r["train_max_steps"]) for r in summary_rows if _si(r["train_max_steps"]) is not None}
    evals = {_si(r["eval_max_batches"]) for r in summary_rows if _si(r["eval_max_batches"]) is not None}

    lines = [
        "Text8 BPC benchmark report",
        "",
        "IMPORTANT: Text8 BPC metrics here were evaluated with fixed train.max_steps (train_max_steps)",
        "and eval.max_batches on the validation loader; see summary columns and each run's meta_metrics.csv.",
        "",
        f"Registry stem: {stem}",
        f"Scanned: {out_root}",
        f"Runs in summary table: {len(summary_rows)}",
        f"distinct train_max_steps in summary: {sorted(trains) if trains else 'n/a'}",
        f"distinct eval_max_batches in summary: {sorted(evals) if evals else 'n/a'}",
        "",
        "Artifacts:",
        f"  {summary_path.relative_to(ROOT)}",
        f"  {score_path.relative_to(ROOT)}",
        "",
        "Per-architecture scorecard (mean ± 95% CI via student-t / normal for n>20):",
    ]
    for r in score_rows:
        lines.append(
            f"  {r['arch_name']}: n_ok={r['n_ok']}  "
            f"best_val_bpc mean={r.get('best_val_bpc_mean', '')} "
            f"[{r.get('best_val_bpc_ci95_low', '')}, {r.get('best_val_bpc_ci95_high', '')}]  "
            f"final_val_bpc mean={r.get('final_val_bpc_mean', '')}"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {summary_path.relative_to(ROOT)}", flush=True)
    print(f"wrote {score_path.relative_to(ROOT)}", flush=True)
    print(f"wrote {report_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
