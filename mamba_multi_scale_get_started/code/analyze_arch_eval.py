#!/usr/bin/env python3
"""Scorecard + narrative report from summary_arch_eval.csv."""
from __future__ import annotations

import csv
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEF_SUMMARY = _ROOT / "outputs" / "ArchEval" / "summary_arch_eval.csv"
_OUT_REP = _ROOT / "outputs" / "ArchEval" / "report_arch_eval.txt"
_OUT_SCORE = _ROOT / "outputs" / "ArchEval" / "scorecard_arch_eval.csv"
_DEFAULT_BASELINE_FALLBACKS = [
    _ROOT / "outputs" / "ArchEval" / "scorecard_arch_eval.csv",
    _ROOT / "outputs" / "ArchEval" / "summary_arch_eval_full.csv",
    _ROOT / "outputs" / "ArchEval" / "summary_arch_eval_baselines.csv",
]


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


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pstdev(xs: list[float]) -> str:
    if len(xs) < 2:
        return ""
    return f"{statistics.pstdev(xs):.8f}"


def _regime_fraction(regimes: list[str], label: str) -> float:
    if not regimes:
        return 0.0
    return sum(1 for r in regimes if r == label) / len(regimes)


def _pct_delta(candidate: float | None, base: float | None) -> str:
    if candidate is None or base is None or base <= 0:
        return ""
    return f"{100.0 * (candidate / base - 1.0):.4f}"


def _pct_speed_penalty(t_cand: float | None, t_base: float | None) -> str:
    """Positive => candidate slower than base (fewer tokens/s)."""
    if t_cand is None or t_base is None or t_base <= 0:
        return ""
    return f"{100.0 * (1.0 - t_cand / t_base):.4f}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "summary",
        nargs="?",
        default=str(_DEF_SUMMARY),
        help="Path to summary_arch_eval.csv (default: outputs/ArchEval/summary_arch_eval.csv).",
    )
    p.add_argument(
        "--baseline-summary",
        type=str,
        default="",
        help="Optional path to summary CSV with baseline rows (mamba2_depth4/depth6).",
    )
    p.add_argument(
        "--out-scorecard",
        type=str,
        default=str(_OUT_SCORE),
        help="Output scorecard CSV path.",
    )
    p.add_argument(
        "--out-report",
        type=str,
        default=str(_OUT_REP),
        help="Output text report path.",
    )
    return p.parse_args()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _has_required_baselines(rows: list[dict[str, str]]) -> bool:
    present = {
        str(r.get("arch_name", "")).strip()
        for r in rows
        if str(r.get("task", "")).strip() == "lm"
    }
    return "mamba2_depth4" in present and "mamba2_depth6" in present


def _append_missing_baseline_rows(
    base_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    existing = {
        (str(r.get("task", "")), str(r.get("arch_name", "")), str(r.get("setting", "")), str(r.get("seed", "")))
        for r in base_rows
    }
    out = list(base_rows)
    for r in baseline_rows:
        arch = str(r.get("arch_name", "")).strip()
        if arch not in ("mamba2_depth4", "mamba2_depth6"):
            continue
        key = (str(r.get("task", "")), arch, str(r.get("setting", "")), str(r.get("seed", "")))
        if key not in existing:
            out.append(r)
            existing.add(key)
    return out


def main() -> int:
    args = _parse_args()
    inp = Path(args.summary).resolve()
    out_score = Path(args.out_scorecard).resolve()
    out_rep = Path(args.out_report).resolve()

    if not inp.is_file():
        print(f"Missing summary {inp}", file=sys.stderr)
        return 1

    rows = _load_csv_rows(inp)
    if not _has_required_baselines(rows):
        baseline_candidates: list[Path] = []
        if args.baseline_summary:
            baseline_candidates.append(Path(args.baseline_summary).resolve())
        baseline_candidates.extend(_DEFAULT_BASELINE_FALLBACKS)
        seen: set[Path] = set()
        for cand in baseline_candidates:
            if cand in seen or not cand.is_file() or cand == inp:
                continue
            seen.add(cand)
            try:
                extra_rows = _load_csv_rows(cand)
            except Exception:
                continue
            rows = _append_missing_baseline_rows(rows, extra_rows)
            if _has_required_baselines(rows):
                break

    if not _has_required_baselines(rows):
        print(
            "Missing baseline rows: mamba2_depth4, mamba2_depth6. "
            "Run full registry or pass --baseline-summary.",
            file=sys.stderr,
        )
        return 1

    mqar = [r for r in rows if r.get("task") == "mqar"]
    lm_rows = [r for r in rows if r.get("task") == "lm"]
    timing = [r for r in rows if r.get("task") == "timing"]

    lm_loss: dict[str, float | None] = {}
    lm_bpc_v: dict[str, float | None] = {}
    arch_type_map: dict[str, str] = {}
    params_map: dict[str, str] = {}
    for r in lm_rows:
        a = str(r.get("arch_name", ""))
        lm_loss[a] = _sf(r.get("lm_val_loss", ""))
        lm_bpc_v[a] = _sf(r.get("lm_bpc", ""))
        arch_type_map[a] = str(r.get("arch_type", ""))
        params_map[a] = str(r.get("params", ""))

    mqar_grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in mqar:
        mqar_grouped[str(r["arch_name"]), str(r["setting"])].append(r)

    def mq_stats(arch: str, setting: str) -> tuple[str, str, str, str, str]:
        """collapse_rate = frac(too_hard); saturation_rate = frac(too_easy)."""
        g = mqar_grouped.get((arch, setting), [])
        accs = [x for y in g if (x := _sf(y.get("mqar_answer_acc", ""))) is not None]
        ems = [x for y in g if (x := _sf(y.get("mqar_exact_match", ""))) is not None]
        regs = [y.get("mqar_regime", "") for y in g]
        m_acc = _mean(accs)
        m_em = _mean(ems)
        collapse_r = _regime_fraction(regs, "too_hard")
        sat_r = _regime_fraction(regs, "too_easy")
        return (
            f"{m_acc:.8f}" if m_acc is not None else "",
            f"{m_em:.8f}" if m_em is not None else "",
            _pstdev(accs) if len(accs) >= 2 else "",
            f"{collapse_r:.6f}",
            f"{sat_r:.6f}",
        )

    timing_tps: dict[tuple[str, str], float | None] = {}
    for r in timing:
        a = str(r.get("arch_name", ""))
        bs = str(r.get("timing_batch", "")).strip()
        if not bs:
            continue
        timing_tps[(a, bs)] = _sf(r.get("tokens_per_second", ""))

    all_arch = sorted(set(arch_type_map.keys()))
    bd4_loss = lm_loss.get("mamba2_depth4")
    bd6_loss = lm_loss.get("mamba2_depth6")
    tpl16_ref = timing_tps.get(("mamba2_depth4", "16"))

    score_lines: list[dict[str, str]] = []
    rep: list[str] = []
    rep.append("Architectural evaluation scorecard")
    rep.append(f"  source: {inp}")
    rep.append("")

    for arch in all_arch:
        at = arch_type_map.get(arch, "")
        loss = lm_loss.get(arch)
        bpc = lm_bpc_v.get(arch)
        ps = params_map.get(arch, "")
        d4pct = ""
        d6pct = ""
        if arch != "mamba2_depth4":
            d4pct = _pct_delta(loss, bd4_loss)
        if arch != "mamba2_depth6":
            d6pct = _pct_delta(loss, bd6_loss)

        mq_e_acc, mq_e_em, mq_e_sa, mq_e_col, mq_e_sat = mq_stats(arch, "easy")
        t704_acc, t704_em, t704_sa, t704_col, t704_sat = mq_stats(arch, "trans704")
        t768_acc, t768_em, t768_sa, t768_col, t768_sat = mq_stats(arch, "trans768")

        tps1 = timing_tps.get((arch, "1"))
        tps8 = timing_tps.get((arch, "8"))
        tps16 = timing_tps.get((arch, "16"))
        spd_pen16 = _pct_speed_penalty(tps16, tpl16_ref)

        decision = ""
        at_l = at.lower()
        if "candidate" in at_l:
            d4_704 = mq_stats("mamba2_depth4", "trans704")
            d6_704 = mq_stats("mamba2_depth6", "trans704")
            d4_768 = mq_stats("mamba2_depth4", "trans768")
            d6_768 = mq_stats("mamba2_depth6", "trans768")
            fv704_em = max(_sf(d4_704[1]) or -1.0, _sf(d6_704[1]) or -1.0)
            fv768_em = max(_sf(d4_768[1]) or -1.0, _sf(d6_768[1]) or -1.0)
            _704d4_col = _sf(d4_704[3])
            _704d6_col = _sf(d6_704[3])
            _768d4_col = _sf(d4_768[3])
            _768d6_col = _sf(d6_768[3])
            cl704_bd4 = 0.0 if _704d4_col is None else _704d4_col
            cl704_bd6 = 0.0 if _704d6_col is None else _704d6_col
            cl768_bd4 = 0.0 if _768d4_col is None else _768d4_col
            cl768_bd6 = 0.0 if _768d6_col is None else _768d6_col
            c704_em = _sf(t704_em)
            c768_em = _sf(t768_em)
            cl704_cand = _sf(t704_col) or 0.0
            cl768_cand = _sf(t768_col) or 0.0
            lm_ok = loss is None or bd6_loss is None or loss <= bd6_loss * 1.02
            lm_win = loss is not None and bd6_loss is not None and loss < bd6_loss
            mq704_win = c704_em is not None and c704_em > fv704_em
            mq768_win = c768_em is not None and c768_em > fv768_em
            eps = 1e-9
            collapse_ok = (cl704_cand + eps < cl704_bd4 and cl704_cand + eps < cl704_bd6) and (
                cl768_cand + eps < cl768_bd4 and cl768_cand + eps < cl768_bd6
            )
            mq_recall_candidate = mq704_win and mq768_win and collapse_ok and lm_ok

            has_mq_transition_signal = mq704_win or mq768_win
            lm_dip = bd6_loss is not None and loss is not None and loss > bd6_loss * 1.02
            speed_dip = (
                tps16 is not None
                and tpl16_ref is not None
                and tpl16_ref > 0
                and tps16 + eps < tpl16_ref
            )
            collapse_dip = (
                cl704_cand > cl704_bd4 + eps
                or cl704_cand > cl704_bd6 + eps
                or cl768_cand > cl768_bd4 + eps
                or cl768_cand > cl768_bd6 + eps
            )
            tradeoff_signal = (
                has_mq_transition_signal and (lm_dip or speed_dip or collapse_dip)
            )

            if mq_recall_candidate:
                decision = "PASS_RECALL"
            elif lm_win:
                decision = "PASS_LM"
            elif tradeoff_signal:
                decision = "TRADEOFF"
            else:
                decision = "FAIL"
        elif "baseline" in at_l:
            decision = "BASELINE"
        elif "equal_param_baseline" in at_l:
            decision = "EQUAL_PARAM_BASELINE"
        else:
            decision = "UNKNOWN_ROLE"

        score_lines.append(
            {
                "arch_name": arch,
                "arch_type": at,
                "params": ps,
                "lm_val_loss": f"{loss:.8f}" if loss is not None else "",
                "lm_bpc": f"{bpc:.8f}" if bpc is not None else "",
                "lm_delta_vs_mamba2_depth4_pct": d4pct,
                "lm_delta_vs_mamba2_depth6_pct": d6pct,
                "mqar_easy_mean_acc": mq_e_acc,
                "mqar_easy_mean_em": mq_e_em,
                "mqar_easy_collapse_rate": mq_e_col,
                "mqar_easy_saturation_rate": mq_e_sat,
                "mqar_trans704_mean_acc": t704_acc,
                "mqar_trans704_mean_em": t704_em,
                "mqar_trans704_std_acc": t704_sa,
                "mqar_trans704_collapse_rate": t704_col,
                "mqar_trans704_saturation_rate": t704_sat,
                "mqar_trans768_mean_acc": t768_acc,
                "mqar_trans768_mean_em": t768_em,
                "mqar_trans768_std_acc": t768_sa,
                "mqar_trans768_collapse_rate": t768_col,
                "mqar_trans768_saturation_rate": t768_sat,
                "timing_tokens_per_sec_batch1": f"{tps1:.6f}" if tps1 is not None else "",
                "timing_tokens_per_sec_batch8": f"{tps8:.6f}" if tps8 is not None else "",
                "timing_tokens_per_sec_batch16": f"{tps16:.6f}" if tps16 is not None else "",
                "speed_penalty_vs_mamba2_depth4_pct_batch16": spd_pen16,
                "decision": decision,
            }
        )

    fields = [
        "arch_name",
        "arch_type",
        "params",
        "lm_val_loss",
        "lm_bpc",
        "lm_delta_vs_mamba2_depth4_pct",
        "lm_delta_vs_mamba2_depth6_pct",
        "mqar_easy_mean_acc",
        "mqar_easy_mean_em",
        "mqar_easy_collapse_rate",
        "mqar_easy_saturation_rate",
        "mqar_trans704_mean_acc",
        "mqar_trans704_mean_em",
        "mqar_trans704_std_acc",
        "mqar_trans704_collapse_rate",
        "mqar_trans704_saturation_rate",
        "mqar_trans768_mean_acc",
        "mqar_trans768_mean_em",
        "mqar_trans768_std_acc",
        "mqar_trans768_collapse_rate",
        "mqar_trans768_saturation_rate",
        "timing_tokens_per_sec_batch1",
        "timing_tokens_per_sec_batch8",
        "timing_tokens_per_sec_batch16",
        "speed_penalty_vs_mamba2_depth4_pct_batch16",
        "decision",
    ]

    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_score.parent.mkdir(parents=True, exist_ok=True)

    if out_score == _OUT_SCORE and not _has_required_baselines(_load_csv_rows(inp)):
        print(
            "Warning: input summary has no baselines; writing to default scorecard path may overwrite a prior full scorecard. "
            "Use --out-scorecard to write to a separate file.",
            file=sys.stderr,
        )

    rep.append("Per-architecture:")
    for s in score_lines:
        rep.append(
            f"  {s['arch_name']} ({s['decision']}): "
            f"lm_loss={s['lm_val_loss']} mq704_em={s['mqar_trans704_mean_em']} "
            f"tps16={s['timing_tokens_per_sec_batch16']} speed_pen16={s['speed_penalty_vs_mamba2_depth4_pct_batch16']}"
        )
    rep.append("")
    rep.append("Notes:")
    rep.append("  collapse_rate = fraction(seeds with regime too_hard); saturation_rate = fraction(too_easy).")
    rep.append("  PASS_RECALL: trans704/768 EM > max(Mamba2 d4,d6); collapse lower than both baselines on both;")
    rep.append("    LM val loss ≤ depth6 × 1.02.")
    rep.append("  PASS_LM: LM below depth6. TRADEOFF: ≥1 transition EM win vs baselines but LM/speed/collapse dip.")
    rep.append("  Inspect raw summary_arch_eval.csv for per-seed rows.")

    with out_score.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(score_lines)

    out_rep.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"Wrote {out_score}")
    print(f"Wrote {out_rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
