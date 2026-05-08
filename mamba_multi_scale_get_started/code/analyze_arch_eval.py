#!/usr/bin/env python3
"""Scorecard + narrative report from summary_arch_eval.csv (optional multi-seed / timing repeats)."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from arch_eval_common import (
    fmt_fin,
    mean_std_sem_ci,
    mqar_group_label,
    vocab_for_setting,
)

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
    if t_cand is None or t_base is None or t_base <= 0:
        return ""
    return f"{100.0 * (1.0 - t_cand / t_base):.4f}"


def _ci_dict(prefix: str, xs: list[float]) -> dict[str, str]:
    st = mean_std_sem_ci(xs)
    out: dict[str, str] = {
        f"{prefix}_n_ok": str(st["n_ok"]),
        f"{prefix}_mean": fmt_fin(st["mean"]) if st.get("mean") == st.get("mean") else "",
        f"{prefix}_std": fmt_fin(st["std"]) if st.get("std") == st.get("std") else "",
        f"{prefix}_sem": fmt_fin(st["sem"]) if st.get("sem") == st.get("sem") else "",
        f"{prefix}_ci95_low": fmt_fin(st["ci95_low"]) if st.get("ci95_low") == st.get("ci95_low") else "",
        f"{prefix}_ci95_high": fmt_fin(st["ci95_high"]) if st.get("ci95_high") == st.get("ci95_high") else "",
    }
    if st["n_ok"] == 1:
        out[f"{prefix}_std"] = ""
        out[f"{prefix}_sem"] = ""
        out[f"{prefix}_ci95_low"] = ""
        out[f"{prefix}_ci95_high"] = ""
    return out


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
        help="Optional path to summary CSV with baseline rows.",
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


def _has_depth4(rows: list[dict[str, str]]) -> bool:
    present = {
        str(r.get("arch_name", "")).strip()
        for r in rows
        if str(r.get("task", "")).strip() == "lm"
    }
    return "mamba2_depth4" in present


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


def _stable_collapsed_vocab(labels: dict[str, str]) -> tuple[str, str]:
    order = ["easy", "trans704", "trans768"]
    last_stable: str | None = None
    first_collapsed: str | None = None
    for setting in order:
        if setting not in labels:
            continue
        vocab = vocab_for_setting(setting)
        if vocab is None:
            continue
        lab = labels[setting]
        if lab == "stable":
            last_stable = str(vocab)
        if lab == "collapsed" and first_collapsed is None:
            first_collapsed = str(vocab)
    return (last_stable or "", first_collapsed or "")


def _timing_rep_key(rr: dict[str, str]) -> str:
    v = rr.get("timing_repeat", "")
    return str(v).strip() if str(v).strip() != "" else "0"


def main() -> int:
    args = _parse_args()
    inp = Path(args.summary).resolve()
    out_score = Path(args.out_scorecard).resolve()
    out_rep = Path(args.out_report).resolve()

    if not inp.is_file():
        print(f"Missing summary {inp}", file=sys.stderr)
        return 1

    rows = _load_csv_rows(inp)
    if not _has_depth4(rows):
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
            except OSError:
                continue
            rows = _append_missing_baseline_rows(rows, extra_rows)
            if _has_depth4(rows):
                break

    if not _has_depth4(rows):
        print(
            "Missing baseline row: mamba2_depth4. Run full registry or pass --baseline-summary.",
            file=sys.stderr,
        )
        return 1

    has_d6 = _has_required_baselines(rows)
    if not has_d6:
        print(
            "Note: mamba2_depth6 not in summary; using mamba2_depth4-only comparisons where needed.",
            file=sys.stderr,
        )

    mqar = [r for r in rows if r.get("task") == "mqar"]
    lm_rows_all = [r for r in rows if r.get("task") == "lm"]
    timing_rows = [r for r in rows if r.get("task") == "timing"]

    lm_loss_series: dict[str, list[float]] = defaultdict(list)
    lm_bpc_series: dict[str, list[float]] = defaultdict(list)
    lm_ppl_series: dict[str, list[float]] = defaultdict(list)
    arch_type_map: dict[str, str] = {}
    params_map: dict[str, str] = {}

    for r in lm_rows_all:
        a = str(r.get("arch_name", ""))
        arch_type_map[a] = str(r.get("arch_type", ""))
        params_map[a] = str(r.get("params", ""))
        lv = _sf(r.get("lm_val_loss", ""))
        bp = _sf(r.get("lm_bpc", ""))
        pp = _sf(r.get("lm_ppl", ""))
        if lv is not None:
            lm_loss_series[a].append(lv)
        if bp is not None:
            lm_bpc_series[a].append(bp)
        if pp is not None:
            lm_ppl_series[a].append(pp)

    mqar_grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in mqar:
        mqar_grouped[str(r["arch_name"]), str(r["setting"])].append(r)

    def mq_regime_stats(arch: str, setting: str) -> dict[str, str]:
        g = mqar_grouped.get((arch, setting), [])
        accs = [x for y in g if (x := _sf(y.get("mqar_answer_acc", ""))) is not None]
        ems = [x for y in g if (x := _sf(y.get("mqar_exact_match", ""))) is not None]
        regs = [str(y.get("mqar_regime", "")).strip() for y in g]
        m_acc = _mean(accs)
        m_em = _mean(ems)
        collapse_hard_r = _regime_fraction(regs, "too_hard")
        sat_easy_r = _regime_fraction(regs, "too_easy")
        grp = mqar_group_label(collapse_hard_r, m_em)
        ci_acc = _ci_dict(f"mqar_{setting}_answer_acc", accs)
        ci_em = _ci_dict(f"mqar_{setting}_exact_match", ems)
        pst_acc = _pstdev(accs) if len(accs) >= 2 else ""
        out: dict[str, str] = {
            "mean_acc": f"{m_acc:.8f}" if m_acc is not None else "",
            "mean_em": f"{m_em:.8f}" if m_em is not None else "",
            "std_acc": pst_acc,
            "collapse_rate": f"{collapse_hard_r:.6f}",
            "saturation_rate": f"{sat_easy_r:.6f}",
            "group_label": grp,
        }
        out.update(ci_acc)
        out.update(ci_em)
        return out

    reps_by_arch_bs: dict[tuple[str, str], dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in timing_rows:
        a = str(r.get("arch_name", "")).strip()
        bs = str(r.get("timing_batch", "")).strip()
        rv = _sf(r.get("tokens_per_second", ""))
        rep = _timing_rep_key(r)
        reps_by_arch_bs[(a, bs)][rep].append(rv)

    timing_mean_ci: dict[tuple[str, str], tuple[float | None, dict[str, str]]] = {}

    def _sort_rep_key(rep: str) -> float:
        try:
            return float(rep)
        except ValueError:
            return 0.0

    for ab, reps_map in reps_by_arch_bs.items():
        series: list[float] = []
        for _rep, ys in sorted(reps_map.items(), key=lambda x: _sort_rep_key(x[0])):
            good = [float(y) for y in ys if y is not None and y == y]
            if good:
                series.append(float(sum(good) / len(good)))
        m = statistics.fmean(series) if series else None
        ci = _ci_dict(f"timing_tps_bs{ab[1]}", series)
        timing_mean_ci[ab] = (m, ci)

    bd4_lm_loss = _mean(lm_loss_series.get("mamba2_depth4", []))
    bd6_lm_loss = _mean(lm_loss_series.get("mamba2_depth6", []))
    ref_lm_loss = bd6_lm_loss if bd6_lm_loss is not None else bd4_lm_loss

    tpl16_ref = timing_mean_ci.get(("mamba2_depth4", "16"), (None, {}))[0]

    rep_lines: list[str] = []
    score_lines: list[dict[str, str]] = []
    arch_list = sorted(arch_type_map.keys())

    for arch in arch_list:
        at = arch_type_map.get(arch, "")
        loss_mu = _mean(lm_loss_series.get(arch, []))
        bpc_mu = _mean(lm_bpc_series.get(arch, []))
        lm_ci_loss = _ci_dict("lm_val_loss", lm_loss_series.get(arch, []))
        lm_ci_ppl = _ci_dict("lm_ppl", lm_ppl_series.get(arch, []))
        lm_ci_bpc = _ci_dict("lm_bpc", lm_bpc_series.get(arch, []))
        ps = params_map.get(arch, "")
        d4pct = ""
        d6pct = ""
        if arch != "mamba2_depth4":
            d4pct = _pct_delta(loss_mu, bd4_lm_loss)
        if arch != "mamba2_depth6" and bd6_lm_loss is not None:
            d6pct = _pct_delta(loss_mu, bd6_lm_loss)

        regime_labels: dict[str, str] = {}
        easy = mq_regime_stats(arch, "easy")
        t704 = mq_regime_stats(arch, "trans704")
        t768 = mq_regime_stats(arch, "trans768")
        for setting, pak in ("easy", easy), ("trans704", t704), ("trans768", t768):
            regime_labels[setting] = pak["group_label"]

        last_stable_v, first_collapsed_v = _stable_collapsed_vocab(regime_labels)

        tps1 = timing_mean_ci.get((arch, "1"), (None, {}))[0]
        tps8 = timing_mean_ci.get((arch, "8"), (None, {}))[0]
        tps16 = timing_mean_ci.get((arch, "16"), (None, {}))[0]
        _, ci_bs1 = timing_mean_ci.get((arch, "1"), (None, {}))
        _, ci_bs8 = timing_mean_ci.get((arch, "8"), (None, {}))
        _, ci_bs16 = timing_mean_ci.get((arch, "16"), (None, {}))
        spd_pen16 = _pct_speed_penalty(tps16, tpl16_ref)

        decision = ""
        at_l = at.lower()
        if "candidate" in at_l:
            d4_704 = mq_regime_stats("mamba2_depth4", "trans704")
            d6_704 = mq_regime_stats("mamba2_depth6", "trans704") if has_d6 else dict(d4_704)
            d4_768 = mq_regime_stats("mamba2_depth4", "trans768")
            d6_768 = mq_regime_stats("mamba2_depth6", "trans768") if has_d6 else dict(d4_768)

            fv704_vals = [_sf(d4_704["mean_em"])]
            if has_d6:
                fv704_vals.append(_sf(d6_704["mean_em"]))
            g704 = [x for x in fv704_vals if x is not None]
            fv704_em = max(g704) if g704 else -1.0

            fv768_vals = [_sf(d4_768["mean_em"])]
            if has_d6:
                fv768_vals.append(_sf(d6_768["mean_em"]))
            g768 = [x for x in fv768_vals if x is not None]
            fv768_em = max(g768) if g768 else -1.0

            cl704_bd4 = _sf(d4_704["collapse_rate"]) or 0.0
            cl704_bd6 = _sf(d6_704["collapse_rate"]) or 0.0
            cl768_bd4 = _sf(d4_768["collapse_rate"]) or 0.0
            cl768_bd6 = _sf(d6_768["collapse_rate"]) or 0.0
            if not has_d6:
                cl704_bd6 = cl704_bd4
                cl768_bd6 = cl768_bd4

            c704_em = _sf(t704["mean_em"])
            c768_em = _sf(t768["mean_em"])
            cl704_cand = _sf(t704["collapse_rate"]) or 0.0
            cl768_cand = _sf(t768["collapse_rate"]) or 0.0
            lm_ok = loss_mu is None or ref_lm_loss is None or loss_mu <= ref_lm_loss * 1.02
            lm_win = loss_mu is not None and ref_lm_loss is not None and loss_mu < ref_lm_loss
            mq704_win = c704_em is not None and c704_em > fv704_em
            mq768_win = c768_em is not None and c768_em > fv768_em
            eps = 1e-9
            collapse_ok = (cl704_cand + eps < cl704_bd4 and cl704_cand + eps < cl704_bd6) and (
                cl768_cand + eps < cl768_bd4 and cl768_cand + eps < cl768_bd6
            )
            mq_recall_candidate = mq704_win and mq768_win and collapse_ok and lm_ok

            has_mq_transition_signal = mq704_win or mq768_win
            ref_worse_lm = bd6_lm_loss if bd6_lm_loss is not None else bd4_lm_loss
            lm_dip = ref_worse_lm is not None and loss_mu is not None and loss_mu > ref_worse_lm * 1.02
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
            tradeoff_signal = has_mq_transition_signal and (lm_dip or speed_dip or collapse_dip)

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
        elif "transformer_baseline" in at_l:
            decision = "TRANSFORMER_BASELINE"
        else:
            decision = "UNKNOWN_ROLE"

        row_out: dict[str, str] = {
            "arch_name": arch,
            "arch_type": at,
            "params": ps,
            "lm_val_loss": f"{loss_mu:.8f}" if loss_mu is not None else "",
            "lm_bpc": f"{bpc_mu:.8f}" if bpc_mu is not None else "",
            "lm_delta_vs_mamba2_depth4_pct": d4pct,
            "lm_delta_vs_mamba2_depth6_pct": d6pct,
            "mqar_easy_mean_acc": easy["mean_acc"],
            "mqar_easy_mean_em": easy["mean_em"],
            "mqar_easy_std_acc": easy["std_acc"],
            "mqar_easy_collapse_rate": easy["collapse_rate"],
            "mqar_easy_saturation_rate": easy["saturation_rate"],
            "mqar_easy_group_label": easy["group_label"],
            "mqar_trans704_mean_acc": t704["mean_acc"],
            "mqar_trans704_mean_em": t704["mean_em"],
            "mqar_trans704_std_acc": t704["std_acc"],
            "mqar_trans704_collapse_rate": t704["collapse_rate"],
            "mqar_trans704_saturation_rate": t704["saturation_rate"],
            "mqar_trans704_group_label": t704["group_label"],
            "mqar_trans768_mean_acc": t768["mean_acc"],
            "mqar_trans768_mean_em": t768["mean_em"],
            "mqar_trans768_std_acc": t768["std_acc"],
            "mqar_trans768_collapse_rate": t768["collapse_rate"],
            "mqar_trans768_saturation_rate": t768["saturation_rate"],
            "mqar_trans768_group_label": t768["group_label"],
            "timing_tokens_per_sec_batch1": f"{tps1:.6f}" if tps1 is not None else "",
            "timing_tokens_per_sec_batch8": f"{tps8:.6f}" if tps8 is not None else "",
            "timing_tokens_per_sec_batch16": f"{tps16:.6f}" if tps16 is not None else "",
            "speed_penalty_vs_mamba2_depth4_pct_batch16": spd_pen16,
            "last_stable_vocab": last_stable_v,
            "first_collapsed_vocab": first_collapsed_v,
            "decision": decision,
        }

        skip_merge = {
            "mean_acc",
            "mean_em",
            "std_acc",
            "collapse_rate",
            "saturation_rate",
            "group_label",
        }
        for d in (easy, t704, t768):
            for ck, cv in d.items():
                if ck in skip_merge:
                    continue
                row_out[ck] = cv

        row_out.update(ci_bs1)
        row_out.update(ci_bs8)
        row_out.update(ci_bs16)
        row_out.update(lm_ci_loss)
        row_out.update(lm_ci_ppl)
        row_out.update(lm_ci_bpc)

        score_lines.append(row_out)

    fields = sorted({k for s in score_lines for k in s.keys()})
    priority = [
        "arch_name",
        "arch_type",
        "params",
        "lm_val_loss",
        "lm_bpc",
        "lm_delta_vs_mamba2_depth4_pct",
        "lm_delta_vs_mamba2_depth6_pct",
        "mqar_easy_mean_acc",
        "mqar_easy_mean_em",
        "mqar_easy_std_acc",
        "mqar_easy_group_label",
        "mqar_trans704_mean_acc",
        "mqar_trans704_mean_em",
        "mqar_trans704_std_acc",
        "mqar_trans704_group_label",
        "mqar_trans768_mean_acc",
        "mqar_trans768_mean_em",
        "mqar_trans768_std_acc",
        "mqar_trans768_group_label",
        "timing_tokens_per_sec_batch1",
        "timing_tokens_per_sec_batch8",
        "timing_tokens_per_sec_batch16",
        "speed_penalty_vs_mamba2_depth4_pct_batch16",
        "last_stable_vocab",
        "first_collapsed_vocab",
        "decision",
    ]
    rest = [x for x in fields if x not in priority]
    fields_out = priority + sorted(rest)

    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_score.parent.mkdir(parents=True, exist_ok=True)

    rep_lines.append("Architectural evaluation scorecard")
    rep_lines.append(f"  source: {inp}")
    rep_lines.append("")
    rep_lines.append("Per-architecture:")
    for s in score_lines:
        rep_lines.append(
            f"  {s['arch_name']} ({s['decision']}): "
            f"lm_loss={s.get('lm_val_loss', '')} mq704_em={s.get('mqar_trans704_mean_em', '')} "
            f"tps16={s.get('timing_tokens_per_sec_batch16', '')} "
            f"speed_pen16={s.get('speed_penalty_vs_mamba2_depth4_pct_batch16', '')}"
        )
    rep_lines.append("")
    rep_lines.append("Notes:")
    rep_lines.append("  mqar_* collapse_rate/saturation_rate: fraction(per-seed) too_hard/too_easy (legacy mqar regime).")
    rep_lines.append("  mqar_* group_label ∈ {stable,transition,collapsed} uses collapse_rate + mean EM threshold rules.")
    rep_lines.append("  LM / MQAR / timing columns with _n_ok/_mean/_std/_sem/_ci95_* summarise across seeds/repeats.")
    rep_lines.append("  PASS_RECALL thresholds use best available Mamba2 baseline (depth6 if present else depth4).")

    with out_score.open("w", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(fs, fieldnames=fields_out, extrasaction="ignore")
        w.writeheader()
        w.writerows(score_lines)

    rep_lines.append("  Inspect the summary CSV for per-seed / per-repeat detail rows.")

    out_rep.write_text("\n".join(rep_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_score}")
    print(f"Wrote {out_rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
