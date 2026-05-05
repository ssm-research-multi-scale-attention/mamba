#!/usr/bin/env python3
"""Analyze MQAR difficulty sweep summary → CSV + human-readable report."""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import click

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
_MQAR = _ROOT / "outputs" / "MQAR"

# Names like: mqar_sweep_A_mamba2_depth4_L160_minNone_kv16_vocab512_train20000_seed42
# Delay: ..._L256_min128_kv16_...
_NAME_RE = re.compile(
    r"^mqar_sweep_([ABCD])_(.+?)_L(\d+)_min([A-Za-z0-9]+)_kv(\d+)_vocab(\d+)_train(\d+)_seed(\d+)$"
)


def _parse_float(s: str) -> float | None:
    t = (s or "").strip()
    if not t:
        return None
    try:
        x = float(t)
    except ValueError:
        return None
    if x != x:  # NaN
        return None
    return x


def _primary_metrics(row: dict[str, str]) -> tuple[float | None, float | None]:
    # Prefer accuracy/EM from best-acc checkpoint; fallback to loss checkpoint / legacy cols.
    acc_cols = ("test_acc_ckpt_answer_accuracy", "test_loss_ckpt_answer_accuracy", "test_answer_accuracy")
    em_cols = ("test_acc_ckpt_exact_match", "test_loss_ckpt_exact_match", "test_exact_match")
    acc: float | None = None
    for k in acc_cols:
        v = _parse_float(row.get(k, ""))
        if v is not None:
            acc = v
            break
    em: float | None = None
    for k in em_cols:
        v = _parse_float(row.get(k, ""))
        if v is not None:
            em = v
            break
    return acc, em


def _regime(acc: float | None, em: float | None) -> str:
    if acc is None:
        return "unstable_or_edge"
    em_v = em if em is not None else 0.0
    if acc < 0.05 and em_v == 0.0:
        return "too_hard"
    if acc > 0.95 and em_v > 0.8:
        return "too_easy"
    if (0.1 <= acc <= 0.9) or (0.05 <= em_v <= 0.8):
        return "interesting"
    return "unstable_or_edge"


def _parse_name(experiment_name: str) -> dict[str, str]:
    out: dict[str, str] = {
        "group": "",
        "model_short": "",
        "setting_key": "",
        "parsed_ok": "0",
    }
    m = _NAME_RE.match(experiment_name.strip())
    if not m:
        return out
    group, model_s, L, min_rest, kv, vocab, train, _seed = m.groups()
    min_tok = f"min{min_rest}"
    out["parsed_ok"] = "1"
    out["group"] = group
    out["model_short"] = model_s
    out["setting_key"] = (
        f"{group}/L{L}/min{min_tok}/kv{kv}/vocab{vocab}/train{train}"
    )
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--summary",
    "summary_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Default: outputs/MQAR/summary_mqar_difficulty_sweep.csv",
)
@click.option(
    "--out-analyzed",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Default: outputs/MQAR/summary_mqar_difficulty_sweep_analyzed.csv",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Default: outputs/MQAR/mqar_sweep_report.txt",
)
def main(
    summary_path: Path | None,
    out_analyzed: Path | None,
    report_path: Path | None,
) -> int:
    sp = summary_path if summary_path is not None else (_MQAR / "summary_mqar_difficulty_sweep.csv")
    oa = out_analyzed if out_analyzed is not None else (_MQAR / "summary_mqar_difficulty_sweep_analyzed.csv")
    rp = report_path if report_path is not None else (_MQAR / "mqar_sweep_report.txt")

    if not sp.is_file():
        print(f"Missing summary: {sp}", file=sys.stderr)
        return 1

    analyzed_rows: list[dict[str, str]] = []
    with sp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fields = reader.fieldnames or []
        for row in reader:
            acc, em = _primary_metrics(row)
            pname = row.get("experiment_name", "") or ""
            meta = _parse_name(pname)
            regime = _regime(acc, em)

            nar: dict[str, str] = {k: row.get(k, "") for k in base_fields}
            nar["primary_acc"] = f"{acc:.8f}" if acc is not None else ""
            nar["primary_em"] = f"{em:.8f}" if em is not None else ""
            nar["regime"] = regime
            nar["group"] = meta["group"]
            nar["model_short"] = meta["model_short"]
            nar["setting_key"] = meta["setting_key"]
            nar["parsed_name_ok"] = meta["parsed_ok"]
            analyzed_rows.append(nar)

    out_fields = list(base_fields) + [
        "primary_acc",
        "primary_em",
        "regime",
        "group",
        "model_short",
        "setting_key",
        "parsed_name_ok",
    ]

    _MQAR.mkdir(parents=True, exist_ok=True)
    with oa.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in analyzed_rows:
            w.writerow({k: r.get(k, "") for k in out_fields})

    regime_counts: dict[str, int] = defaultdict(int)
    for r in analyzed_rows:
        regime_counts[r["regime"]] += 1

    by_setting: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in analyzed_rows:
        sk = r.get("setting_key", "")
        if sk:
            by_setting[sk].append(r)

    setting_stats: list[tuple[str, int, float, float, float]] = []
    # (setting_key, n_interesting, spread, mean_acc, n_models)
    for sk, grp in by_setting.items():
        n_int = sum(1 for row in grp if row.get("regime") == "interesting")
        accs: list[float] = []
        for row in grp:
            a = _parse_float(row.get("primary_acc", ""))
            if a is not None:
                accs.append(a)
        spread = max(accs) - min(accs) if len(accs) >= 2 else 0.0
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        setting_stats.append((sk, n_int, spread, mean_acc, len(grp)))

    setting_stats.sort(key=lambda x: (-x[1], -x[2], -x[3]))

    interesting_settings = [(sk, ni, spv) for sk, ni, spv, _m, _n in setting_stats if ni > 0]
    lines: list[str] = []

    lines.append("MQAR difficulty sweep analysis")
    lines.append(f"  source: {sp}")
    lines.append("")
    lines.append("1. Runs by regime:")
    for reg in sorted(regime_counts.keys(), key=lambda k: (-regime_counts[k], k)):
        lines.append(f"   {reg}: {regime_counts[reg]}")
    lines.append("")

    lines.append(
        "2. Top \"interesting\" settings (by count of interesting models desc, "
        "then spread of primary_acc across models desc):"
    )
    if not interesting_settings:
        lines.append("   (none)")
    else:
        rank = 0
        for sk, ni, spv in interesting_settings[:20]:
            rank += 1
            grp = by_setting.get(sk, [])
            acc_vals = [_parse_float(x.get("primary_acc", "")) for x in grp]
            acc_vals = [x for x in acc_vals if x is not None]
            sp_actual = max(acc_vals) - min(acc_vals) if len(acc_vals) >= 2 else 0.0
            lines.append(f"   {rank}. {sk} — interesting_models={ni} spread_acc={sp_actual:.4f}")

    lines.append("")
    lines.append("3. All models per setting_key:")
    for sk in sorted(by_setting.keys()):
        lines.append(f"   [{sk}]")
        grp = sorted(by_setting[sk], key=lambda x: x.get("model_short", ""))
        for row in grp:
            mn = row.get("model_short", "")
            rg = row.get("regime", "")
            pa = row.get("primary_acc", "?")
            pe = row.get("primary_em", "?")
            en = row.get("experiment_name", "")
            lines.append(f"      {mn}: regime={rg} primary_acc={pa} primary_em={pe} ({en})")

    lines.append("")
    lines.append("4. Recommendation:")
    if not interesting_settings:
        lines.append(
            "MQAR not useful in current setup; regimes are mostly too easy/too hard."
        )
        lines.append(
            "   (No setting had ≥1 model in the \"interesting\" band per rules.)"
        )
    else:
        top5 = [t[0] for t in interesting_settings[:5]]
        lines.append("Interesting settings exist. Rerun top settings with seeds 42/43/44, e.g.:")
        lines.append("")
        for sk in top5:
            lines.append(f"   - {sk}")
            lines.append(
                "     template: same overrides as sweep row for this setting_key;"
                " set experiment.seed=42/43/44 and unique experiment.name per run."
            )

    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {oa}")
    print(f"Wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
