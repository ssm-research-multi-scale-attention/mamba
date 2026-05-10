"""Shared ArchEval helpers: seed lists, output paths, t-critical values, meta/skip checks."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

# Student t two-sided 95%, df = n-1; precomputed for df 1..19; n>20 -> 1.96
T_CRITICAL_N: dict[int, float] = {
    2: 12.706204736,
    3: 4.302652730,
    4: 3.182446305,
    5: 2.776445105,
    6: 2.570581836,
    7: 2.446911851,
    8: 2.364624252,
    9: 2.306004135,
    10: 2.262157163,
    11: 2.228138852,
    12: 2.200985160,
    13: 2.178812830,
    14: 2.160368656,
    15: 2.144786688,
    16: 2.131449546,
    17: 2.119905299,
    18: 2.109815578,
    19: 2.100922040,
    20: 2.093024054,
}

DEFAULT_LM_SEEDS: tuple[int, ...] = (42,)
DEFAULT_MQAR_SEEDS: tuple[int, ...] = (42, 43, 44)
DEFAULT_TIMING_REPEATS = 1


def parse_int_list_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    seen: set[int] = set()
    deduped: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def lm_dir_stem(arch: str, seed: int, lm_seeds: list[int]) -> str:
    """Legacy: only [42] -> lm_{arch}; else lm_{arch}_seed{N}."""
    if lm_seeds == [42]:
        return f"lm_{arch}"
    return f"lm_{arch}_seed{seed}"


def timing_csv_relpath(arch: str, repeat: int, timing_repeats: int) -> str:
    if timing_repeats <= 1:
        return f"outputs/ArchEval/timing_{arch}.csv"
    return f"outputs/ArchEval/timing_{arch}_repeat{repeat}.csv"


def tcrit(n: int) -> float | None:
    if n < 2:
        return None
    if n > 20:
        return 1.96
    return T_CRITICAL_N.get(n)


def mean_std_sem_ci(xs: list[float]) -> dict[str, Any]:
    """n_ok, mean, std, sem, ci95_low, ci95_high; NaN for insufficient n."""
    ys = [float(x) for x in xs if x == x]
    n = len(ys)
    nan = float("nan")
    base: dict[str, Any] = {
        "n_ok": n,
        "mean": nan,
        "std": nan,
        "sem": nan,
        "ci95_low": nan,
        "ci95_high": nan,
    }
    if n < 1:
        return base
    m = statistics.fmean(ys)
    base["mean"] = m
    if n < 2:
        return base
    sd = statistics.stdev(ys)
    base["std"] = sd
    sem = sd / math.sqrt(n)
    base["sem"] = sem
    t = tcrit(n)
    if t is None:
        return base
    half = t * sem
    base["ci95_low"] = m - half
    base["ci95_high"] = m + half
    return base


def _row_status_ok(row: Mapping[str, str], key: str = "status") -> bool:
    st = str(row.get(key, "")).strip().lower()
    return st == "ok"


def lm_meta_ok(meta_path: Path) -> bool:
    if not meta_path.is_file():
        return False
    with meta_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    row = rows[0]
    if _row_status_ok(row):
        return True
    if row.get("best_val_loss", "").strip() and str(row.get("status", "")).strip() == "":
        return True
    return False


def mqar_meta_ok(meta_path: Path) -> bool:
    if not meta_path.is_file():
        return False
    with meta_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    return _row_status_ok(rows[0])


def timing_csv_ok(csv_path: Path, block_size: int = 1024) -> bool:
    """True if file has block_size rows with status ok for batches 1/8/16 (requested)."""
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    need = {"1", "8", "16"}
    found: dict[str, bool] = {k: False for k in need}
    for r in rows:
        if str(r.get("block_size", "")).strip() != str(block_size):
            continue
        br = str(r.get("batch_size_requested", r.get("batch_size", ""))).strip()
        if br not in found:
            continue
        if _row_status_ok(r):
            found[br] = True
    return all(found.values())


def fmt_fin(x: Any, nd: int = 8) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return ""
    if xf != xf:
        return ""
    return f"{xf:.{nd}f}"


def mqar_group_label(collapse_rate: float, mean_em: float | None) -> str:
    """Group-level label from aggregate collapse_rate (frac too_hard seeds) and mean EM."""
    em = 0.0 if mean_em is None or mean_em != mean_em else mean_em
    if collapse_rate == 0 and em >= 0.80:
        return "stable"
    if collapse_rate >= 0.5 or em < 0.10:
        return "collapsed"
    return "transition"


def vocab_for_setting(setting: str) -> int | None:
    if setting == "easy":
        return 512
    if setting == "trans704":
        return 704
    if setting == "trans768":
        return 768
    return None
