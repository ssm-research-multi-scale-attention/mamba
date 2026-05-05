"""Multi-Query Associative Recall (MQAR) synthetic data (Zoology-style).

Ported from ``zoology.data.multiquery_ar.multiquery_ar`` (local, no Zoology deps).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MQARDataset(Dataset):
    """Causal LM samples: ``input_ids [L]``, ``labels [L]`` with ``-100`` on ignored positions."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        if inputs.shape != labels.shape:
            raise ValueError(f"inputs {inputs.shape} vs labels {labels.shape}")
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx].long(), self.labels[idx].long()


def validate_mqar_config(
    *,
    input_seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_passes: int,
    min_query_pos: int | None = None,
) -> None:
    if input_seq_len % 2 != 0:
        raise ValueError("MQAR: input_seq_len must be even")
    if vocab_size < input_seq_len:
        raise ValueError("MQAR: require vocab_size >= input_seq_len (keys/values must fit the scheme)")
    ctx = num_kv_pairs * 2 * num_passes
    if ctx + num_kv_pairs * 2 > input_seq_len:
        raise ValueError(
            f"MQAR: need num_kv_pairs*2*num_passes + num_kv_pairs*2 <= input_seq_len "
            f"(got ctx={ctx}, pairs={num_kv_pairs}, L={input_seq_len})"
        )
    if min_query_pos is not None:
        mqp = int(min_query_pos)
        space = (input_seq_len - ctx) // 2
        if mqp < 2 * num_kv_pairs:
            raise ValueError(
                f"MQAR: min_query_pos must be >= 2*num_kv_pairs (got {mqp}, need >= {2 * num_kv_pairs})"
            )
        if input_seq_len - mqp < num_kv_pairs:
            raise ValueError(
                f"MQAR: require input_seq_len - min_query_pos >= num_kv_pairs "
                f"(got L-mqp={input_seq_len - mqp}, pairs={num_kv_pairs})"
            )
        if mqp >= input_seq_len:
            raise ValueError(f"MQAR: min_query_pos must be < input_seq_len (got {mqp}, L={input_seq_len})")
        if space < num_kv_pairs:
            raise ValueError(f"MQAR: not enough slot space (space={space}) for num_kv_pairs={num_kv_pairs}")
        gap_min = max(0, math.ceil((mqp - ctx) / 2))
        if gap_min >= space:
            raise ValueError(
                f"MQAR: min_query_pos={mqp} too large for context_size={ctx}: no valid gap indices "
                f"(gap_min={gap_min}, space={space})"
            )
        if space - gap_min < num_kv_pairs:
            raise ValueError(
                f"MQAR: not enough delayed gap slots: need {num_kv_pairs} distinct gaps in "
                f"[{gap_min}, {space}), have {space - gap_min}"
            )


def multiquery_ar(
    *,
    num_examples: int,
    input_seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_passes: int,
    random_non_queries: bool,
    power_a: float,
    seed: int,
    min_query_pos: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate MQAR inputs/labels of shape ``(num_examples, input_seq_len)``.

    Labels are ``-100`` except at answer positions. See Zoology paper / ``multiquery_ar`` docstring.

    If ``min_query_pos`` is set, gap indices are sampled so each query token index ``t``
    satisfies ``t >= min_query_pos``.
    """
    validate_mqar_config(
        input_seq_len=input_seq_len,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        min_query_pos=min_query_pos,
    )

    np.random.seed(int(seed))

    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    space = (input_seq_len - context_size) // 2
    if space < num_kv_pairs:
        raise ValueError(
            f"MQAR: not enough slot space (space={space}) for num_kv_pairs={num_kv_pairs}; "
            "increase input_seq_len or reduce num_kv_pairs / num_passes."
        )
    if min_query_pos is None:
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()
        x_idx = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x_idx, replace=False, p=p, size=num_kv_pairs)
    else:
        mqp = int(min_query_pos)
        gap_min = max(0, int(math.ceil((mqp - context_size) / 2)))
        g_positions = np.arange(gap_min, space, dtype=int)
        if len(g_positions) < num_kv_pairs:
            raise ValueError(
                f"MQAR: delayed mode: need at least {num_kv_pairs} gap slots in "
                f"[{gap_min}, {space}), got {len(g_positions)}"
            )
        p_delayed = power_a * (g_positions.astype(np.float64) + 1.0) ** (power_a - 1)
        p_delayed = p_delayed / p_delayed.sum()
        x_idx_delayed = np.stack([g_positions] * num_examples)
        gaps = np.apply_along_axis(
            np.random.choice,
            axis=1,
            arr=x_idx_delayed,
            replace=False,
            p=p_delayed,
            size=num_kv_pairs,
        )

    query_width = input_seq_len - context_size + 1
    queries = np.zeros((num_examples, query_width), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels_full = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels_full, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs_np = examples[:, :-1]
    labels_np = labels_full[:, 1:]

    inputs = torch.tensor(inputs_np, dtype=torch.long)
    labels = torch.tensor(labels_np, dtype=torch.long)

    if random_non_queries:
        mask_zero = inputs == 0
        if mask_zero.any():
            rnd = torch.randint(0, vocab_size, size=inputs.shape, dtype=torch.long)
            inputs = torch.where(mask_zero, rnd, inputs)

    return inputs, labels


IGNORE_LABEL = -100


def verify_mqar_sample(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_kv_pairs: int,
    num_passes: int,
    min_query_pos: int | None = None,
) -> None:
    """Assert that every supervised position has label == KV[value] for query key at same index.

    Uses only the local key--value pairs from the context prefix of ``input_ids`` (first
    ``2 * num_kv_pairs`` tokens define the association; repeats across passes match).
    """
    if input_ids.dim() != 1 or labels.dim() != 1:
        raise ValueError("verify_mqar_sample expects 1-D tensors")
    if input_ids.shape != labels.shape:
        raise ValueError(f"shape mismatch {input_ids.shape} vs {labels.shape}")
    L = int(input_ids.shape[0])
    context_size = num_kv_pairs * 2 * num_passes
    if L < context_size:
        raise ValueError(f"L={L} < context_size={context_size}")

    kv: dict[int, int] = {}
    for i in range(num_kv_pairs):
        k = int(input_ids[2 * i].item())
        v = int(input_ids[2 * i + 1].item())
        if k in kv and kv[k] != v:
            raise ValueError(f"MQAR oracle: duplicate key {k} with inconsistent values")
        kv[k] = v

    n_sup = 0
    for t in range(L):
        lab = int(labels[t].item())
        if lab == IGNORE_LABEL:
            continue
        n_sup += 1
        if min_query_pos is not None and t < int(min_query_pos):
            raise ValueError(
                f"MQAR oracle: supervised position t={t} < min_query_pos={min_query_pos}"
            )
        qk = int(input_ids[t].item())
        if qk not in kv:
            raise ValueError(
                f"MQAR oracle: query token {qk} at t={t} not in local KV table (keys={sorted(kv.keys())})"
            )
        if lab != kv[qk]:
            raise ValueError(
                f"MQAR oracle: at t={t} label={lab} but KV[{qk}]={kv[qk]}"
            )
    if n_sup != num_kv_pairs:
        raise ValueError(
            f"MQAR oracle: expected {num_kv_pairs} supervised positions, got {n_sup}"
        )


def verify_mqar_dataset(
    ds: MQARDataset,
    *,
    num_kv_pairs: int,
    num_passes: int,
    min_query_pos: int | None = None,
) -> None:
    for i in range(len(ds)):
        x, y = ds[i]
        verify_mqar_sample(
            x,
            y,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            min_query_pos=min_query_pos,
        )


def _mqar_kv_prefix_map(input_ids: torch.Tensor, *, num_kv_pairs: int) -> dict[int, int]:
    kv: dict[int, int] = {}
    for i in range(num_kv_pairs):
        k = int(input_ids[2 * i].item())
        v = int(input_ids[2 * i + 1].item())
        kv[k] = v
    return kv


def dump_mqar_samples(
    ds: MQARDataset,
    path: Path | str,
    *,
    num_samples: int,
    num_kv_pairs: int,
    num_passes: int,
    min_query_pos: int | None = None,
    ignore_label: int = IGNORE_LABEL,
) -> None:
    """Write a human-readable MQAR debug dump (no stdout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(0, min(int(num_samples), len(ds)))
    context_size = num_kv_pairs * 2 * num_passes
    lines_out: list[str] = []
    lines_out.append("MQAR sample dump")
    lines_out.append(f"  min_query_pos: {min_query_pos if min_query_pos is not None else 'null'}")
    lines_out.append("")
    if n == 0:
        path.write_text("\n".join(lines_out), encoding="utf-8")
        return

    probe_inp, _probe_lab = ds[0]
    seq_len = int(probe_inp.numel())
    if min_query_pos is not None:
        lines_out.append(
            f"  query index range (delayed): positions t in [{min_query_pos}, {seq_len}) "
            f"where labels[t] != {ignore_label}"
        )
    else:
        lines_out.append(
            f"  query index range (default): any t in [{context_size}, {seq_len}) per generator"
        )
    lines_out.append("")

    for sample_id in range(n):
        input_ids, labels = ds[sample_id]
        inp = input_ids.long().cpu()
        lab = labels.long().cpu()
        L = int(inp.numel())
        li = [int(inp[i].item()) for i in range(L)]
        ll = [int(lab[i].item()) for i in range(L)]

        lines_out.append("")
        lines_out.append("=" * 72)
        lines_out.append(f"sample_id = {sample_id}")
        lines_out.append("")
        lines_out.append("input_ids:")
        lines_out.append(f"  {li}")
        lines_out.append("")
        lines_out.append("labels:")
        lines_out.append(f"  {ll}")

        kv = _mqar_kv_prefix_map(inp, num_kv_pairs=num_kv_pairs)
        lines_out.append("")
        lines_out.append("KV prefix (first 2 * num_kv_pairs positions):")
        for i in range(num_kv_pairs):
            k = int(inp[2 * i].item())
            v = int(inp[2 * i + 1].item())
            lines_out.append(f"  {k} -> {v}")

        query_ts: list[int] = [t for t in range(L) if int(lab[t].item()) != ignore_label]

        lines_out.append("")
        if query_ts:
            lines_out.append(
                f"Query position range (this sample): min_t={min(query_ts)} max_t={max(query_ts)} "
                f"(count={len(query_ts)})"
            )
        else:
            lines_out.append("Query position range (this sample): (no supervised positions)")
        lines_out.append("")
        lines_out.append("Query positions:")
        if len(query_ts) != num_kv_pairs:
            lines_out.append(
                f"  ERROR: expected {num_kv_pairs} query positions, got {len(query_ts)} "
                f"(indices {query_ts})"
            )
        for t in query_ts:
            q = int(inp[t].item())
            exp_lab = int(lab[t].item())
            kv_lookup = kv.get(q)
            if kv_lookup is None:
                lines_out.append(
                    f"  pos={t}, query={q}, expected_label={exp_lab}, kv_lookup=<missing>, "
                    "ERROR: query token not in KV prefix"
                )
                continue
            if exp_lab != kv_lookup:
                lines_out.append(
                    f"  pos={t}, query={q}, expected_label={exp_lab}, kv_lookup={kv_lookup}, "
                    f"ERROR: expected_label!=kv_lookup (sanity labels[t]==kv[input_ids[t]])"
                )
            else:
                lines_out.append(
                    f"  pos={t}, query={q}, expected_label={exp_lab}, kv_lookup={kv_lookup}, OK"
                )

        ann: list[str] = []
        for t in range(L):
            tid = int(inp[t].item())
            ann.append(str(tid))
            if int(lab[t].item()) != ignore_label:
                ann.append(f"[Q={tid} A={int(lab[t].item())}]")
        lines_out.append("")
        lines_out.append("Annotated sequence:")
        lines_out.append("  " + " ".join(ann))

        if L < context_size:
            lines_out.append("")
            lines_out.append(f"ERROR: seq len L={L} < context_size={context_size}")

    path.write_text("\n".join(lines_out).lstrip("\n"), encoding="utf-8")


def _mqar_row_fingerprint(input_ids: torch.Tensor, labels: torch.Tensor) -> str:
    xb = input_ids.detach().cpu().contiguous().numpy().tobytes()
    yb = labels.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(xb + yb).hexdigest()


def mqar_answer_positions_counts(ds: MQARDataset) -> list[int]:
    """Per-sample counts of supervised (non-ignore) labels."""
    out: list[int] = []
    for i in range(len(ds)):
        _, y = ds[i]
        out.append(int((y != IGNORE_LABEL).sum().item()))
    return out


def mqar_split_disjoint_report(
    train_ds: MQARDataset,
    val_ds: MQARDataset,
    test_ds: MQARDataset,
) -> dict[str, int | tuple[int, float, int]]:
    """Hashes each row as ``SHA256(inputs||labels)``; checks pairwise split overlap.

    Returns counts: ``n_train``, ``n_val``, ``n_test``, ``uniq_train``, ``uniq_val``, ``uniq_test``,
    and ``answer_pos_{split}``: ``(min, mean, max)`` of supervised label counts.

    Raises:
        ValueError: if any pairwise intersection of fingerprint sets is non-empty.
    """
    def split_stats(ds: MQARDataset) -> tuple[set[str], int, tuple[int, float, int]]:
        fps: set[str] = set()
        for i in range(len(ds)):
            x, y = ds[i]
            fps.add(_mqar_row_fingerprint(x, y))
        counts = mqar_answer_positions_counts(ds)
        if not counts:
            return fps, 0, (0, 0.0, 0)
        mn, mx = min(counts), max(counts)
        mu = sum(counts) / len(counts)
        return fps, len(ds), (mn, mu, mx)

    ht, nt, st_tr = split_stats(train_ds)
    hv, nv, st_va = split_stats(val_ds)
    hte, nte, st_te = split_stats(test_ds)

    def _inter_msg(a: str, b: str, s: set[str]) -> str:
        snip = sorted(s)[:8]
        return f"MQAR split overlap {a}∩{b}: {len(s)} rows, e.g. {snip}"

    x_tr_va = ht & hv
    if x_tr_va:
        raise ValueError(_inter_msg("train", "val", x_tr_va))
    x_tr_te = ht & hte
    if x_tr_te:
        raise ValueError(_inter_msg("train", "test", x_tr_te))
    x_va_te = hv & hte
    if x_va_te:
        raise ValueError(_inter_msg("val", "test", x_va_te))

    return {
        "n_train": nt,
        "n_val": nv,
        "n_test": nte,
        "uniq_train": len(ht),
        "uniq_val": len(hv),
        "uniq_test": len(hte),
        "answer_pos_train": st_tr,
        "answer_pos_val": st_va,
        "answer_pos_test": st_te,
    }


def log_mqar_split_report(
    rep: dict[str, int | tuple[int, float, int]],
    log: Callable[[str], None],
    *,
    num_kv_pairs: int,
) -> None:
    """Log disjoint-set report; asserts oracle-style answer counts."""
    log(
        "mqar_split_hashes: "
        f"n_train={rep['n_train']} uniq_train={rep['uniq_train']} | "
        f"n_val={rep['n_val']} uniq_val={rep['uniq_val']} | "
        f"n_test={rep['n_test']} uniq_test={rep['uniq_test']}"
    )
    for split_key, n_key, u_key in (
        ("train", "n_train", "uniq_train"),
        ("val", "n_val", "uniq_val"),
        ("test", "n_test", "uniq_test"),
    ):
        n_i, u_i = int(rep[n_key]), int(rep[u_key])  # type: ignore[arg-type]
        if u_i < n_i:
            log(f"mqar_split_hashes WARNING: duplicate samples within {split_key} (uniq={u_i} < n={n_i})")
    for split in ("train", "val", "test"):
        key = f"answer_pos_{split}"
        tup = rep[key]
        if not isinstance(tup, tuple):
            raise TypeError(key)
        mn_i, mu_f, mx_i = int(tup[0]), float(tup[1]), int(tup[2])
        log(
            f"mqar_answer_positions[{split}]: min={mn_i} mean={mu_f:.4f} max={mx_i} "
            f"(expected {num_kv_pairs} per sample)"
        )
        if mn_i != num_kv_pairs or mx_i != num_kv_pairs:
            raise ValueError(
                f"MQAR answer position count mismatch on {split}: "
                f"min={mn_i} max={mx_i} expected {num_kv_pairs}"
            )


def build_mqar_dataset(
    *,
    num_examples: int,
    input_seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_passes: int,
    random_non_queries: bool,
    power_a: float,
    seed: int,
    min_query_pos: int | None = None,
) -> MQARDataset:
    inp, lab = multiquery_ar(
        num_examples=num_examples,
        input_seq_len=input_seq_len,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
        power_a=power_a,
        seed=seed,
        min_query_pos=min_query_pos,
    )
    return MQARDataset(inp, lab)


def build_mqar_train_val_test(
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    input_seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_passes: int,
    random_non_queries: bool,
    power_a: float,
    seed: int,
    fixed_examples: bool,
    min_query_pos: int | None = None,
) -> tuple[MQARDataset, MQARDataset, MQARDataset]:
    """Build train/val/test from the same generative process.

    If ``fixed_examples`` is True, one call to ``multiquery_ar`` produces
    ``train_n + val_n + test_n`` rows with a single RNG stream; slices are disjoint.
    If False, three independent runs with seeds ``seed``, ``seed+10_000``, ``seed+20_000``.
    """
    if fixed_examples:
        total = int(train_n) + int(val_n) + int(test_n)
        inp, lab = multiquery_ar(
            num_examples=total,
            input_seq_len=input_seq_len,
            vocab_size=vocab_size,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            random_non_queries=random_non_queries,
            power_a=power_a,
            seed=seed,
            min_query_pos=min_query_pos,
        )
        a, b = train_n, train_n + val_n
        return (
            MQARDataset(inp[:a], lab[:a]),
            MQARDataset(inp[a:b], lab[a:b]),
            MQARDataset(inp[b:], lab[b:]),
        )
    return (
        build_mqar_dataset(
            num_examples=train_n,
            input_seq_len=input_seq_len,
            vocab_size=vocab_size,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            random_non_queries=random_non_queries,
            power_a=power_a,
            seed=seed,
            min_query_pos=min_query_pos,
        ),
        build_mqar_dataset(
            num_examples=val_n,
            input_seq_len=input_seq_len,
            vocab_size=vocab_size,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            random_non_queries=random_non_queries,
            power_a=power_a,
            seed=seed + 10_000,
            min_query_pos=min_query_pos,
        ),
        build_mqar_dataset(
            num_examples=test_n,
            input_seq_len=input_seq_len,
            vocab_size=vocab_size,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            random_non_queries=random_non_queries,
            power_a=power_a,
            seed=seed + 20_000,
            min_query_pos=min_query_pos,
        ),
    )
