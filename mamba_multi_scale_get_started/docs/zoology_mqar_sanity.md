# Zoology MQAR external sanity check

This compares qualitative MQAR behaviour against **upstream [HazyResearch/zoology](https://github.com/HazyResearch/zoology)** (same synthetic task family as our ported `code/datasets/mqar.py`). It does **not** change ArchEval or `train_mqar.py`.

## Where Zoology lives in this workspace

| Location | Note |
|----------|------|
| `repos/zoology/` | Full clone next to `repos/mamba/` (recommended). |
| `mamba_multi_scale_get_started/external/zoology` | Optional symlink or extra clone — used if present. |

Default resolver in `code/zoology_mqar_sanity.py`: `../../zoology` from the **mamba_multi_scale_get_started** project root → `repos/zoology`.

## Environment

- Conda env example: **`nvidenisov-other-4`** (Zoology already installed there per project notes).
- Otherwise: `pip install -e /path/to/repos/zoology` (see upstream README; optional `[extra]` for Mamba).

Dependencies: at minimum **torch** plus Zoology’s base deps (`einops`, `pydantic`, `tqdm`, …). **mamba_ssm** / **mamba2** needed only for `mamba2` runs in the sanity script.

## Commands

From `mamba_multi_scale_get_started` root:

```bash
# Optional: explicit repo path (must contain the `zoology` package)
export ZOOLOGY_ROOT=/path/to/repos/zoology

chmod +x runners/mqar/run_zoology_mqar_sanity.sh
./runners/mqar/run_zoology_mqar_sanity.sh
```

Direct Python (same logic):

```bash
export PYTHONPATH="$ZOOLOGY_ROOT:$PYTHONPATH"
python code/zoology_mqar_sanity.py --output-root outputs/ZoologyMQAR --models mha,mamba2 --seeds 42,43,44
python code/aggregate_zoology_mqar.py
```

Environment knobs for the shell runner:

| Variable | Meaning |
|----------|---------|
| `SEEDS` | Default `42,43,44` |
| `MODELS` | Default `mha,mamba2` (add `mamba` for Mamba v1 fallback) |
| `REGIMES` | Default `easy,trans704,trans768` (`easy` = vocab 512) |
| `INCLUDE_HARD=1` | Also run vocab 1024 |

## Output layout (isolated from ArchEval)

| Path | Content |
|------|---------|
| `outputs/ZoologyMQAR/<regime>_<model>_v<V>_seed<S>/` | `results.json`, `train_config.json` |
| `outputs/ZoologyMQAR/summary_zoology_mqar.csv` | Aggregated table |
| `outputs/ZoologyMQAR/report_zoology_mqar.txt` | Short qualitative summary |
| `logs/zoology_mqar/zmq_sanity_*.log` | Full stdout |

## Regime mapping vs our ArchEval names

| Sanity regime | `vocab_size` | ArchEval `setting` |
|---------------|--------------|---------------------|
| `easy` | 512 | `easy` |
| `trans704` | 704 | `trans704` |
| `trans768` | 768 | `trans768` |
| `hard` | 1024 | (optional; we often use 512–768 in sweeps) |

Shared: **`input_seq_len=128`**, **`num_kv_pairs=16`**, **`num_examples` train 20k / test 2k** (aligned with our MQAR configs).

## Models used in the wrapper

- **`mha`**: Zoology `zoology.mixers.attention.MHA` (Transformer baseline).
- **`mamba2`**: `zoology.mixers.mamba2.Mamba2` when imports succeed; otherwise runs are **skipped** with `status=skipped: mamba2_import_failed`.
- **`mamba`**: `zoology.mixers.mamba.Mamba` — use if Mamba2 is unavailable.

## Metric comparability

- Zoology reports **`valid/accuracy`** (mean token accuracy where `label != -100`).  
- Our `train_mqar.py` may log additional MQAR-specific metrics; numbers will not match bit-for-bit. Compare **easy vs transition vs hard** trends only.
