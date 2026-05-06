#!/usr/bin/env python3
"""One-batch diagnostics for multiscale_mamba2_window_summary_lm."""
from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.mqar import build_mqar_train_val_test  # noqa: E402
from exp_config import load_config, set_seed  # noqa: E402
from models.language_models import build_lm  # noqa: E402

IGNORE = -100


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.argument("overrides", nargs=-1)
def main(config_path: str, overrides: tuple[str, ...]) -> None:
    cfg = load_config(config_path, list(overrides))
    set_seed(int(cfg.experiment.seed))
    cfg.model.vocab_size = int(cfg.data.vocab_size)
    cfg.model.multiscale.diagnostics = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_ds, _, _ = build_mqar_train_val_test(
        train_n=int(cfg.data.train_examples),
        val_n=max(8, int(cfg.data.val_examples)),
        test_n=max(8, int(cfg.data.test_examples)),
        input_seq_len=int(cfg.data.input_seq_len),
        vocab_size=int(cfg.data.vocab_size),
        num_kv_pairs=int(cfg.data.num_kv_pairs),
        num_passes=int(OmegaConf.select(cfg, "data.num_passes", default=1)),
        random_non_queries=bool(OmegaConf.select(cfg, "data.random_non_queries", default=True)),
        power_a=float(OmegaConf.select(cfg, "data.power_a", default=0.01)),
        seed=int(cfg.experiment.seed),
        fixed_examples=bool(OmegaConf.select(cfg, "data.fixed_examples", default=False)),
        min_query_pos=OmegaConf.select(cfg, "data.min_query_pos", default=None),
    )
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    x, y = next(iter(loader))
    model = build_lm(cfg).to(device).eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        logits = model(x)

    print("logits shape:", tuple(logits.shape))
    answer_positions = (y[0] != IGNORE).nonzero(as_tuple=False).squeeze(-1).tolist()
    print("answer positions sample0:", answer_positions[:32])
    dbg = getattr(model, "last_window_debug", {})
    print("diagnostics:")
    for k in (
        "window_size",
        "offsets",
        "summary_mode",
        "causal_mode",
        "fusion",
        "slow_up_nonzero_fraction",
        "fast_norm_mean",
        "fast_norm_std",
        "slow_norm_mean",
        "slow_norm_std",
        "gate_mean",
        "gate_std",
        "nan_in_fast",
        "nan_in_logits",
    ):
        print(f"  {k}: {dbg.get(k)}")
    print("  gate_sample:", dbg.get("gate_sample", []))
    print("mapping sample (token->window):")
    for row in dbg.get("mapping_sample", [])[:80]:
        print(
            " ",
            f"t={row['token']:>3} off={row['offset']} idx={row['window_index']}",
            f"win=[{row['window_start']},{row['window_end']}]",
        )
    # explicit causal verification in printed sample
    bad = [
        row
        for row in dbg.get("mapping_sample", [])
        if int(row["window_end"]) > int(row["token"])
        and str(dbg.get("causal_mode", "completed")) == "completed"
    ]
    print("causal violations in sample:", len(bad))


if __name__ == "__main__":
    main()
