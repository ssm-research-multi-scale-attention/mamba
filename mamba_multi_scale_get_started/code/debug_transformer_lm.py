#!/usr/bin/env python3
"""Sanity checks for transformer_lm: causal mask, shapes, CE, tiny overfit on one batch."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_CODE = Path(__file__).resolve().parent
_PROJECT = _CODE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.tiny_shakespeare import build_char_lm_dataset  # noqa: E402
from exp_config import load_config, set_seed  # noqa: E402
from models.language_models import TransformerLanguageModel, build_lm  # noqa: E402


def lm_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Same reduction as train_lm.lm_token_cross_entropy."""
    v = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1))


def causal_mask_checks() -> None:
    print("=== Causal mask (T=8) ===")
    t = 8
    mask = TransformerLanguageModel._causal_self_attn_mask(
        t, device=torch.device("cpu"), dtype=torch.float32
    )
    assert mask.shape == (t, t)
    for i in range(t):
        for j in range(t):
            v = mask[i, j].item()
            if j > i:
                assert not math.isfinite(v) or v <= -1e30, (i, j, v)
            else:
                assert math.isfinite(v) and abs(v) < 1e-6, (i, j, v)
    print(f"upper triangle all -inf, lower+diag finite: OK\nmask=\n{mask}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config with model.backbone=transformer_lm",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--overfit-steps", type=int, default=15)
    args = p.parse_args()

    causal_mask_checks()

    cfg = load_config(args.config, [])
    if str(cfg.model.backbone).strip().lower() != "transformer_lm":
        print("Warning: backbone is not transformer_lm; build_lm output may differ.", file=sys.stderr)

    seed = int(cfg.experiment.seed)
    set_seed(seed)
    device = torch.device(args.device)

    train_loader, _val_loader, vocab = build_char_lm_dataset(
        dataset="tiny_shakespeare",
        data_dir=_PROJECT / str(cfg.data.data_dir),
        block_size=int(cfg.data.block_size),
        batch_size=int(cfg.loader.batch_size),
        train_ratio=float(cfg.data.train_ratio),
        num_workers=0,
        sampling=str(cfg.data.sampling),
        steps_per_epoch=min(256, int(cfg.data.steps_per_epoch)),
        seed=seed,
    )
    cfg.model.vocab_size = len(vocab)

    model = build_lm(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Model ===\nparameters={nparams}")

    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    b, tk = x.shape
    logits = model(x)
    print(f"batch shape x={tuple(x.shape)} logits={tuple(logits.shape)} (expect [B,T,{len(vocab)}])")
    assert logits.shape == (b, tk, len(vocab))
    loss0 = lm_ce(logits, y)
    print(f"loss (random-ish init) CE={loss0.item():.6f} (ln(vocab)≈{math.log(len(vocab)):.6f})")
    assert torch.isfinite(loss0)

    # Position indices
    tmax = int(cfg.model.transformer.max_seq_len)
    assert tk <= tmax

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=0.01)
    losses = []
    for step in range(args.overfit_steps):
        opt.zero_grad(set_to_none=True)
        logits_s = model(x)
        ls = lm_ce(logits_s, y)
        ls.backward()
        opt.step()
        losses.append(ls.item())
        print(f"  overfit step {step + 1}: CE={losses[-1]:.6f}")
    dropped = losses[0] - losses[-1]
    print(f"CE drop on same batch over {args.overfit_steps} steps: {dropped:.6f} (expect > 0)")
    if dropped < 0.01:
        print("WARNING: very small CE drop — check LR/implementation.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
