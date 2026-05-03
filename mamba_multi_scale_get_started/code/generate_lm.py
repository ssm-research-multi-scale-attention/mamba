#!/usr/bin/env python3
"""Autoregressive character generation from a trained LM checkpoint."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.tiny_shakespeare import CharVocab
from models.language_models import build_lm

PROJECT_ROOT = _CODE.parent


def _device(s: str) -> torch.device:
    s = str(s).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[..., [-1]]
    return logits.masked_fill(logits < thresh, float("-inf"))


@torch.no_grad()
def _sample_next(
    logits_1v: torch.Tensor,
    temperature: float,
    top_k: int,
) -> int:
    logits = logits_1v / max(float(temperature), 1e-8)
    logits = _top_k_filter(logits, int(top_k))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=str, default=None, help="YAML config (optional if checkpoint embeds cfg).")
@click.option("--checkpoint", "checkpoint_path", type=click.Path(exists=True, path_type=str), required=True)
@click.option("--vocab", "vocab_path", type=str, default=None, help="vocab.json (default: next to checkpoint).")
@click.option("--prompt", type=str, default="")
@click.option("--max-new-tokens", type=int, default=200)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top-k", type=int, default=0)
@click.option("--device", type=str, default="auto")
def main(
    config_path: str | None,
    checkpoint_path: str,
    vocab_path: str | None,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> None:
    ckpt_path = Path(checkpoint_path).resolve()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if config_path:
        cfg = OmegaConf.load(config_path)
    else:
        cfg_yaml = ckpt.get("cfg")
        if not isinstance(cfg_yaml, str):
            raise ValueError("Checkpoint has no embedded cfg string; pass --config.")
        cfg = OmegaConf.load(StringIO(cfg_yaml))

    vp = Path(vocab_path).resolve() if vocab_path else ckpt_path.parent / "vocab.json"
    vocab = CharVocab.from_json(json.loads(vp.read_text(encoding="utf-8")))

    if not prompt:
        raise click.UsageError("--prompt must be non-empty (empty context breaks the LM forward).")

    cfg.model.vocab_size = len(vocab)
    model = build_lm(cfg).to(_device(device))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dev = next(model.parameters()).device
    ids = vocab.encode(prompt)
    block_cap = int(OmegaConf.select(cfg, "data.block_size", default=256))

    out: list[int] = list(ids)
    for _ in range(int(max_new_tokens)):
        inp = out[-block_cap:] if len(out) > block_cap else out
        x = torch.tensor([inp], dtype=torch.long, device=dev)
        logits = model(x)
        nxt = _sample_next(logits[0, -1], temperature, top_k)
        out.append(nxt)

    text = vocab.decode(out)
    print(text, flush=True)


if __name__ == "__main__":
    main()
