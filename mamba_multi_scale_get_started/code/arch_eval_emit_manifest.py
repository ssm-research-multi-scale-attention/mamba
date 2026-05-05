#!/usr/bin/env python3
"""Emit TSV manifest lines for run_arch_eval.sh (validate config paths exist)."""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

_CODE = Path(__file__).resolve().parent
ROOT = _CODE.parent


def main() -> int:
    reg_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "configs/EvalRegistry/architectures.yaml"
    if not reg_path.is_file():
        print(f"arch_eval_emit_manifest: missing registry {reg_path}", file=sys.stderr)
        return 1

    cfg = OmegaConf.load(reg_path)
    archs = OmegaConf.select(cfg, "architectures", default=None)
    if archs is None:
        print("arch_eval_emit_manifest: no architectures:", reg_path, file=sys.stderr)
        return 1
    mqar_settings = [
        ("easy", "512", "null"),
        ("trans704", "704", "null"),
        ("trans768", "768", "null"),
    ]
    seeds = (42, 43, 44)

    lm_out: list[str] = []
    mqar_out: list[str] = []
    timing_out: list[str] = []

    for a in archs:
        name = str(a.name)
        atype = str(a.type)
        for key in ("lm_config", "mqar_config", "timing_config"):
            rel = Path(str(OmegaConf.select(a, key, default="")))
            ap = ROOT / rel
            if not ap.is_file():
                print(f"arch_eval_emit_manifest: missing file for {name} {key}={rel}", file=sys.stderr)
                return 1

        lm = str(a.lm_config)
        mqar = str(a.mqar_config)
        timing = str(a.timing_config)
        lm_out.append(f"lm\t{name}\t{atype}\t{lm}")
        timing_out.append(f"timing\t{name}\t{atype}\t{timing}")
        for setting, vocab, mnul in mqar_settings:
            for seed in seeds:
                mqar_out.append(f"mqar\t{name}\t{atype}\t{mqar}\t{setting}\t{seed}\t{vocab}\t{mnul}")

    for line in lm_out + mqar_out + timing_out:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
