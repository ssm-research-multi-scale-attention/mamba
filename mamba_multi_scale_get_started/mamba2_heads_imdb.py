#!/usr/bin/env python3
"""
Train from config: Hugging Face text loaders + Lightning module path in YAML.

Uses OmegaConf merged with CLI overrides. Default recipe: Mamba multi-head LM on IMDB.

Examples:
  python mamba2_heads_imdb.py
  python mamba2_heads_imdb.py --config path/to/config.yaml
  python mamba2_heads_imdb.py train.epochs=5 data.eval_subset=2000 experiment.name=quick
"""
from __future__ import annotations

import os

import click

from exp_config import load_config, package_directory
from experiment_runner import run_experiment


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=lambda: os.path.join(package_directory(), "configs", "config.yaml"),
    show_default="config.yaml next to script",
    help="Path to YAML config.",
)
@click.argument("overrides", nargs=-1)
def main(config_path: str, overrides: tuple[str, ...]) -> None:
    """Run an experiment from YAML + Lightning."""
    cfg = load_config(config_path, list(overrides))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
