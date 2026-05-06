#!/usr/bin/env python3
"""
Run Zoology upstream MQAR sanity experiments (external Zoology install or ZOOLOGY_ROOT).
Writes per-run artifacts under outputs/ZoologyMQAR/ — does not touch ArchEval outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_zoology_to_path(zoology_root: Path) -> None:
    root_str = str(zoology_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _trainer_subclass():
    from torch import nn
    from torch import optim
    import torch

    from zoology.train import Trainer

    class TrainerWithLastMetrics(Trainer):
        """Same as Zoology Trainer.fit but returns final validation metrics."""

        def fit(self) -> dict:
            self.model.to(self.device)
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs, eta_min=0.0
            )
            last: dict = {}
            for epoch_idx in range(self.max_epochs):
                self.train_epoch(epoch_idx)
                last = self.test(epoch_idx)
                if (self.early_stopping_metric is not None) and last[
                    self.early_stopping_metric
                ] > self.early_stopping_threshold:
                    print(
                        f"Early stopping triggered at epoch {epoch_idx} with "
                        f"{self.early_stopping_metric} "
                        f"{last[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                    )
                    break
                self.scheduler.step()
            return last

    return TrainerWithLastMetrics


def train_zoology_run(config, artifact_dir: Path) -> dict:
    """Mirror zoology.train.train but save metrics + config to artifact_dir."""
    import torch
    from zoology.config import TrainConfig
    from zoology.data.utils import prepare_data
    from zoology.logger import WandbLogger
    from zoology.model import LanguageModel
    from zoology.utils import set_determinism

    TrainerWithLastMetrics = _trainer_subclass()

    set_determinism(config.seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(config)
    logger.log_config(config)

    model = LanguageModel(config.model)
    train_dataloader, test_dataloader = prepare_data(config.data)
    logger.log_model(model, config=config)

    task = TrainerWithLastMetrics(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )
    try:
        last_metrics = task.fit()
        status = "ok"
    except Exception as e:
        last_metrics = {"error": str(e)}
        status = f"failed: {e!r}"

    out = {
        "status": status,
        "metrics_final": last_metrics if isinstance(last_metrics, dict) else {},
        "run_id": config.run_id,
        "seed": config.seed,
    }
    (artifact_dir / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        cfg_json = config.model_dump_json(indent=2)
    except AttributeError:
        cfg_json = config.json(indent=2)
    (artifact_dir / "train_config.json").write_text(cfg_json, encoding="utf-8")
    logger.finish()
    return out


def _model_mha(vocab_size: int, input_seq_len: int) -> "ModelConfig":
    from zoology.config import ModelConfig, ModuleConfig

    return ModelConfig(
        name="zoology_mha_mqar_sanity",
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        max_position_embeddings=input_seq_len,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 4},
        ),
        state_mixer=ModuleConfig(
            name="zoology.mixers.mlp.MLP",
            kwargs={"hidden_mult": 4},
        ),
        block_type="TransformerBlock",
        resid_dropout=0.0,
        embed_dropout=0.1,
    )


def _model_mamba2(vocab_size: int, input_seq_len: int) -> "ModelConfig":
    from zoology.config import ModelConfig, ModuleConfig

    return ModelConfig(
        name="zoology_mamba2_mqar_sanity",
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        max_position_embeddings=0,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.mamba2.Mamba2",
            kwargs={"d_state": 16},
        ),
        state_mixer=ModuleConfig(
            name="zoology.mixers.mlp.MLP",
            kwargs={"hidden_mult": 4},
        ),
        block_type="Mamba2Block",
        resid_dropout=0.0,
        embed_dropout=0.1,
    )


def _model_mamba(vocab_size: int, input_seq_len: int) -> "ModelConfig":
    from zoology.config import ModelConfig, ModuleConfig

    return ModelConfig(
        name="zoology_mamba_mqar_sanity",
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        max_position_embeddings=0,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.mamba.Mamba",
            kwargs={"d_state": 16},
        ),
        state_mixer=ModuleConfig(
            name="zoology.mixers.mlp.MLP",
            kwargs={"hidden_mult": 4},
        ),
        block_type="MambaBlock",
        resid_dropout=0.0,
        embed_dropout=0.1,
    )


def build_train_config(
    *,
    regime: str,
    vocab_size: int,
    seed: int,
    model_key: str,
    train_examples: int = 20_000,
    test_examples: int = 2_000,
    input_seq_len: int = 128,
    num_kv_pairs: int = 16,
    max_epochs: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
) -> "TrainConfig":
    from zoology.config import TrainConfig, DataConfig, LoggerConfig
    from zoology.data.multiquery_ar import MQARConfig

    builders = {
        "mha": _model_mha,
        "mamba2": _model_mamba2,
        "mamba": _model_mamba,
    }
    if model_key not in builders:
        raise ValueError(f"model_key must be one of {sorted(builders)}, got {model_key!r}")
    model = builders[model_key](vocab_size, input_seq_len)
    run_id = f"zmq_{regime}_{model_key}_v{vocab_size}_seed{seed}"

    factory_kwargs = {
        "power_a": 0.01,
        "num_kv_pairs": num_kv_pairs,
        "random_non_queries": True,
        "num_passes": 1,
    }
    data = DataConfig(
        train_configs=[
            MQARConfig(
                num_examples=train_examples,
                vocab_size=vocab_size,
                input_seq_len=input_seq_len,
                **factory_kwargs,
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=test_examples,
                vocab_size=vocab_size,
                input_seq_len=input_seq_len,
                **factory_kwargs,
            )
        ],
        batch_size=128,
        seed=seed,
        cache_dir=None,
        num_passes=1,
    )

    return TrainConfig(
        run_id=run_id,
        seed=seed,
        data=data,
        model=model,
        logger=LoggerConfig(project_name=None, entity=None),
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_metric="valid/accuracy",
        early_stopping_threshold=0.99,
        slice_keys=["num_kv_pairs"],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zoology-root",
        type=Path,
        default=None,
        help="Path to Zoology repo (optional; for PYTHONPATH if package not installed)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ZoologyMQAR"),
        help="Root directory for run artifacts",
    )
    p.add_argument(
        "--models",
        type=str,
        default="mha,mamba2",
        help="Comma list: mha, mamba2, mamba (mamba2 skipped if import fails)",
    )
    p.add_argument(
        "--regimes",
        type=str,
        default="easy,trans704,trans768",
        help="Comma list: easy (vocab512), trans704, trans768; add hard for vocab1024",
    )
    p.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    p.add_argument(
        "--include-hard",
        action="store_true",
        help="Also run vocab_size=1024 (hard) if listed in regimes",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    zr = args.zoology_root
    if zr is None:
        # mamba_multi_scale_get_started -> ../../zoology == repos/zoology
        cand = (project_root / ".." / ".." / "zoology").resolve()
        if cand.is_dir():
            zr = cand
        else:
            ext = (project_root / "external" / "zoology").resolve()
            zr = ext if ext.is_dir() else None
    if zr and Path(zr).is_dir():
        _add_zoology_to_path(Path(zr))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    models_requested = [x.strip() for x in args.models.split(",") if x.strip()]

    regime_map = {
        "easy": 512,
        "trans704": 704,
        "trans768": 768,
        "hard": 1024,
    }
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    if args.include_hard and "hard" not in regimes:
        regimes.append("hard")

    out_root = (project_root / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Probe mamba2
    mamba2_ok = True
    try:
        import zoology.mixers.mamba2  # noqa: F401
    except Exception as e:
        mamba2_ok = False
        print(f"Note: Mamba2 mixer not available ({e}); runs with model mamba2 will be skipped or fail.")

    for regime in regimes:
        if regime not in regime_map:
            print(f"Unknown regime {regime!r}, skip", file=sys.stderr)
            continue
        vocab = regime_map[regime]
        for model_key in models_requested:
            if model_key == "mamba2" and not mamba2_ok:
                for seed in seeds:
                    sub = out_root / f"{regime}_{model_key}_v{vocab}_seed{seed}"
                    sub.mkdir(parents=True, exist_ok=True)
                    err = {
                        "status": "skipped: mamba2_import_failed",
                        "metrics_final": {},
                        "run_id": f"zmq_{regime}_{model_key}_v{vocab}_seed{seed}",
                        "seed": seed,
                    }
                    (sub / "results.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
                continue
            for seed in seeds:
                sub = out_root / f"{regime}_{model_key}_v{vocab}_seed{seed}"
                cfg = build_train_config(
                    regime=regime,
                    vocab_size=vocab,
                    seed=seed,
                    model_key=model_key,
                )
                print(f"=== {cfg.run_id} -> {sub} ===")
                train_zoology_run(cfg, sub)

    print(f"Done. Artifacts under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
