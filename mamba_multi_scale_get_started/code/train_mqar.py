#!/usr/bin/env python3
"""Train causal LM on MQAR (Multi-Query Associative Recall) synthetic data."""
from __future__ import annotations

import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.mqar import (
    build_mqar_train_val_test,
    dump_mqar_samples,
    log_mqar_split_report,
    mqar_split_disjoint_report,
    verify_mqar_dataset,
)
from exp_config import load_config, set_seed
from models.language_models import build_lm

PROJECT_ROOT = _CODE.parent
IGNORE = -100


def _resolve_lm_device(cfg: DictConfig) -> torch.device:
    raw = str(OmegaConf.select(cfg, "device", default="auto")).strip()
    rl = raw.lower()
    cuda_idx_cfg = int(OmegaConf.select(cfg, "cuda_device", default=0))

    if rl in ("auto", ""):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        dev = torch.device(f"cuda:{cuda_idx_cfg}")
        torch.cuda.set_device(dev)
        return dev

    if rl == "cpu":
        return torch.device("cpu")

    if ":" in raw and raw.lower().startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={raw!r} but CUDA is not available.")
        dev = torch.device(raw)
        torch.cuda.set_device(dev)
        return dev

    if rl in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={raw!r} but CUDA is not available.")
        dev = torch.device(f"cuda:{cuda_idx_cfg}")
        torch.cuda.set_device(dev)
        return dev

    raise ValueError(
        f"Unknown device={raw!r}. Use auto, cpu, cuda, gpu, or cuda:N. "
        f"For a physical index with cuda/gpu, set cuda_device (int), default 0."
    )


def _resolve_out_dir(cfg: DictConfig) -> Path:
    raw = str(cfg.logging.output_dir)
    return (PROJECT_ROOT / raw).resolve()


def _resolve_log_dir(cfg: DictConfig) -> Path:
    name = str(cfg.experiment.name)
    return (PROJECT_ROOT / "logs" / "mqar" / name).resolve()


def mqar_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=IGNORE,
    )


def _fmt_cfg_seq(cfg: DictConfig, key: str, default: str = "—") -> str:
    v = OmegaConf.select(cfg, key, default=None)
    if v is None or v == []:
        return default
    if OmegaConf.is_list(v):
        return str(OmegaConf.to_object(v))
    return str(v)


def _mqar_log_startup_resolved(
    cfg: DictConfig,
    *,
    log_line,
    vocab_data: int,
    num_params: int,
    tags: dict[str, str],
    input_seq_len: int,
    num_kv_pairs: int,
    train_n: int,
    val_n: int,
    test_n: int,
    fixed_examples: bool,
    min_query_pos: int | None,
) -> None:
    log_line("--- mqar resolved config (sanity) ---")
    log_line(f"  experiment.name: {cfg.experiment.name}")
    log_line(f"  model.backbone: {OmegaConf.select(cfg, 'model.backbone', default='')}")
    log_line(f"  model.layer_headdims: {_fmt_cfg_seq(cfg, 'model.layer_headdims')}")
    log_line(f"  model.multiscale.fast_layer_headdims: {_fmt_cfg_seq(cfg, 'model.multiscale.fast_layer_headdims')}")
    log_line(f"  model.multiscale.slow_layer_headdims: {_fmt_cfg_seq(cfg, 'model.multiscale.slow_layer_headdims')}")
    log_line(f"  data.input_seq_len: {input_seq_len}")
    log_line(f"  data.num_kv_pairs: {num_kv_pairs}")
    log_line(f"  data.vocab_size: {vocab_data}")
    log_line(f"  data.train_examples: {train_n}  val_examples: {val_n}  test_examples: {test_n}")
    log_line(f"  data.fixed_examples: {fixed_examples}")
    log_line(f"  data.min_query_pos: {min_query_pos if min_query_pos is not None else 'null'}")
    log_line(f"  train.lr: {float(cfg.train.lr)}")
    log_line(f"  train.epochs: {int(cfg.train.epochs)}")
    log_line(f"  num_parameters_trainable: {num_params}")
    log_line(f"  tags(backbone,fusion,stride): {tags['backbone']}, {tags['fusion']}, {tags['stride']}")
    log_line("--- end resolved config ---")


def _model_tags(cfg: DictConfig) -> dict[str, str]:
    backbone = str(OmegaConf.select(cfg, "model.backbone", default="")).strip()
    fusion_v = OmegaConf.select(cfg, "model.multiscale.fusion", default="")
    fusion_s = str(fusion_v).strip() if fusion_v is not None else ""
    stride_v = OmegaConf.select(cfg, "model.multiscale.stride", default="")
    stride_s = str(int(stride_v)) if stride_v not in (None, "") else ""
    return {"backbone": backbone, "fusion": fusion_s, "stride": stride_s}


@torch.no_grad()
def mqar_epoch_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_answer = 0
    correct = 0
    em_correct = 0
    n_em = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        mask = y != IGNORE
        loss_b = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=IGNORE,
            reduction="sum",
        )
        loss_sum += float(loss_b.item())
        n_answer += int(mask.sum().item())
        pred = logits.argmax(dim=-1)
        correct += int((pred[mask] == y[mask]).sum().item())
        for b in range(y.size(0)):
            mb = mask[b]
            if not mb.any():
                continue
            em_correct += int((pred[b, mb] == y[b, mb]).all().item())
            n_em += 1

    ans_loss = loss_sum / max(n_answer, 1)
    acc = correct / max(n_answer, 1)
    em = em_correct / max(n_em, 1)
    return {
        "answer_loss": float(ans_loss),
        "answer_ppl": float(math.exp(ans_loss)),
        "answer_accuracy": float(acc),
        "exact_match": float(em),
    }


def _append_metrics_csv(path: Path, row: dict[str, float | int | str], write_header: bool) -> None:
    fieldnames = ["epoch", "split", "answer_loss", "answer_ppl", "answer_accuracy", "exact_match"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row[k] for k in fieldnames})


def _write_test_metrics_csv(
    path: Path,
    loss_ckpt_metrics: dict[str, float],
    acc_ckpt_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "test_loss_ckpt_answer_loss",
        "test_loss_ckpt_answer_ppl",
        "test_loss_ckpt_answer_accuracy",
        "test_loss_ckpt_exact_match",
        "test_acc_ckpt_answer_loss",
        "test_acc_ckpt_answer_ppl",
        "test_acc_ckpt_answer_accuracy",
        "test_acc_ckpt_exact_match",
        "answer_loss",
        "answer_ppl",
        "answer_accuracy",
        "exact_match",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        row = {
            "split": "test",
            "test_loss_ckpt_answer_loss": f"{loss_ckpt_metrics['answer_loss']:.8f}",
            "test_loss_ckpt_answer_ppl": f"{loss_ckpt_metrics['answer_ppl']:.8f}",
            "test_loss_ckpt_answer_accuracy": f"{loss_ckpt_metrics['answer_accuracy']:.8f}",
            "test_loss_ckpt_exact_match": f"{loss_ckpt_metrics['exact_match']:.8f}",
            "test_acc_ckpt_answer_loss": f"{acc_ckpt_metrics['answer_loss']:.8f}",
            "test_acc_ckpt_answer_ppl": f"{acc_ckpt_metrics['answer_ppl']:.8f}",
            "test_acc_ckpt_answer_accuracy": f"{acc_ckpt_metrics['answer_accuracy']:.8f}",
            "test_acc_ckpt_exact_match": f"{acc_ckpt_metrics['exact_match']:.8f}",
            "answer_loss": f"{loss_ckpt_metrics['answer_loss']:.8f}",
            "answer_ppl": f"{loss_ckpt_metrics['answer_ppl']:.8f}",
            "answer_accuracy": f"{loss_ckpt_metrics['answer_accuracy']:.8f}",
            "exact_match": f"{loss_ckpt_metrics['exact_match']:.8f}",
        }
        w.writerow(row)


def _write_meta_metrics_csv(path: Path, row: dict[str, str | int | float]) -> None:
    fieldnames = [
        "experiment_name",
        "output_dir",
        "input_seq_len",
        "num_kv_pairs",
        "vocab_size",
        "min_query_pos",
        "seed",
        "lr",
        "backbone",
        "fusion",
        "stride",
        "d_model",
        "num_parameters_trainable",
        "epochs_planned",
        "epochs_completed",
        "best_val_answer_loss",
        "best_val_answer_accuracy",
        "test_answer_loss",
        "test_answer_accuracy",
        "test_exact_match",
        "test_loss_ckpt_answer_loss",
        "test_loss_ckpt_answer_accuracy",
        "test_loss_ckpt_exact_match",
        "test_acc_ckpt_answer_loss",
        "test_acc_ckpt_answer_accuracy",
        "test_acc_ckpt_exact_match",
        "random_baseline_accuracy",
        "wall_training_loop_sec",
        "finished_at_utc",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Path to MQAR YAML config.",
)
@click.argument("overrides", nargs=-1)
def main(config_path: str, overrides: tuple[str, ...]) -> None:
    cfg = load_config(config_path, list(overrides))
    exp_name = str(cfg.experiment.name)
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    device = _resolve_lm_device(cfg)
    tags = _model_tags(cfg)

    input_seq_len = int(cfg.data.input_seq_len)
    num_kv_pairs = int(cfg.data.num_kv_pairs)
    vocab_data = int(cfg.data.vocab_size)
    train_n = int(cfg.data.train_examples)
    val_n = int(cfg.data.val_examples)
    test_n = int(cfg.data.test_examples)
    num_passes = int(OmegaConf.select(cfg, "data.num_passes", default=1))
    random_non_queries = bool(OmegaConf.select(cfg, "data.random_non_queries", default=True))
    power_a = float(OmegaConf.select(cfg, "data.power_a", default=0.01))
    fixed_examples = bool(OmegaConf.select(cfg, "data.fixed_examples", default=False))

    mq_raw = OmegaConf.select(cfg, "data.min_query_pos", default=None)
    min_query_pos: int | None
    if mq_raw is None:
        min_query_pos = None
    elif isinstance(mq_raw, str) and mq_raw.strip().lower() in ("null", "none", ""):
        min_query_pos = None
    else:
        min_query_pos = int(mq_raw)

    cfg.model.vocab_size = vocab_data
    d_model = int(cfg.model.d_model)

    random_baseline_acc = 1.0 / float(vocab_data)

    out_dir = _resolve_out_dir(cfg)
    log_dir = _resolve_log_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_txt = log_dir / "train.log"

    def log_line(msg: str) -> None:
        print(msg)
        with log_txt.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    train_ds, val_ds, test_ds = build_mqar_train_val_test(
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        input_seq_len=input_seq_len,
        vocab_size=vocab_data,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
        power_a=power_a,
        seed=seed,
        fixed_examples=fixed_examples,
        min_query_pos=min_query_pos,
    )

    for split_name, ds in ("train", train_ds), ("val", val_ds), ("test", test_ds):
        try:
            verify_mqar_dataset(
                ds,
                num_kv_pairs=num_kv_pairs,
                num_passes=num_passes,
                min_query_pos=min_query_pos,
            )
        except ValueError as e:
            raise ValueError(f"MQAR oracle failed on split={split_name}") from e

    split_rep = mqar_split_disjoint_report(train_ds, val_ds, test_ds)
    log_mqar_split_report(split_rep, log_line, num_kv_pairs=num_kv_pairs)

    dump_samples = bool(OmegaConf.select(cfg, "data.dump_samples", default=False))
    dump_n_samples = int(OmegaConf.select(cfg, "data.dump_n_samples", default=5))
    if dump_samples:
        dump_mqar_samples(
            train_ds,
            out_dir / "samples_train.txt",
            num_samples=dump_n_samples,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            min_query_pos=min_query_pos,
        )
        dump_mqar_samples(
            val_ds,
            out_dir / "samples_val.txt",
            num_samples=dump_n_samples,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            min_query_pos=min_query_pos,
        )
        dump_mqar_samples(
            test_ds,
            out_dir / "samples_test.txt",
            num_samples=dump_n_samples,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            min_query_pos=min_query_pos,
        )
        log_line(
            f"dumped MQAR samples (n={dump_n_samples}) to "
            f"{out_dir / 'samples_train.txt'}, {out_dir / 'samples_val.txt'}, "
            f"{out_dir / 'samples_test.txt'}"
        )

    pin_default = torch.cuda.is_available()
    pin_memory = bool(OmegaConf.select(cfg, "loader.pin_memory", default=pin_default))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.loader.batch_size),
        shuffle=True,
        num_workers=int(cfg.loader.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.loader.batch_size),
        shuffle=False,
        num_workers=int(cfg.loader.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.loader.batch_size),
        shuffle=False,
        num_workers=int(cfg.loader.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_lm(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr = float(cfg.train.lr)
    wd = float(OmegaConf.select(cfg, "train.weight_decay", default=0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    grad_clip = float(OmegaConf.select(cfg, "train.grad_clip", default=0.0))

    OmegaConf.save(cfg, out_dir / "config_resolved.yaml")

    metrics_path = out_dir / "metrics.csv"
    if metrics_path.is_file():
        metrics_path.unlink()

    epochs = int(cfg.train.epochs)
    best_loss_path = out_dir / "best_loss.pt"
    best_acc_path = out_dir / "best_acc.pt"
    best_loss_running = float("inf")
    best_acc_running = -1.0
    best_val_loss_ever = float("inf")
    best_val_acc_ever = -1.0

    patience = int(OmegaConf.select(cfg, "train.early_stopping.patience", default=3))
    min_delta = float(OmegaConf.select(cfg, "train.early_stopping.min_delta", default=1e-4))
    epochs_no_improve = 0

    log_line(f"experiment={exp_name} device={device} out_dir={out_dir}")
    log_line(
        f"mqar data: fixed_examples={fixed_examples} random_baseline_acc≈1/vocab={random_baseline_acc:.6f} "
        f"(vocab_size={vocab_data})"
    )
    _mqar_log_startup_resolved(
        cfg,
        log_line=log_line,
        vocab_data=vocab_data,
        num_params=num_params,
        tags=tags,
        input_seq_len=input_seq_len,
        num_kv_pairs=num_kv_pairs,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        fixed_examples=fixed_examples,
        min_query_pos=min_query_pos,
    )

    t_loop0 = time.perf_counter()
    last_epoch = 0

    def save_ckpt(path: Path) -> None:
        torch.save({"model_state_dict": model.state_dict(), "cfg": OmegaConf.to_yaml(cfg)}, path)

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = mqar_cross_entropy(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        tr_metrics = mqar_epoch_metrics(model, train_loader, device)
        va_metrics = mqar_epoch_metrics(model, val_loader, device)

        best_val_loss_ever = min(best_val_loss_ever, float(va_metrics["answer_loss"]))
        best_val_acc_ever = max(best_val_acc_ever, float(va_metrics["answer_accuracy"]))

        first_row = epoch == 1
        _append_metrics_csv(metrics_path, {"epoch": epoch, "split": "train", **tr_metrics}, write_header=first_row)
        _append_metrics_csv(metrics_path, {"epoch": epoch, "split": "val", **va_metrics}, write_header=False)
        log_line(
            f"epoch {epoch}/{epochs} train loss={tr_metrics['answer_loss']:.4f} acc={tr_metrics['answer_accuracy']:.4f} em={tr_metrics['exact_match']:.4f} | "
            f"val loss={va_metrics['answer_loss']:.4f} acc={va_metrics['answer_accuracy']:.4f} em={va_metrics['exact_match']:.4f} | "
            f"random≈{random_baseline_acc:.4f}"
        )

        improved_loss = va_metrics["answer_loss"] < best_loss_running - min_delta
        improved_acc = va_metrics["answer_accuracy"] > best_acc_running + min_delta

        if improved_loss:
            best_loss_running = float(va_metrics["answer_loss"])
            save_ckpt(best_loss_path)
            log_line(f"  saved best_loss.pt (val_answer_loss={best_loss_running:.6f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if improved_acc:
            best_acc_running = float(va_metrics["answer_accuracy"])
            save_ckpt(best_acc_path)
            log_line(f"  saved best_acc.pt (val_answer_accuracy={best_acc_running:.6f})")

        if not improved_loss:
            if patience > 0 and epochs_no_improve >= patience:
                log_line(
                    f"early stopping at epoch {epoch}: val answer_loss did not improve by "
                    f"min_delta={min_delta} for {patience} epochs"
                )
                break

    t_loop1 = time.perf_counter()
    wall_loop = t_loop1 - t_loop0

    def _load_and_eval(path: Path) -> dict[str, float]:
        if not path.is_file():
            return {
                "answer_loss": float("nan"),
                "answer_ppl": float("nan"),
                "answer_accuracy": float("nan"),
                "exact_match": float("nan"),
            }
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return mqar_epoch_metrics(model, test_loader, device)

    test_loss_ckpt = _load_and_eval(best_loss_path)
    test_acc_ckpt = _load_and_eval(best_acc_path)
    _write_test_metrics_csv(out_dir / "test_metrics.csv", test_loss_ckpt, test_acc_ckpt)

    def _fmt_metric(x: float) -> str:
        if x != x:  # NaN
            return ""
        return f"{x:.8f}"

    out_rel = str(out_dir.relative_to(PROJECT_ROOT))
    meta_row = {
        "experiment_name": exp_name,
        "output_dir": out_rel,
        "input_seq_len": input_seq_len,
        "num_kv_pairs": num_kv_pairs,
        "vocab_size": vocab_data,
        "min_query_pos": "" if min_query_pos is None else str(int(min_query_pos)),
        "seed": seed,
        "lr": f"{lr:.12g}",
        "backbone": tags["backbone"],
        "fusion": tags["fusion"],
        "stride": tags["stride"],
        "d_model": d_model,
        "num_parameters_trainable": num_params,
        "epochs_planned": epochs,
        "epochs_completed": last_epoch,
        "best_val_answer_loss": f"{best_val_loss_ever:.8f}" if best_val_loss_ever < float("inf") else "",
        "best_val_answer_accuracy": (
            f"{best_val_acc_ever:.8f}" if best_val_acc_ever >= 0.0 else ""
        ),
        "test_answer_loss": _fmt_metric(test_loss_ckpt["answer_loss"]),
        "test_answer_accuracy": _fmt_metric(test_loss_ckpt["answer_accuracy"]),
        "test_exact_match": _fmt_metric(test_loss_ckpt["exact_match"]),
        "test_loss_ckpt_answer_loss": _fmt_metric(test_loss_ckpt["answer_loss"]),
        "test_loss_ckpt_answer_accuracy": _fmt_metric(test_loss_ckpt["answer_accuracy"]),
        "test_loss_ckpt_exact_match": _fmt_metric(test_loss_ckpt["exact_match"]),
        "test_acc_ckpt_answer_loss": _fmt_metric(test_acc_ckpt["answer_loss"]),
        "test_acc_ckpt_answer_accuracy": _fmt_metric(test_acc_ckpt["answer_accuracy"]),
        "test_acc_ckpt_exact_match": _fmt_metric(test_acc_ckpt["exact_match"]),
        "random_baseline_accuracy": f"{random_baseline_acc:.8f}",
        "wall_training_loop_sec": round(wall_loop, 6),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
    }
    _write_meta_metrics_csv(out_dir / "meta_metrics.csv", meta_row)
    log_line(
        f"test [best_loss_ckpt] loss={test_loss_ckpt['answer_loss']:.6f} acc={test_loss_ckpt['answer_accuracy']:.6f} "
        f"em={test_loss_ckpt['exact_match']:.6f}"
    )
    log_line(
        f"test [best_acc_ckpt]  loss={test_acc_ckpt['answer_loss']:.6f} acc={test_acc_ckpt['answer_accuracy']:.6f} "
        f"em={test_acc_ckpt['exact_match']:.6f} random_baseline≈{random_baseline_acc:.6f}"
    )
    log_line(f"done. meta_metrics → {out_dir / 'meta_metrics.csv'}")


if __name__ == "__main__":
    main()
