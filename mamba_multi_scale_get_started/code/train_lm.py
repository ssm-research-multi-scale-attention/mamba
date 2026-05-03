#!/usr/bin/env python3
"""Train character-level LM (Tiny Shakespeare) from YAML + CLI overrides."""
from __future__ import annotations

import csv
import json
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

# ``code/`` on path for ``datasets``, ``models``, ``exp_config``
_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.tiny_shakespeare import build_tiny_shakespeare
from exp_config import load_config, set_seed
from models.language_models import build_lm

PROJECT_ROOT = _CODE.parent


def lm_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean CE over all positions (matches PyTorch default reduction for LM)."""
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


def _resolve_lm_device(cfg: DictConfig) -> torch.device:
    """
    Pick training device from config (same spirit as IMDB ``device``).

    - ``device: auto`` — CUDA if available, else CPU. On CUDA uses physical GPU index ``cuda_device`` (default 0).
    - ``device: cpu`` — CPU.
    - ``device: cuda`` / ``gpu`` — CUDA required; index from ``cuda_device`` (default 0).
    - ``device: cuda:2`` — explicit GPU index (``cuda_device`` ignored).

    CLI override: ``device=cuda:1`` or ``cuda_device=2``.
    """
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
    return (PROJECT_ROOT / "logs" / "TinyShakespeare" / name).resolve()


def _metrics_row(
    epoch: int,
    split: str,
    loss: float,
) -> dict[str, float | int | str]:
    ppl = math.exp(loss)
    bpc = loss / math.log(2.0)
    return {
        "epoch": epoch,
        "split": split,
        "loss": loss,
        "perplexity": ppl,
        "bpc": bpc,
    }


def _append_metrics_csv(path: Path, row: dict[str, float | int | str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "split", "loss", "perplexity", "bpc"]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row[k] for k in fieldnames})


def _write_meta_metrics_csv(path: Path, row: dict[str, str | int | float | bool]) -> None:
    """One row per run: wall times, best loss, epoch counts, device."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "seed",
        "device",
        "cuda_device_name",
        "epochs_planned",
        "epochs_completed",
        "early_stopped",
        "best_val_loss",
        "num_parameters_trainable",
        "wall_setup_sec",
        "wall_train_phases_sec",
        "wall_eval_phases_sec",
        "wall_training_loop_sec",
        "wall_overhead_sec",
        "finished_at_utc",
        "data_sampling",
        "debug_initial_train_ce",
        "debug_initial_val_ce",
        "debug_shift_ok",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    assert not model.training
    total_n = 0
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        assert x.shape == y.shape and x.dim() == 2
        logits = model(x)
        assert logits.shape == (x.shape[0], x.shape[1], logits.size(-1))
        loss = lm_token_cross_entropy(logits, y)
        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    assert model.training
    total_n = 0
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        assert x.shape == y.shape and x.dim() == 2
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        assert logits.shape == (x.shape[0], x.shape[1], logits.size(-1))
        loss = lm_token_cross_entropy(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


def _run_lm_initial_diagnostics(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    vocab,
    vocab_size: int,
    device: torch.device,
    log_line,
) -> tuple[str, str, str]:
    """
    Random-init loss on one train/val batch, shift check on dataset[0], embedding dtype/shape.

    Returns ``(initial_train_ce_str, initial_val_ce_str, shift_ok_str)`` for meta_metrics.csv.
    """
    voc = int(vocab_size)
    expected = math.log(voc)
    log_line(f"[debug] vocab_size={voc} expected_random_CE={expected:.6f} (=ln(vocab))")

    ds = train_loader.dataset
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0)
    x0, y0 = ds[0]
    n_show = min(80, int(x0.numel()))
    log_line(f"[debug] sample x[:{n_show}] decode: {repr(vocab.decode(x0[:n_show].tolist()))}")
    log_line(f"[debug] sample y[:{n_show}] decode: {repr(vocab.decode(y0[:n_show].tolist()))}")

    shift_ok = "false"
    try:
        if torch.equal(y0[:-1], x0[1:]):
            shift_ok = "true"
            log_line("[debug] shift check OK: y[:-1] == x[1:]")
        else:
            log_line("[debug] ERROR: y[:-1] != x[1:] (shift mismatch)")
    except Exception as e:
        log_line(f"[debug] shift check exception: {e}")

    init_train = ""
    init_val = ""
    model.eval()
    assert not model.training
    with torch.no_grad():
        for name, loader in (("train", train_loader), ("val", val_loader)):
            x, y = next(iter(loader))
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            assert x.shape == y.shape and x.dim() == 2
            b, t = x.shape
            emb = model.embed(x)
            assert emb.shape == (b, t, model.embed.embedding_dim)
            assert emb.dtype in (torch.float32, torch.float16, torch.bfloat16)
            logits = model(x)
            assert logits.shape == (b, t, voc)
            loss = lm_token_cross_entropy(logits, y)
            ce_s = f"{float(loss.item()):.6f}"
            log_line(f"[debug] initial {name} CE={ce_s} (random weights)")
            if name == "train":
                init_train = ce_s
            else:
                init_val = ce_s
    model.train()
    assert model.training
    return init_train, init_val, shift_ok


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Path to YAML config (e.g. configs/TinyShakespeare/tiny_shakespeare_mamba2.yaml).",
)
@click.option(
    "--debug-initial-eval",
    "debug_initial_eval",
    is_flag=True,
    default=False,
    help="Before training: one-batch CE on random weights, shift check, shapes (also train.debug_initial_eval=true).",
)
@click.argument("overrides", nargs=-1)
def main(config_path: str, debug_initial_eval: bool, overrides: tuple[str, ...]) -> None:
    t_run0 = time.perf_counter()
    cfg = load_config(config_path, list(overrides))
    exp_name = str(cfg.experiment.name)
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    device = _resolve_lm_device(cfg)

    data_dir = PROJECT_ROOT / str(cfg.data.data_dir)
    sampling = str(OmegaConf.select(cfg, "data.sampling", default="sequential")).strip().lower()
    steps_per_epoch = int(OmegaConf.select(cfg, "data.steps_per_epoch", default=1000))
    train_loader, val_loader, vocab = build_tiny_shakespeare(
        data_dir=data_dir,
        block_size=int(cfg.data.block_size),
        batch_size=int(cfg.loader.batch_size),
        train_ratio=float(cfg.data.train_ratio),
        num_workers=int(cfg.loader.num_workers),
        sampling=sampling,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )

    cfg.model.vocab_size = len(vocab)
    model = build_lm(cfg).to(device)
    t_after_setup = time.perf_counter()
    wall_setup_sec = t_after_setup - t_run0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lr = float(cfg.train.lr)
    wd = float(OmegaConf.select(cfg, "train.weight_decay", default=0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    grad_clip = float(OmegaConf.select(cfg, "train.grad_clip", default=0.0))

    out_dir = _resolve_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = _resolve_log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, out_dir / "config_resolved.yaml")
    (out_dir / "vocab.json").write_text(json.dumps(vocab.to_json(), indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.csv"
    if metrics_path.is_file():
        metrics_path.unlink()

    epochs = int(cfg.train.epochs)
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    log_txt = log_dir / "train.log"

    patience = int(OmegaConf.select(cfg, "train.early_stopping.patience", default=3))
    min_delta = float(OmegaConf.select(cfg, "train.early_stopping.min_delta", default=1e-4))
    epochs_no_improve = 0

    def log_line(msg: str) -> None:
        print(msg)
        with log_txt.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_line(f"experiment={exp_name} device={device} out_dir={out_dir}")
    if patience > 0:
        log_line(
            f"early_stopping: patience={patience} min_delta={min_delta} "
            f"(disable with train.early_stopping.patience=0)"
        )

    dbg_train_ce = ""
    dbg_val_ce = ""
    dbg_shift = ""
    do_debug = debug_initial_eval or bool(OmegaConf.select(cfg, "train.debug_initial_eval", default=False))
    if do_debug:
        dbg_train_ce, dbg_val_ce, dbg_shift = _run_lm_initial_diagnostics(
            model, train_loader, val_loader, vocab, len(vocab), device, log_line
        )
    model.train()

    cuda_name = ""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            cuda_name = str(torch.cuda.get_device_name(device.index or 0))
        except Exception:
            cuda_name = ""

    t_loop0 = time.perf_counter()
    wall_train_sum = 0.0
    wall_eval_sum = 0.0
    last_epoch = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        tr_ds = train_loader.dataset
        if hasattr(tr_ds, "set_epoch"):
            tr_ds.set_epoch(epoch)
        t_tr0 = time.perf_counter()
        tr_loss = _train_epoch(model, train_loader, opt, device, grad_clip)
        wall_train_sum += time.perf_counter() - t_tr0
        t_ev0 = time.perf_counter()
        va_loss = _eval_epoch(model, val_loader, device)
        wall_eval_sum += time.perf_counter() - t_ev0
        first_row = epoch == 1
        for row in (
            _metrics_row(epoch, "train", tr_loss),
            _metrics_row(epoch, "val", va_loss),
        ):
            _append_metrics_csv(metrics_path, row, write_header=first_row and row["split"] == "train")
            first_row = False
        log_line(
            f"epoch {epoch}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_ppl={math.exp(va_loss):.2f} val_bpc={va_loss / math.log(2):.4f}"
        )

        improved = va_loss < best_val - min_delta
        if improved:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "cfg": OmegaConf.to_yaml(cfg)},
                best_path,
            )
            log_line(f"  saved best.pt (val_loss={best_val:.6f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                early_stopped = True
                log_line(
                    f"early stopping at epoch {epoch}: val_loss did not improve by "
                    f"min_delta={min_delta} for {patience} epochs (best val_loss={best_val:.6f})"
                )
                break

    t_loop1 = time.perf_counter()
    wall_loop_sec = t_loop1 - t_loop0
    wall_overhead = wall_loop_sec - wall_train_sum - wall_eval_sum
    best_str = f"{best_val:.8f}" if best_val < float("inf") else ""

    meta_path = out_dir / "meta_metrics.csv"
    _write_meta_metrics_csv(
        meta_path,
        {
            "experiment_name": exp_name,
            "seed": seed,
            "device": str(device),
            "cuda_device_name": cuda_name,
            "epochs_planned": epochs,
            "epochs_completed": last_epoch,
            "early_stopped": early_stopped,
            "best_val_loss": best_str,
            "num_parameters_trainable": num_params,
            "wall_setup_sec": round(wall_setup_sec, 6),
            "wall_train_phases_sec": round(wall_train_sum, 6),
            "wall_eval_phases_sec": round(wall_eval_sum, 6),
            "wall_training_loop_sec": round(wall_loop_sec, 6),
            "wall_overhead_sec": round(wall_overhead, 6),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_sampling": sampling,
            "debug_initial_train_ce": dbg_train_ce,
            "debug_initial_val_ce": dbg_val_ce,
            "debug_shift_ok": dbg_shift,
        },
    )
    log_line(
        f"meta_metrics: train={wall_train_sum:.2f}s eval={wall_eval_sum:.2f}s "
        f"loop={wall_loop_sec:.2f}s setup={wall_setup_sec:.2f}s → {meta_path.name}"
    )


if __name__ == "__main__":
    main()
