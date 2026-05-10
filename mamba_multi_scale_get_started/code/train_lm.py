#!/usr/bin/env python3
"""Train character-level LM from YAML + CLI overrides."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# ``code/`` on path for ``datasets``, ``models``, ``exp_config``
_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from datasets.tiny_shakespeare import build_char_lm_dataset
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
    dataset = str(OmegaConf.select(cfg, "data.dataset", default="tiny_shakespeare")).strip().lower()
    return (PROJECT_ROOT / "logs" / dataset / name).resolve()


def loss_to_bpc(mean_ce: float) -> float:
    """Bits per character for char LM: CE (nats) / ln(2)."""
    return float(mean_ce) / math.log(2.0)


def _metrics_row(
    epoch: int,
    split: str,
    loss: float,
) -> dict[str, float | int | str]:
    ppl = math.exp(loss)
    bpc = loss_to_bpc(loss)
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
        "best_val_bpc",
        "final_val_loss",
        "final_val_bpc",
        "num_parameters_trainable",
        "train_max_steps",
        "eval_max_batches",
        "wall_setup_sec",
        "wall_train_phases_sec",
        "wall_eval_phases_sec",
        "wall_training_loop_sec",
        "wall_overhead_sec",
        "finished_at_utc",
        "dataset",
        "data_sampling",
        "debug_initial_train_ce",
        "debug_initial_val_ce",
        "debug_shift_ok",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    assert not model.training
    total_n = 0
    total_loss = 0.0
    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        assert x.shape == y.shape and x.dim() == 2
        logits = model(x)
        assert logits.shape == (x.shape[0], x.shape[1], logits.size(-1))
        loss = lm_token_cross_entropy(logits, y)
        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n
        if max_batches is not None and (bi + 1) >= int(max_batches):
            break
    return total_loss / max(total_n, 1)


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    *,
    max_steps: int | None = None,
    debug_timing: bool = False,
    log_line: Any | None = None,
    global_step_max: int | None = None,
    global_steps_done_before: int = 0,
    log_every_steps: int = 50,
) -> tuple[float, int]:
    model.train()
    assert model.training
    total_n = 0
    total_loss = 0.0
    steps_done = 0
    t_enter = time.perf_counter()
    t_after_fetch = t_enter

    def _dbg(msg: str) -> None:
        if debug_timing and log_line is not None:
            wall = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            log_line(f"[debug_timing] {wall} (+{time.perf_counter() - t_enter:.3f}s) {msg}")

    for x, y in loader:
        if steps_done == 0:
            t_after_fetch = time.perf_counter()
            _dbg(f"train first batch fetched from DataLoader (+{t_after_fetch - t_enter:.3f}s since train_epoch start)")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        assert x.shape == y.shape and x.dim() == 2
        opt.zero_grad(set_to_none=True)
        t_fwd0 = time.perf_counter()
        logits = model(x)
        assert logits.shape == (x.shape[0], x.shape[1], logits.size(-1))
        loss = lm_token_cross_entropy(logits, y)
        if steps_done == 0 and debug_timing:
            _dbg(f"first forward done (+{time.perf_counter() - t_fwd0:.3f}s for forward pass)")
        t_bwd0 = time.perf_counter()
        loss.backward()
        if steps_done == 0 and debug_timing:
            _dbg(f"first backward done (+{time.perf_counter() - t_bwd0:.3f}s for backward)")
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        t_step0 = time.perf_counter()
        opt.step()
        if steps_done == 0 and debug_timing:
            _dbg(f"first optimizer.step done (+{time.perf_counter() - t_step0:.3f}s for optim step)")
        n = y.numel()
        total_loss += float(loss.item()) * n
        total_n += n
        steps_done += 1
        gstep = global_steps_done_before + steps_done
        le = max(1, int(log_every_steps))
        if debug_timing and log_line is not None:
            if global_step_max is not None:
                if gstep == 1 or gstep % le == 0 or gstep >= global_step_max:
                    log_line(f"[step] train {gstep}/{global_step_max}")
            elif steps_done <= 3 or steps_done % max(50, le) == 0:
                log_line(f"[step] train step={steps_done} (within epoch)")

        if max_steps is not None and steps_done >= int(max_steps):
            break

    mean = total_loss / max(total_n, 1)
    return mean, steps_done


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
    embed = getattr(model, "embed", None) or getattr(model, "tok_embed", None)
    if embed is None:
        raise AttributeError(
            "LM diagnostics need model.embed or model.tok_embed (e.g. Mamba vs transformer_lm)."
        )
    emb_dim = int(embed.embedding_dim)
    with torch.no_grad():
        for name, loader in (("train", train_loader), ("val", val_loader)):
            x, y = next(iter(loader))
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            assert x.shape == y.shape and x.dim() == 2
            b, t = x.shape
            emb = embed(x)
            assert emb.shape == (b, t, emb_dim)
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
@click.option(
    "--debug-timing",
    "debug_timing_cli",
    is_flag=True,
    default=False,
    help="Structured phase timings / step logs (same as train.debug_timing=true).",
)
@click.argument("overrides", nargs=-1)
def main(
    config_path: str,
    debug_initial_eval: bool,
    debug_timing_cli: bool,
    overrides: tuple[str, ...],
) -> None:
    t_run0 = time.perf_counter()
    cfg = load_config(config_path, list(overrides))
    exp_name = str(cfg.experiment.name)
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    debug_timing = bool(debug_timing_cli) or bool(OmegaConf.select(cfg, "train.debug_timing", default=False))
    ms_raw = OmegaConf.select(cfg, "train.max_steps", default=None)
    train_max_steps = int(ms_raw) if ms_raw not in (None, "") else None
    eb_raw = OmegaConf.select(cfg, "eval.max_batches", default=None)
    eval_max_batches = int(eb_raw) if eb_raw not in (None, "") else None
    log_every_steps = int(OmegaConf.select(cfg, "train.log_every_steps", default=50))
    pin_mem_raw = OmegaConf.select(cfg, "loader.pin_memory", default=None)
    pin_memory: bool | None
    if pin_mem_raw is None:
        pin_memory = None
    else:
        pin_memory = bool(pin_mem_raw)
    persistent_workers = bool(OmegaConf.select(cfg, "loader.persistent_workers", default=False))
    use_encoded_cache = bool(OmegaConf.select(cfg, "data.use_encoded_cache", default=False))
    save_best = bool(OmegaConf.select(cfg, "train.save_best_checkpoint", default=True))

    device = _resolve_lm_device(cfg)

    data_dir = PROJECT_ROOT / str(cfg.data.data_dir)
    dataset = str(OmegaConf.select(cfg, "data.dataset", default="tiny_shakespeare")).strip().lower()

    log_dir = _resolve_log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_txt = log_dir / "train.log"

    def log_line(msg: str) -> None:
        print(msg)
        with log_txt.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def dbg_phase(msg: str) -> None:
        if debug_timing:
            wall = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            log_line(f"[debug_timing] {wall} (+{time.perf_counter() - t_run0:.3f}s since start) {msg}")

    sampling = str(OmegaConf.select(cfg, "data.sampling", default="sequential")).strip().lower()
    steps_per_epoch = int(OmegaConf.select(cfg, "data.steps_per_epoch", default=1000))
    dbg_phase("resolved config merged (load_config done)")
    t_ds0 = time.perf_counter()
    train_loader, val_loader, vocab = build_char_lm_dataset(
        dataset=dataset,
        data_dir=data_dir,
        block_size=int(cfg.data.block_size),
        batch_size=int(cfg.loader.batch_size),
        train_ratio=float(cfg.data.train_ratio),
        num_workers=int(cfg.loader.num_workers),
        sampling=sampling,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        use_encoded_cache=use_encoded_cache,
        log=log_line if debug_timing else None,
    )
    dbg_phase(f"build dataset + DataLoaders done (+{time.perf_counter() - t_ds0:.3f}s)")
    epochs = int(cfg.train.epochs)

    cfg.model.vocab_size = len(vocab)
    t_md0 = time.perf_counter()
    model = build_lm(cfg).to(device)
    dbg_phase(f"build model done (+{time.perf_counter() - t_md0:.3f}s)")
    t_after_setup = time.perf_counter()
    wall_setup_sec = t_after_setup - t_run0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if debug_timing:
        bs = int(cfg.loader.batch_size)
        nw = int(cfg.loader.num_workers)
        pin_shown = pin_memory if pin_memory is not None else bool(torch.cuda.is_available())
        tr_ds = train_loader.dataset
        val_ds_obj = val_loader.dataset
        tr_tensor = getattr(tr_ds, "data", None)
        val_tensor = getattr(val_ds_obj, "data", None)
        tr_tokens = (
            int(tr_tensor.numel()) if isinstance(tr_tensor, torch.Tensor) else "n/a (random_windows parent)"
        )
        val_tokens = int(val_tensor.numel()) if isinstance(val_tensor, torch.Tensor) else "n/a"
        enc_cache_pt = (Path(data_dir) / f".char_lm_{dataset}_encoded.pt").resolve()

        log_line(
            f"[debug_timing] summary: seq_len/block_size={int(cfg.data.block_size)} batch_size={bs} "
            f"num_workers={nw} pin_memory={pin_shown} "
            f"persistent_workers={bool(persistent_workers and nw > 0)} device={device} "
            f"trainable_params={num_params}"
        )
        log_line(
            f"[debug_timing] train_samples/epoch(len dataset)={len(train_loader.dataset)} "
            f"val_windows={len(val_loader.dataset)} train_batches_epoch={len(train_loader)} "
            f"val_batches_full={len(val_loader)} epochs={epochs} train.max_steps={train_max_steps} "
            f"eval.max_batches={eval_max_batches}"
        )
        log_line(f"[debug_timing] corpus_chars train_split_tokens≈{tr_tokens} val_split_tokens≈{val_tokens}")
        if use_encoded_cache:
            log_line(f"[debug_timing] encoded_cache file (if saved)={enc_cache_pt}")

    lr = float(cfg.train.lr)
    wd = float(OmegaConf.select(cfg, "train.weight_decay", default=0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    grad_clip = float(OmegaConf.select(cfg, "train.grad_clip", default=0.0))

    out_dir = _resolve_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_resolved.yaml")
    (out_dir / "vocab.json").write_text(json.dumps(vocab.to_json(), indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.csv"
    if metrics_path.is_file():
        metrics_path.unlink()

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    patience = int(OmegaConf.select(cfg, "train.early_stopping.patience", default=3))
    min_delta = float(OmegaConf.select(cfg, "train.early_stopping.min_delta", default=1e-4))
    epochs_no_improve = 0

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
    total_train_steps = 0
    final_va_loss = float("nan")

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        if train_max_steps is not None and total_train_steps >= train_max_steps:
            log_line(f"[train] train.max_steps={train_max_steps} already reached ({total_train_steps} steps)")
            break

        tr_ds = train_loader.dataset
        if hasattr(tr_ds, "set_epoch"):
            tr_ds.set_epoch(epoch)
        remaining: int | None = None
        if train_max_steps is not None:
            remaining = int(train_max_steps) - total_train_steps
            if remaining <= 0:
                break

        t_tr0 = time.perf_counter()
        tr_loss, n_steps = _train_epoch(
            model,
            train_loader,
            opt,
            device,
            grad_clip,
            max_steps=remaining,
            debug_timing=debug_timing,
            log_line=log_line,
            global_step_max=train_max_steps,
            global_steps_done_before=total_train_steps,
            log_every_steps=log_every_steps,
        )
        total_train_steps += n_steps
        wall_train_sum += time.perf_counter() - t_tr0

        if device.type == "cuda":
            torch.cuda.synchronize()

        if debug_timing:
            dbg_phase(
                f"epoch {epoch}/{epochs}: train_updates={n_steps} cumulative_steps={total_train_steps} "
                f"mean_train_loss={tr_loss:.6f}"
            )

        if debug_timing:
            dbg_phase(
                f"validation start epoch={epoch} "
                f"eval.max_batches={'full' if eval_max_batches is None else eval_max_batches}"
            )

        t_ev0 = time.perf_counter()
        va_loss = _eval_epoch(model, val_loader, device, max_batches=eval_max_batches)
        wall_eval_sum += time.perf_counter() - t_ev0
        final_va_loss = float(va_loss)

        if debug_timing:
            dbg_phase(f"validation end epoch={epoch} val_mean_ce={va_loss:.6f} val_bpc={loss_to_bpc(va_loss):.6f}")

        first_row = epoch == 1
        for row in (
            _metrics_row(epoch, "train", tr_loss),
            _metrics_row(epoch, "val", va_loss),
        ):
            _append_metrics_csv(metrics_path, row, write_header=first_row and row["split"] == "train")
            first_row = False
        log_line(
            f"epoch {epoch}/{epochs} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"val_ppl={math.exp(va_loss):.2f} val_bpc={loss_to_bpc(va_loss):.6f} "
            f"(train_updates_epoch={n_steps})"
        )

        improved = va_loss < best_val - min_delta
        if improved:
            best_val = va_loss
            epochs_no_improve = 0
            if save_best:
                t_ckpt0 = time.perf_counter()
                torch.save(
                    {"model_state_dict": model.state_dict(), "cfg": OmegaConf.to_yaml(cfg)},
                    best_path,
                )
                if debug_timing:
                    dbg_phase(f"checkpoint best.pt saved (+{time.perf_counter() - t_ckpt0:.3f}s)")
                log_line(f"  saved best.pt (val_loss={best_val:.6f} val_bpc={loss_to_bpc(best_val):.6f})")
            elif debug_timing:
                log_line("[debug_timing] train.save_best_checkpoint=false — skipping best.pt")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                early_stopped = True
                log_line(
                    f"early stopping at epoch {epoch}: val_loss did not improve by "
                    f"min_delta={min_delta} for {patience} epochs (best val_loss={best_val:.6f})"
                )
                break

        if train_max_steps is not None and total_train_steps >= train_max_steps:
            log_line(
                f"[train] stopping after train.max_steps={train_max_steps} "
                f"(performed {total_train_steps} optimizer steps)"
            )
            break

    t_loop1 = time.perf_counter()
    wall_loop_sec = t_loop1 - t_loop0
    wall_overhead = wall_loop_sec - wall_train_sum - wall_eval_sum
    best_str = f"{best_val:.8f}" if best_val < float("inf") else ""
    best_bpc_str = f"{loss_to_bpc(best_val):.8f}" if best_val < float("inf") else ""
    final_loss_str = f"{final_va_loss:.8f}" if math.isfinite(final_va_loss) else ""
    final_bpc_str = f"{loss_to_bpc(final_va_loss):.8f}" if math.isfinite(final_va_loss) else ""
    mx_str = str(train_max_steps) if train_max_steps is not None else ""
    emx_str = str(eval_max_batches) if eval_max_batches is not None else ""

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
            "best_val_bpc": best_bpc_str,
            "final_val_loss": final_loss_str,
            "final_val_bpc": final_bpc_str,
            "num_parameters_trainable": num_params,
            "train_max_steps": mx_str,
            "eval_max_batches": emx_str,
            "wall_setup_sec": round(wall_setup_sec, 6),
            "wall_train_phases_sec": round(wall_train_sum, 6),
            "wall_eval_phases_sec": round(wall_eval_sum, 6),
            "wall_training_loop_sec": round(wall_loop_sec, 6),
            "wall_overhead_sec": round(wall_overhead, 6),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "data_sampling": sampling,
            "debug_initial_train_ce": dbg_train_ce,
            "debug_initial_val_ce": dbg_val_ce,
            "debug_shift_ok": dbg_shift,
            "status": "ok",
        },
    )
    log_line(
        f"meta_metrics: train={wall_train_sum:.2f}s eval={wall_eval_sum:.2f}s "
        f"loop={wall_loop_sec:.2f}s setup={wall_setup_sec:.2f}s → {meta_path.name}"
    )


if __name__ == "__main__":
    main()
