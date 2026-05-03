"""HuggingFace Dataset text classification loaders (config-driven splits/fields)."""
from __future__ import annotations

import torch
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def _split(ds: DatasetDict, name: str):
    try:
        return ds[name]
    except KeyError as e:
        available = sorted(ds.keys())
        raise KeyError(f"Split '{name}' not found. Available: {available}") from e


def collate_text_classification(batch, tokenizer, *, text_key: str, label_key: str, max_length: int):
    toks = tokenizer(
        [b[text_key] for b in batch],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([b[label_key] for b in batch], dtype=torch.long)
    return toks["input_ids"], labels


def build_text_classification_dataloaders(
    cfg: DictConfig,
    tokenizer,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset(cfg.data.dataset)
    seed = int(cfg.experiment.seed)

    train_name = str(cfg.data.train_split)
    eval_name = str(cfg.data.eval_split)
    text_key = str(cfg.data.text_key)
    label_key = str(cfg.data.label_key)

    if cfg.data.train_subset is not None:
        n = int(cfg.data.train_subset)
        split = _split(ds, train_name).shuffle(seed=seed).select(range(n))
        ds[train_name] = split
    if cfg.data.eval_subset is not None:
        n = int(cfg.data.eval_subset)
        split = _split(ds, eval_name).shuffle(seed=seed).select(range(n))
        ds[eval_name] = split

    max_len = int(cfg.data.max_length)
    bs = int(cfg.loader.batch_size)
    pin_memory = bool(cfg.loader.pin_memory) and device.type == "cuda"
    num_workers = int(cfg.loader.num_workers)

    def collate(b):
        return collate_text_classification(
            b,
            tokenizer,
            text_key=text_key,
            label_key=label_key,
            max_length=max_len,
        )

    train_loader = DataLoader(
        _split(ds, train_name),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        _split(ds, eval_name),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader
