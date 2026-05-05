"""Character-level corpora (Tiny Shakespeare, text8): download, vocab, DataLoaders."""
from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TEXT8_ZIP_URL = "http://mattmahoney.net/dc/text8.zip"


class CharVocab:
    def __init__(self, stoi: dict[str, int], itos: list[str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = list(chars)
        return cls(stoi, itos)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_json(self) -> dict[str, Any]:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CharVocab:
        return cls(stoi=dict(d["stoi"]), itos=list(d["itos"]))


class CharLMDataset(Dataset):
    """Fixed-length windows: ``x = data[idx:idx+T]``, ``y = data[idx+1:idx+T+1]``."""

    def __init__(self, data: torch.Tensor, block_size: int):
        if data.numel() <= block_size:
            raise ValueError(
                f"Sequence length {data.numel()} must exceed block_size={block_size} for LM windows."
            )
        self.data = data
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return self.data.numel() - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = slice(idx, idx + self.block_size)
        x = self.data[sl].long()
        y = self.data[idx + 1 : idx + self.block_size + 1].long()
        return x, y


class CharLMRandomWindowDataset(Dataset):
    """
    ``steps_per_epoch`` windows with random starts (re-seeded each ``set_epoch``).

    Use ``shuffle=False`` in DataLoader; call ``set_epoch(epoch)`` once per training epoch
    so windows change while staying deterministic for a given (epoch, idx, worker).
    """

    def __init__(self, data: torch.Tensor, block_size: int, steps_per_epoch: int, seed: int):
        if data.numel() <= int(block_size):
            raise ValueError(
                f"Sequence length {data.numel()} must exceed block_size={block_size} for LM windows."
            )
        self.data = data
        self.block_size = int(block_size)
        self.length = max(1, int(steps_per_epoch))
        self.seed = int(seed)
        self.max_start = int(data.numel()) - self.block_size - 1
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        salt = (self._epoch * 1_000_003 + wid * 10_007 + int(idx) * 300_011 + self.seed) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(salt)
        start = rng.randint(0, self.max_start)
        sl = slice(start, start + self.block_size)
        x = self.data[sl].long()
        y = self.data[start + 1 : start + self.block_size + 1].long()
        return x, y


def _download_if_missing(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 — fixed benchmark URL
        dest.write_bytes(r.read())


def _ensure_tiny_shakespeare_text(data_dir: Path) -> str:
    txt_path = data_dir / "input.txt"
    _download_if_missing(TINY_SHAKESPEARE_URL, txt_path)
    return txt_path.read_text(encoding="utf-8")


def _ensure_text8_text(data_dir: Path) -> str:
    txt_path = data_dir / "text8"
    if txt_path.is_file() and txt_path.stat().st_size > 0:
        return txt_path.read_text(encoding="utf-8")
    zip_path = data_dir / "text8.zip"
    _download_if_missing(TEXT8_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = set(zf.namelist())
        if "text8" not in members:
            raise ValueError("text8.zip has no 'text8' file.")
        zf.extract("text8", path=data_dir)
    return txt_path.read_text(encoding="utf-8")


def build_char_lm_dataset(
    dataset: str,
    data_dir: str | Path,
    block_size: int,
    batch_size: int,
    train_ratio: float,
    num_workers: int,
    *,
    vocab: CharVocab | None = None,
    sampling: str = "sequential",
    steps_per_epoch: int = 1000,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, CharVocab]:
    """Build char LM loaders for supported corpora."""
    data_dir = Path(data_dir)
    ds = str(dataset).strip().lower()
    if ds == "tiny_shakespeare":
        text = _ensure_tiny_shakespeare_text(data_dir)
    elif ds == "text8":
        text = _ensure_text8_text(data_dir)
    else:
        raise ValueError(f"Unsupported data.dataset={dataset!r}. Use tiny_shakespeare or text8.")

    if vocab is None:
        vocab = CharVocab.from_text(text)
    try:
        ids = torch.tensor(vocab.encode(text), dtype=torch.long)
    except KeyError as e:
        ch = str(e).strip("'")
        raise KeyError(
            f"Dataset {dataset!r} contains character {ch!r} missing from provided vocab."
        ) from e

    n = ids.numel()
    need = int(block_size) + 1
    if n <= 2 * need:
        raise ValueError(f"Corpus length {n} too short for block_size={block_size} (need > {2 * need}).")
    n_train = int(float(train_ratio) * n)
    n_train = max(need, min(n_train, n - need))
    train_ids = ids[:n_train].contiguous()
    val_ids = ids[n_train:].contiguous()

    samp = str(sampling).strip().lower()
    if samp == "random_windows":
        train_ds: Dataset = CharLMRandomWindowDataset(train_ids, block_size, steps_per_epoch, seed)
        train_shuffle = False
    elif samp == "sequential":
        train_ds = CharLMDataset(train_ids, block_size)
        train_shuffle = True
    else:
        raise ValueError(f"data.sampling must be sequential or random_windows, got {sampling!r}")

    val_ds = CharLMDataset(val_ids, block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, vocab


def build_tiny_shakespeare(
    data_dir: str | Path,
    block_size: int,
    batch_size: int,
    train_ratio: float,
    num_workers: int,
    *,
    sampling: str = "sequential",
    steps_per_epoch: int = 1000,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, CharVocab]:
    """Backward-compatible alias for Tiny Shakespeare char LM loaders."""
    return build_char_lm_dataset(
        dataset="tiny_shakespeare",
        data_dir=data_dir,
        block_size=block_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
        num_workers=num_workers,
        sampling=sampling,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )
