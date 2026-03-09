#!/usr/bin/env python3
"""
train_mamba_imdb.py

Example training script: builds an nn.Sequential Mamba2 model with per-layer head control,
trains on IMDB (binary sentiment), prints params and accuracy/precision/recall, and plots
loss + confusion matrix.

Replace the Mamba2 import with the real path where Mamba2Simple is available.
"""
import argparse
import math
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- IMPORTANT: replace this import with the actual module exposing Mamba2Simple ---
# Example: from mamba_module import Mamba2Simple as Mamba2
from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2
# -------------------------------------------------------------------------------

# Simple wrapper to allow varying headdim per layer and to keep model sequential.
class MambaLayer(nn.Module):
    def __init__(self, d_model: int, headdim: int, **mamba_kwargs):
        super().__init__()
        # Use residual connection around Mamba2 block
        self.block = Mamba2(d_model=d_model, headdim=headdim, **mamba_kwargs)

    def forward(self, x):
        return x + self.block(x)


def build_model(d_model: int, layer_headdims: List[int], vocab_size: int, num_classes: int = 2, **mamba_kwargs):
    # Token embedding -> Mamba layers -> pool -> classifier
    embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
    mamba_layers = [MambaLayer(d_model=d_model, headdim=h, **mamba_kwargs) for h in layer_headdims]
    seq = nn.Sequential(*mamba_layers)
    classifier = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, num_classes)
    )
    model = nn.Module()
    model.embed = embed
    model.seq = seq
    model.classifier = classifier
    # forward method as nested function
    def forward_fn(x):
        # x: (B, L) token ids
        emb = model.embed(x)  # (B, L, D)
        out = model.seq(emb)  # (B, L, D)
        pooled = out.mean(dim=1)  # simple mean pool
        logits = model.classifier(pooled)
        return logits
    model.forward = forward_fn
    return model


def collate_fn(batch, tokenizer, max_length):
    toks = tokenizer([b["text"] for b in batch], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return toks["input_ids"], labels


def train_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--layers", type=str, default="32,32,32,32,32,32,32,32")  # comma-separated head dims
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    layer_headdims = [int(x) for x in args.layers.split(",") if x]
    device = torch.device(args.device)

    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = load_dataset("imdb")
    # small quick subset for example if you want; comment out to use full
    # ds["train"] = ds["train"].shuffle(seed=42).select(range(2000))
    # ds["test"] = ds["test"].shuffle(seed=42).select(range(1000))

    train_loader = DataLoader(ds["train"], batch_size=args.bs, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))
    test_loader = DataLoader(ds["test"], batch_size=args.bs, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))

    print("Building model...")
    model = build_model(d_model=args.d_model, layer_headdims=layer_headdims,
                        vocab_size=tokenizer.vocab_size, num_classes=2,
                        # pass any extra Mamba kwargs you like; keep defaults small for speed
                        d_state=32, d_conv=4, expand=2, ngroups=1)
    model.to(device)

    # parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": []}
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, device, criterion)
        history["train_loss"].append(loss)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {loss:.4f}")

    print("Evaluating on test set...")
    preds, trues = eval_model(model, test_loader, device)
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, digits=4)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    # confusion matrix
    cm = confusion_matrix(trues, preds)

    # Plots
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/train_loss.png")
    print("Saved: outputs/train_loss.png")

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Saved: outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
