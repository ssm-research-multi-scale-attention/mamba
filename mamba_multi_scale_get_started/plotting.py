"""Training / evaluation figures."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def save_train_loss_plot(output_dir: str, train_losses: list[float], filename: str = "train_loss.png") -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def save_confusion_matrix_plot(
    output_dir: str,
    cm: np.ndarray,
    filename: str = "confusion_matrix.png",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
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
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def save_training_artifacts(
    output_dir: str,
    train_losses: list[float],
    confusion_matrix: np.ndarray | None,
) -> None:
    save_train_loss_plot(output_dir, train_losses)
    if confusion_matrix is not None:
        save_confusion_matrix_plot(output_dir, confusion_matrix)
