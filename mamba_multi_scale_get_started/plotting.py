"""Interactive training / evaluation figures (Plotly → HTML for IDE or browser)."""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _write_html(fig: go.Figure, path: str, *, include_plotlyjs: str | bool) -> None:
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True, config={"displayModeBar": True})


def save_train_loss_html(
    output_dir: str,
    train_losses: list[float],
    *,
    include_plotlyjs: str | bool = "cdn",
    filename: str = "train_loss.html",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode="lines+markers",
                name="train_loss",
                hovertemplate="epoch=%{x}<br>loss=%{y:.6f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Train loss (per epoch)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        hovermode="x unified",
        height=420,
    )
    _write_html(fig, path, include_plotlyjs=include_plotlyjs)
    print(f"Saved: {path}")


def save_confusion_matrix_html(
    output_dir: str,
    cm: np.ndarray,
    *,
    include_plotlyjs: str | bool = "cdn",
    filename: str = "confusion_matrix.html",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    z_text = [[str(int(cm[i, j])) for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=list(range(cm.shape[1])),
            y=list(range(cm.shape[0])),
            text=z_text,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Test confusion matrix",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template="plotly_white",
        yaxis_autorange="reversed",
        width=520,
        height=480,
    )
    _write_html(fig, path, include_plotlyjs=include_plotlyjs)
    print(f"Saved: {path}")


def save_interactive_report_html(
    output_dir: str,
    train_losses: list[float],
    cm: np.ndarray | None,
    *,
    include_plotlyjs: str | bool = "cdn",
    filename: str = "interactive_report.html",
) -> None:
    """Single scrollable HTML: loss curve plus confusion matrix (if available)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    epochs = list(range(1, len(train_losses) + 1))

    if cm is not None:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.34, 0.66],
            vertical_spacing=0.14,
            subplot_titles=("Train loss (per epoch)", "Test confusion matrix"),
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode="lines+markers",
                name="train_loss",
                hovertemplate="epoch=%{x}<br>loss=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        z_text = [[str(int(cm[i, j])) for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=list(range(cm.shape[1])),
                y=list(range(cm.shape[0])),
                text=z_text,
                texttemplate="%{text}",
                colorscale="Blues",
                hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text="Predicted label", row=2, col=1)
        fig.update_yaxes(title_text="True label", autorange="reversed", row=2, col=1)
        fig.update_layout(height=900, template="plotly_white", title_text="Training run (interactive)")
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=epochs,
                    y=train_losses,
                    mode="lines+markers",
                    name="train_loss",
                    hovertemplate="epoch=%{x}<br>loss=%{y:.6f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Train loss (per epoch)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            height=480,
        )

    _write_html(fig, path, include_plotlyjs=include_plotlyjs)
    print(f"Saved: {path}")


def save_training_artifacts(
    output_dir: str,
    train_losses: list[float],
    confusion_matrix: np.ndarray | None,
    *,
    include_plotlyjs: str | bool = "cdn",
) -> None:
    """Write Plotly HTML files under ``output_dir`` (open locally or via port-forward)."""
    save_train_loss_html(output_dir, train_losses, include_plotlyjs=include_plotlyjs)
    if confusion_matrix is not None:
        save_confusion_matrix_html(output_dir, confusion_matrix, include_plotlyjs=include_plotlyjs)
    save_interactive_report_html(output_dir, train_losses, confusion_matrix, include_plotlyjs=include_plotlyjs)
