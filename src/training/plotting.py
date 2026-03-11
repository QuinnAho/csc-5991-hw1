"""Plotting helpers for loss curves and report-ready figures.

The homework report needs clear training and validation plots. This module
keeps that logic separate from training so figures are easy to regenerate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _prepare_figure_path(output_path: str | Path) -> Path:
    """Create the parent directory for a figure path and return the Path object."""

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    return figure_path


def plot_training_loss(history: dict[str, Any], output_path: str | Path) -> None:
    """Create a plot with only training loss over epochs."""

    train_loss = history.get("train_loss", [])
    epochs = range(1, len(train_loss) + 1)
    figure_path = _prepare_figure_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", color="#1f77b4")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_validation_loss(history: dict[str, Any], output_path: str | Path) -> None:
    """Create a plot with only validation loss over epochs."""

    val_loss = history.get("val_loss", [])
    epochs = range(1, len(val_loss) + 1)
    figure_path = _prepare_figure_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_loss, label="Validation Loss", marker="s", color="#d62728")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_loss_curves(history: dict[str, Any], output_path: str | Path) -> None:
    """Create a training-versus-validation loss plot and save it to disk."""

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    figure_path = _prepare_figure_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_evaluation_loss_summary(
    history: dict[str, Any],
    test_metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Create a compact bar chart comparing best validation loss and test loss."""

    val_losses = history.get("val_loss", [])
    best_val_loss = min(val_losses) if val_losses else 0.0
    test_loss = float(test_metrics.get("loss", 0.0))
    figure_path = _prepare_figure_path(output_path)

    plt.figure(figsize=(6, 5))
    plt.bar(["Best Validation Loss", "Test Loss"], [best_val_loss, test_loss], color=["#ff7f0e", "#2ca02c"])
    plt.ylabel("Loss")
    plt.title("Validation vs Test Loss")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
