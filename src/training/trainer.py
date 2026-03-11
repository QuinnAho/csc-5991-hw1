"""Training loop orchestration for the seq2seq homework project.

This module manages:
- epoch loops
- teacher forcing during training
- validation after each epoch
- metric logging
- best-checkpoint saving
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.config import ProjectConfig
from src.data.vocab import Vocabulary
from src.training.evaluate import evaluate_model


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve a CLI or config device string into a PyTorch device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_loss_function(pad_index: int) -> nn.CrossEntropyLoss:
    """Create the token-level loss used for seq2seq training and evaluation."""

    return nn.CrossEntropyLoss(ignore_index=pad_index, reduction="sum")


def _config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    """Convert the config dataclass tree into a JSON-safe dictionary."""

    return json.loads(json.dumps(asdict(config), default=str))


@dataclass
class TrainingHistory:
    """Container for per-epoch training and validation statistics."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_token_accuracy: list[float] = field(default_factory=list)
    val_exact_match_accuracy: list[float] = field(default_factory=list)
    val_slot_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the training history."""

        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_token_accuracy": self.val_token_accuracy,
            "val_exact_match_accuracy": self.val_exact_match_accuracy,
            "val_slot_accuracy": self.val_slot_accuracy,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def save(self, path: str | Path) -> None:
        """Write the training history to disk as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: ProjectConfig,
    history: TrainingHistory,
) -> None:
    """Save model, optimizer, and run metadata to a checkpoint file."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": _config_to_dict(config),
            "history": history.to_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a saved checkpoint into a model and optionally an optimizer."""

    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_run_config(config: ProjectConfig, path: str | Path) -> None:
    """Write the resolved run configuration to disk as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_config_to_dict(config), indent=2),
        encoding="utf-8",
    )


class Trainer:
    """Small trainer wrapper to keep scripts and report references clean."""

    def __init__(
        self,
        config: ProjectConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        target_vocab: Vocabulary,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.target_vocab = target_vocab
        self.device = device
        self.model.to(device)

    def _train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Run one training epoch and return average token-level loss."""

        self.model.train()
        total_loss = 0.0
        total_target_tokens = 0

        for batch in dataloader:
            source_tokens = batch["source_tokens"].to(self.device)
            source_lengths = batch["source_lengths"].to(self.device)
            target_tokens = batch["target_tokens"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(
                source_tokens=source_tokens,
                source_lengths=source_lengths,
                target_tokens=target_tokens,
                teacher_forcing_ratio=self.config.training.teacher_forcing_ratio,
            )
            target_output = target_tokens[:, 1:]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_output.reshape(-1),
            )
            loss.backward()

            if self.config.training.gradient_clip and self.config.training.gradient_clip > 0.0:
                clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)

            self.optimizer.step()

            total_loss += float(loss.item())
            total_target_tokens += int(target_output.ne(self.target_vocab.pad_index).sum().item())

        return total_loss / max(total_target_tokens, 1)

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> TrainingHistory:
        """Run the full training workflow and save the best checkpoint."""

        history = TrainingHistory()
        epochs_without_improvement = 0

        for epoch in range(1, self.config.training.num_epochs + 1):
            train_loss = self._train_epoch(train_dataloader)
            val_metrics, _ = evaluate_model(
                model=self.model,
                dataloader=val_dataloader,
                criterion=self.criterion,
                target_vocab=self.target_vocab,
                device=self.device,
                max_decode_steps=self.config.data.target_max_length,
            )

            history.train_loss.append(train_loss)
            history.val_loss.append(val_metrics.loss)
            history.val_token_accuracy.append(val_metrics.token_accuracy)
            history.val_exact_match_accuracy.append(val_metrics.exact_match_accuracy)
            history.val_slot_accuracy.append(val_metrics.slot_accuracy)

            improved = val_metrics.loss < history.best_val_loss
            if improved:
                history.best_val_loss = val_metrics.loss
                history.best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    path=self.config.paths.best_checkpoint_file,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_loss=history.best_val_loss,
                    config=self.config,
                    history=history,
                )
            else:
                epochs_without_improvement += 1

            history.save(self.config.paths.train_history_file)
            print(
                f"Epoch {epoch:02d}/{self.config.training.num_epochs:02d} "
                f"- train_loss: {train_loss:.4f} "
                f"- val_loss: {val_metrics.loss:.4f} "
                f"- val_token_acc: {val_metrics.token_accuracy:.4f} "
                f"- val_exact_acc: {val_metrics.exact_match_accuracy:.4f}"
            )

            patience = self.config.training.early_stopping_patience
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

        return history
