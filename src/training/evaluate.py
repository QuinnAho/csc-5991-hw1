"""Evaluation helpers for validation and final test reporting.

This module will hold code for computing loss and accuracy metrics, exporting
representative predictions, and optionally generating attention visualizations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from src.data.vocab import Vocabulary
from src.training.metrics import (
    EvaluationMetrics,
    compute_exact_match_accuracy,
    compute_slot_accuracy,
    compute_token_accuracy,
)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a collated batch to the target device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_prediction_record(
    source_text: str,
    predicted_tokens: list[str],
    target_tokens: list[str],
) -> dict[str, object]:
    """Create one report-friendly prediction comparison record."""

    predicted_text = " ".join(predicted_tokens)
    target_text = " ".join(target_tokens)
    exact_match = int(predicted_tokens == target_tokens)

    max_length = max(len(predicted_tokens), len(target_tokens), 1)
    token_matches = sum(
        int(
            (predicted_tokens[index] if index < len(predicted_tokens) else None)
            == (target_tokens[index] if index < len(target_tokens) else None)
        )
        for index in range(max_length)
    )

    return {
        "source_text": source_text,
        "predicted_text": predicted_text,
        "target_text": target_text,
        "exact_match": exact_match,
        "token_match_fraction": token_matches / max_length,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    target_vocab: Vocabulary,
    device: torch.device,
    max_decode_steps: int,
) -> tuple[EvaluationMetrics, list[dict[str, object]]]:
    """Evaluate a model and return summary metrics plus prediction examples."""

    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    predicted_sequences: list[list[str]] = []
    target_sequences: list[list[str]] = []
    prediction_records: list[dict[str, object]] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        logits = model(
            source_tokens=batch["source_tokens"],
            source_lengths=batch["source_lengths"],
            target_tokens=batch["target_tokens"],
            teacher_forcing_ratio=0.0,
        )
        target_output = batch["target_tokens"][:, 1:]
        batch_loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_output.reshape(-1),
        )
        total_loss += float(batch_loss.item())
        total_target_tokens += int(target_output.ne(target_vocab.pad_index).sum().item())

        predicted_ids = model.greedy_decode(
            source_tokens=batch["source_tokens"],
            source_lengths=batch["source_lengths"],
            max_steps=max_decode_steps,
        )

        for index, source_text in enumerate(batch["source_texts"]):
            predicted_tokens = target_vocab.denumericalize(predicted_ids[index].tolist())
            target_tokens = target_vocab.denumericalize(batch["target_tokens"][index].tolist())
            predicted_sequences.append(predicted_tokens)
            target_sequences.append(target_tokens)
            prediction_records.append(
                build_prediction_record(
                    source_text=source_text,
                    predicted_tokens=predicted_tokens,
                    target_tokens=target_tokens,
                )
            )

    average_loss = total_loss / max(total_target_tokens, 1)
    metrics = EvaluationMetrics(
        loss=average_loss,
        token_accuracy=compute_token_accuracy(predicted_sequences, target_sequences),
        exact_match_accuracy=compute_exact_match_accuracy(predicted_sequences, target_sequences),
        slot_accuracy=compute_slot_accuracy(predicted_sequences, target_sequences),
        num_examples=len(prediction_records),
    )
    return metrics, prediction_records


def save_metrics(metrics: EvaluationMetrics, path: str | Path) -> None:
    """Write evaluation metrics to disk as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")


def save_prediction_records(
    prediction_records: list[dict[str, object]],
    path: str | Path,
    limit: int | None = None,
) -> None:
    """Write representative prediction comparisons to disk as CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = prediction_records[:limit] if limit is not None else prediction_records

    fieldnames = [
        "source_text",
        "predicted_text",
        "target_text",
        "exact_match",
        "token_match_fraction",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
