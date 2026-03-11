"""Metric definitions for sequence prediction quality.

The final project will report token accuracy, exact-sequence accuracy, and
field-level accuracy so the results are easy to interpret in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SPECIAL_TOKENS = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
SLOT_NAMES = ("effect", "velocity", "direction", "spread", "drift", "burst", "flicker", "density")


@dataclass
class EvaluationMetrics:
    """Summary values for one validation or test run."""

    loss: float
    token_accuracy: float
    exact_match_accuracy: float
    slot_accuracy: float
    num_examples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly metrics dictionary."""

        return {
            "loss": self.loss,
            "token_accuracy": self.token_accuracy,
            "exact_match_accuracy": self.exact_match_accuracy,
            "slot_accuracy": self.slot_accuracy,
            "num_examples": self.num_examples,
        }


def _clean_tokens(tokens: Sequence[str]) -> list[str]:
    """Remove special tokens while preserving the generated token order."""

    return [token for token in tokens if token not in SPECIAL_TOKENS]


def _pairwise_token_accuracy(predicted_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
    """Compute token accuracy for one predicted-target pair."""

    predicted = _clean_tokens(predicted_tokens)
    target = _clean_tokens(target_tokens)
    denominator = max(len(predicted), len(target))
    if denominator == 0:
        return 1.0

    correct = 0
    for index in range(denominator):
        predicted_token = predicted[index] if index < len(predicted) else None
        target_token = target[index] if index < len(target) else None
        correct += int(predicted_token == target_token)
    return correct / denominator


def _tokens_to_slots(tokens: Sequence[str]) -> dict[str, str]:
    """Parse a flat canonical token sequence into slot-value pairs."""

    cleaned = _clean_tokens(tokens)
    slots: dict[str, str] = {}
    for index in range(0, len(cleaned) - 1, 2):
        slot_name = cleaned[index]
        slot_value = cleaned[index + 1]
        if slot_name in SLOT_NAMES:
            slots[slot_name] = slot_value
    return slots


def compute_token_accuracy(
    predicted_sequences: Sequence[Sequence[str]],
    target_sequences: Sequence[Sequence[str]],
) -> float:
    """Compute average token-level accuracy across a dataset."""

    if not predicted_sequences:
        return 0.0
    pair_scores = [
        _pairwise_token_accuracy(predicted_tokens, target_tokens)
        for predicted_tokens, target_tokens in zip(predicted_sequences, target_sequences)
    ]
    return sum(pair_scores) / len(pair_scores)


def compute_exact_match_accuracy(
    predicted_sequences: Sequence[Sequence[str]],
    target_sequences: Sequence[Sequence[str]],
) -> float:
    """Compute the fraction of examples with a fully correct target sequence."""

    if not predicted_sequences:
        return 0.0
    exact_matches = [
        int(_clean_tokens(predicted_tokens) == _clean_tokens(target_tokens))
        for predicted_tokens, target_tokens in zip(predicted_sequences, target_sequences)
    ]
    return sum(exact_matches) / len(exact_matches)


def compute_slot_accuracy(
    predicted_sequences: Sequence[Sequence[str]],
    target_sequences: Sequence[Sequence[str]],
) -> float:
    """Compute average slot-value accuracy across a dataset."""

    if not predicted_sequences:
        return 0.0

    total_slots = 0
    correct_slots = 0
    for predicted_tokens, target_tokens in zip(predicted_sequences, target_sequences):
        predicted_slots = _tokens_to_slots(predicted_tokens)
        target_slots = _tokens_to_slots(target_tokens)
        for slot_name, target_value in target_slots.items():
            total_slots += 1
            correct_slots += int(predicted_slots.get(slot_name) == target_value)

    return correct_slots / total_slots if total_slots else 0.0
