"""Dataset containers and collation utilities for seq2seq training.

This module loads saved JSONL splits, converts text to token ids with source
and target vocabularies, and pads minibatches for GRU-based seq2seq training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.vocab import Vocabulary, tokenize


@dataclass
class Seq2SeqExample:
    """One tokenized source-target pair plus optional structured metadata."""

    source_text: str
    target_text: str
    source_tokens: list[str]
    target_tokens: list[str]
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Seq2SeqExample":
        """Build an example object from one saved JSONL record."""

        reserved_keys = {"source_text", "target_text", "source_tokens", "target_tokens"}
        return cls(
            source_text=str(record["source_text"]),
            target_text=str(record["target_text"]),
            source_tokens=list(record.get("source_tokens") or tokenize(str(record["source_text"]))),
            target_tokens=list(record.get("target_tokens") or tokenize(str(record["target_text"]))),
            metadata={key: str(value) for key, value in record.items() if key not in reserved_keys},
        )


class VisualEffectDataset(Dataset[dict[str, Any]]):
    """Dataset wrapper that returns numericalized seq2seq training examples."""

    def __init__(
        self,
        examples: Sequence[Seq2SeqExample],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        self.examples = list(examples)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "source_text": example.source_text,
            "target_text": example.target_text,
            "source_tokens": example.source_tokens,
            "target_tokens": example.target_tokens,
            "source_ids": self.source_vocab.numericalize(example.source_tokens, add_eos=True),
            "target_ids": self.target_vocab.numericalize(
                example.target_tokens, add_sos=True, add_eos=True
            ),
            "source_pad_index": self.source_vocab.pad_index,
            "target_pad_index": self.target_vocab.pad_index,
            "metadata": example.metadata,
        }


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of Python dictionaries."""

    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_examples_from_jsonl(path: str | Path) -> list[Seq2SeqExample]:
    """Load JSONL records and convert them into example objects."""

    return [Seq2SeqExample.from_record(record) for record in load_jsonl_records(path)]


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad one minibatch of numericalized examples into PyTorch tensors."""

    if not batch:
        raise ValueError("collate_batch received an empty batch.")

    source_pad_index = int(batch[0]["source_pad_index"])
    target_pad_index = int(batch[0]["target_pad_index"])

    source_sequences = [torch.tensor(item["source_ids"], dtype=torch.long) for item in batch]
    target_sequences = [torch.tensor(item["target_ids"], dtype=torch.long) for item in batch]

    source_lengths = torch.tensor([len(sequence) for sequence in source_sequences], dtype=torch.long)
    target_lengths = torch.tensor([len(sequence) for sequence in target_sequences], dtype=torch.long)

    padded_source = pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=source_pad_index,
    )
    padded_target = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=target_pad_index,
    )

    return {
        "source_tokens": padded_source,
        "source_lengths": source_lengths,
        "source_mask": padded_source.ne(source_pad_index),
        "target_tokens": padded_target,
        "target_lengths": target_lengths,
        "target_mask": padded_target.ne(target_pad_index),
        "source_texts": [item["source_text"] for item in batch],
        "target_texts": [item["target_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
