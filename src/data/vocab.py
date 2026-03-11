"""Vocabulary helpers for source and target token sequences.

The final project will use these utilities to convert token lists into integer
indices for the encoder and decoder. The scaffold keeps the class intentionally
small so it is easy to explain in a report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")


def normalize_text(text: str) -> str:
    """Lowercase text, remove punctuation, and collapse repeated whitespace."""

    normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    """Normalize a string and split it into whitespace-delimited tokens."""

    normalized = normalize_text(text)
    return normalized.split() if normalized else []


@dataclass
class Vocabulary:
    """Simple token-index mapping for the homework project."""

    token_to_index: dict[str, int] = field(default_factory=dict)
    index_to_token: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure special tokens exist when the vocabulary is first created."""

        if not self.token_to_index and not self.index_to_token:
            for token in SPECIAL_TOKENS:
                self.add_token(token)

    def add_token(self, token: str) -> int:
        """Add one token if needed and return its index."""

        if token not in self.token_to_index:
            index = len(self.index_to_token)
            self.token_to_index[token] = index
            self.index_to_token.append(token)
        return self.token_to_index[token]

    def add_tokens(self, tokens: list[str]) -> None:
        """Add a list of tokens to the vocabulary."""

        for token in tokens:
            self.add_token(token)

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens to indices using `<unk>` for missing items."""

        unk_index = self.unk_index
        return [self.token_to_index.get(token, unk_index) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """Convert indices back to string tokens."""

        return [self.index_to_token[index] for index in indices]

    def numericalize(
        self, tokens: list[str], add_sos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode a token list and optionally wrap it with SOS and EOS markers."""

        sequence = list(tokens)
        if add_sos:
            sequence.insert(0, self.sos_token)
        if add_eos:
            sequence.append(self.eos_token)
        return self.encode(sequence)

    def denumericalize(
        self,
        indices: list[int],
        remove_special: bool = True,
        stop_at_eos: bool = True,
    ) -> list[str]:
        """Convert indices back to tokens with optional special-token cleanup."""

        tokens: list[str] = []
        for index in indices:
            token = self.index_to_token[index]
            if stop_at_eos and token == self.eos_token:
                break
            if remove_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    @property
    def pad_token(self) -> str:
        return "<PAD>"

    @property
    def sos_token(self) -> str:
        return "<SOS>"

    @property
    def eos_token(self) -> str:
        return "<EOS>"

    @property
    def unk_token(self) -> str:
        return "<UNK>"

    @property
    def pad_index(self) -> int:
        return self.token_to_index[self.pad_token]

    @property
    def sos_index(self) -> int:
        return self.token_to_index[self.sos_token]

    @property
    def eos_index(self) -> int:
        return self.token_to_index[self.eos_token]

    @property
    def unk_index(self) -> int:
        return self.token_to_index[self.unk_token]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the vocabulary."""

        return {
            "special_tokens": list(SPECIAL_TOKENS),
            "token_to_index": self.token_to_index,
            "index_to_token": self.index_to_token,
        }

    def save(self, path: str | Path) -> None:
        """Write the vocabulary to disk as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from a JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_index={token: int(index) for token, index in payload["token_to_index"].items()},
            index_to_token=list(payload["index_to_token"]),
        )

    def __len__(self) -> int:
        """Return the current vocabulary size."""

        return len(self.index_to_token)


def build_vocabulary_from_token_sequences(token_sequences: Iterable[list[str]]) -> Vocabulary:
    """Create a vocabulary from an iterable of token sequences."""

    vocab = Vocabulary()
    for tokens in token_sequences:
        vocab.add_tokens(tokens)
    return vocab


def build_vocabularies_from_records(
    records: Iterable[dict[str, object]],
) -> tuple[Vocabulary, Vocabulary]:
    """Build source and target vocabularies from dataset records."""

    source_sequences: list[list[str]] = []
    target_sequences: list[list[str]] = []
    for record in records:
        source_tokens = record.get("source_tokens") or tokenize(str(record["source_text"]))
        target_tokens = record.get("target_tokens") or tokenize(str(record["target_text"]))
        source_sequences.append(list(source_tokens))
        target_sequences.append(list(target_tokens))
    return (
        build_vocabulary_from_token_sequences(source_sequences),
        build_vocabulary_from_token_sequences(target_sequences),
    )
