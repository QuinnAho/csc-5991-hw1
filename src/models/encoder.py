"""GRU encoder for natural-language input descriptions.

This encoder embeds padded source token sequences, runs a GRU over them, and
returns both:
- per-time-step encoder outputs for attention
- the final hidden state for decoder initialization

The implementation is intentionally straightforward so the data flow and tensor
shapes are easy to explain in a report.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderGRU(nn.Module):
    """Encode source token sequences into hidden states."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, source_tokens: torch.Tensor, source_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one batch of padded source sequences.

        Shapes:
        - source_tokens: [batch_size, source_length]
        - source_lengths: [batch_size]
        - encoder_outputs: [batch_size, source_length, hidden_dim]
        - hidden_state: [num_layers, batch_size, hidden_dim]
        """

        embedded = self.dropout(self.embedding(source_tokens))
        packed = pack_padded_sequence(
            embedded,
            lengths=source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden_state = self.gru(packed)
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=source_tokens.size(1),
        )
        return encoder_outputs, hidden_state
