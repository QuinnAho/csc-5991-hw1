"""
Bahdanau additive attention for the decoder.

This file is intentionally separated because the assignment requires an
annotated attention code block in the report. Keeping the logic isolated makes
the later explanation much cleaner.
"""

from __future__ import annotations

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    """Compute attention weights over encoder outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a context vector and normalized attention weights.

        Shapes:
        - decoder_hidden: [batch_size, hidden_dim]
        - encoder_outputs: [batch_size, source_length, hidden_dim]
        - source_mask: [batch_size, source_length]
        - context_vector: [batch_size, hidden_dim]
        - attention_weights: [batch_size, source_length]
        """

        # Project the decoder state and encoder states into the same space.
        query = self.query_layer(decoder_hidden).unsqueeze(1)
        keys = self.key_layer(encoder_outputs)

        # Bahdanau attention uses tanh(query + key) before the final score.
        energy = torch.tanh(query + keys)
        scores = self.energy_layer(energy).squeeze(-1)

        # Ignore padded source positions so the decoder cannot attend to them.
        scores = scores.masked_fill(~source_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of encoder outputs gives the context vector.
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)
        return context_vector, attention_weights
