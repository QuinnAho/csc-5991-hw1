"""GRU decoder that predicts canonical parameter tokens.

At each decoding step, the decoder:
- embeds the previous target token
- computes a context vector with Bahdanau attention
- updates its GRU hidden state
- predicts the next target token

The implementation uses a single clear decode step so it is easy to discuss in
the report and easy to annotate line by line.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.attention import BahdanauAttention


class DecoderGRU(nn.Module):
    """Decode target tokens using GRU recurrence plus attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        attention: BahdanauAttention,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.gru = nn.GRU(
            input_size=embedding_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim * 2 + embedding_dim, vocab_size)

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a single time step.

        Shapes:
        - input_tokens: [batch_size]
        - hidden_state: [num_layers, batch_size, hidden_dim]
        - encoder_outputs: [batch_size, source_length, hidden_dim]
        - source_mask: [batch_size, source_length]
        - logits: [batch_size, target_vocab_size]
        - next_hidden_state: [num_layers, batch_size, hidden_dim]
        - attention_weights: [batch_size, source_length]
        """

        embedded = self.dropout(self.embedding(input_tokens))
        context_vector, attention_weights = self.attention(
            decoder_hidden=hidden_state[-1],
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
        )

        gru_input = torch.cat([embedded, context_vector], dim=-1).unsqueeze(1)
        decoder_output, next_hidden_state = self.gru(gru_input, hidden_state)
        decoder_output = decoder_output.squeeze(1)

        logits = self.output_layer(torch.cat([decoder_output, context_vector, embedded], dim=-1))
        return logits, next_hidden_state, attention_weights
