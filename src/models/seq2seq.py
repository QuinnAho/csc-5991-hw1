"""
High-level seq2seq wrapper for training and greedy decoding.

This module ties together the encoder, attention layer, and decoder so scripts
only need to interact with one model class. The wrapper supports:
- teacher forcing during training
- greedy autoregressive decoding during inference
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.decoder import DecoderGRU
from src.models.encoder import EncoderGRU


class Seq2SeqGRU(nn.Module):
    """End-to-end sequence-to-sequence model with attention."""

    def __init__(
        self,
        encoder: EncoderGRU,
        decoder: DecoderGRU,
        pad_index: int = 0,
        sos_index: int = 1,
        eos_index: int = 2,
    ) -> None:
        super().__init__()
        if encoder.hidden_dim != decoder.hidden_dim:
            raise ValueError("Encoder and decoder hidden dimensions must match.")
        if encoder.num_layers != decoder.num_layers:
            raise ValueError("Encoder and decoder must use the same number of GRU layers.")

        self.encoder = encoder
        self.decoder = decoder
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_lengths: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Run a training-time forward pass over a batch.

        Shapes:
        - source_tokens: [batch_size, source_length]
        - source_lengths: [batch_size]
        - target_tokens: [batch_size, target_length]
        - returned logits: [batch_size, target_length - 1, target_vocab_size]

        The logits align with `target_tokens[:, 1:]` because the first target
        token is the `<SOS>` start symbol fed into the decoder.
        """

        batch_size, target_length = target_tokens.shape
        target_vocab_size = self.decoder.output_layer.out_features

        encoder_outputs, hidden_state = self.encoder(source_tokens, source_lengths)
        source_mask = source_tokens.ne(self.pad_index)

        logits = torch.zeros(
            batch_size,
            target_length - 1,
            target_vocab_size,
            device=source_tokens.device,
        )

        decoder_input = target_tokens[:, 0]
        decoder_hidden = hidden_state

        for step in range(1, target_length):
            step_logits, decoder_hidden, _ = self.decoder(
                input_tokens=decoder_input,
                hidden_state=decoder_hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )
            logits[:, step - 1, :] = step_logits

            teacher_force = torch.rand(1, device=source_tokens.device).item() < teacher_forcing_ratio
            predicted_tokens = step_logits.argmax(dim=-1)
            decoder_input = target_tokens[:, step] if teacher_force else predicted_tokens

        return logits

    @torch.no_grad()
    def greedy_decode(
        self, source_tokens: torch.Tensor, source_lengths: torch.Tensor, max_steps: int
    ) -> torch.Tensor:
        """Generate target sequences without teacher forcing.

        Shapes:
        - source_tokens: [batch_size, source_length]
        - returned predictions: [batch_size, decoded_length]

        The returned tensor does not include the initial `<SOS>` token.
        """

        batch_size = source_tokens.size(0)
        encoder_outputs, hidden_state = self.encoder(source_tokens, source_lengths)
        source_mask = source_tokens.ne(self.pad_index)

        decoder_input = torch.full(
            (batch_size,),
            fill_value=self.sos_index,
            dtype=torch.long,
            device=source_tokens.device,
        )
        decoder_hidden = hidden_state

        predictions: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_tokens.device)

        for _ in range(max_steps):
            step_logits, decoder_hidden, _ = self.decoder(
                input_tokens=decoder_input,
                hidden_state=decoder_hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )
            decoder_input = step_logits.argmax(dim=-1)
            predictions.append(decoder_input)
            finished = finished | decoder_input.eq(self.eos_index)
            if torch.all(finished):
                break

        if not predictions:
            return torch.empty(batch_size, 0, dtype=torch.long, device=source_tokens.device)

        return torch.stack(predictions, dim=1)
