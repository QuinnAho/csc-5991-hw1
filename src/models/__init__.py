"""Model components for the encoder-decoder architecture with attention.

This module also exposes a small factory function so training and evaluation
scripts can instantiate the same model configuration without duplicating setup
logic.
"""

from __future__ import annotations

from src.config import ProjectConfig
from src.data.vocab import Vocabulary
from src.models.attention import BahdanauAttention
from src.models.decoder import DecoderGRU
from src.models.encoder import EncoderGRU
from src.models.seq2seq import Seq2SeqGRU


def build_seq2seq_model(
    config: ProjectConfig,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
) -> Seq2SeqGRU:
    """Construct the seq2seq model from config and dataset vocabularies."""

    encoder = EncoderGRU(
        vocab_size=len(source_vocab),
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.encoder_layers,
        dropout=config.model.dropout,
        pad_index=source_vocab.pad_index,
    )
    attention = BahdanauAttention(hidden_dim=config.model.hidden_dim)
    decoder = DecoderGRU(
        vocab_size=len(target_vocab),
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        attention=attention,
        num_layers=config.model.decoder_layers,
        dropout=config.model.dropout,
        pad_index=target_vocab.pad_index,
    )
    return Seq2SeqGRU(
        encoder=encoder,
        decoder=decoder,
        pad_index=source_vocab.pad_index,
        sos_index=target_vocab.sos_index,
        eos_index=target_vocab.eos_index,
    )


__all__ = [
    "BahdanauAttention",
    "DecoderGRU",
    "EncoderGRU",
    "Seq2SeqGRU",
    "build_seq2seq_model",
]
