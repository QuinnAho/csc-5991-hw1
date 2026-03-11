"""
Entry point for final model evaluation.

This script loads the best saved checkpoint, evaluates on the held-out test
split, and exports metrics plus representative prediction comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data.dataset import VisualEffectDataset, collate_batch, load_examples_from_jsonl
from src.data.generate_dataset import build_and_save_dataset
from src.data.vocab import Vocabulary
from src.models import build_seq2seq_model
from src.training.evaluate import evaluate_model, save_metrics, save_prediction_records
from src.training.trainer import build_loss_function, load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate the Seq2Seq attention model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to load. Defaults to the configured best checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, help="Device name such as cpu, cuda, or auto.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=25,
        help="Number of prediction rows to export. Use 0 to export all.",
    )
    return parser.parse_args()


def main() -> None:
    """Run held-out evaluation from a saved checkpoint."""

    args = parse_args()
    config = get_default_config()
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.training.device = args.device

    if not config.paths.test_file.exists() or not config.paths.src_vocab_file.exists():
        build_and_save_dataset(config)

    source_vocab = Vocabulary.load(config.paths.src_vocab_file)
    target_vocab = Vocabulary.load(config.paths.tgt_vocab_file)
    model = build_seq2seq_model(config, source_vocab, target_vocab)
    device = resolve_device(config.training.device)
    checkpoint_path = args.checkpoint or str(config.paths.best_checkpoint_file)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run `python -m src.train` first."
        )
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.to(device)

    test_examples = load_examples_from_jsonl(config.paths.test_file)
    test_dataset = VisualEffectDataset(test_examples, source_vocab, target_vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    criterion = build_loss_function(target_vocab.pad_index)
    metrics, prediction_records = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        target_vocab=target_vocab,
        device=device,
        max_decode_steps=config.data.target_max_length,
    )

    export_limit = None if args.num_examples <= 0 else args.num_examples
    save_metrics(metrics, config.paths.test_metrics_file)
    save_metrics(metrics, config.paths.test_metrics_assignment_file)
    save_prediction_records(
        prediction_records,
        path=config.paths.test_predictions_file,
        limit=export_limit,
    )
    save_prediction_records(
        prediction_records,
        path=config.paths.test_predictions_assignment_file,
        limit=export_limit,
    )

    print(f"Test loss: {metrics.loss:.4f}")
    print(f"Token accuracy: {metrics.token_accuracy:.4f}")
    print(f"Exact-match accuracy: {metrics.exact_match_accuracy:.4f}")
    print(f"Slot accuracy: {metrics.slot_accuracy:.4f}")
    print(f"Metrics saved to: {config.paths.test_metrics_file}")
    print(f"Predictions saved to: {config.paths.test_predictions_file}")
    print(f"Assignment metrics saved to: {config.paths.test_metrics_assignment_file}")
    print(f"Assignment predictions saved to: {config.paths.test_predictions_assignment_file}")


if __name__ == "__main__":
    main()
