"""Entry point for model training.

This script assembles the dataset, vocabularies, model, optimizer, and trainer,
then runs end-to-end training with validation and best-checkpoint saving.
"""

from __future__ import annotations

import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data.dataset import VisualEffectDataset, collate_batch, load_examples_from_jsonl
from src.data.generate_dataset import build_and_save_dataset
from src.data.vocab import Vocabulary
from src.models import build_seq2seq_model
from src.training.plotting import plot_loss_curves, plot_training_loss, plot_validation_loss
from src.training.trainer import (
    Trainer,
    build_loss_function,
    resolve_device,
    save_run_config,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the training run."""

    parser = argparse.ArgumentParser(description="Train the Seq2Seq attention model.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training and validation.")
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument("--teacher-forcing", type=float, help="Teacher forcing ratio.")
    parser.add_argument("--gradient-clip", type=float, help="Gradient clipping norm.")
    parser.add_argument("--device", type=str, help="Device name such as cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate synthetic dataset files before training.",
    )
    return parser.parse_args()


def _dataset_artifacts_exist(config_paths: object) -> bool:
    """Return True when the required dataset and vocabulary files already exist."""

    required_paths = [
        config_paths.train_file,
        config_paths.val_file,
        config_paths.test_file,
        config_paths.src_vocab_file,
        config_paths.tgt_vocab_file,
    ]
    return all(path.exists() for path in required_paths)


def main() -> None:
    """Run the full training pipeline from dataset loading to saved artifacts."""

    args = parse_args()
    config = get_default_config()
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.teacher_forcing is not None:
        config.training.teacher_forcing_ratio = args.teacher_forcing
    if args.gradient_clip is not None:
        config.training.gradient_clip = args.gradient_clip
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    set_random_seed(config.training.seed)
    device = resolve_device(config.training.device)
    save_run_config(config, config.paths.run_config_file)

    if args.regenerate_data or not _dataset_artifacts_exist(config.paths):
        build_and_save_dataset(config)

    source_vocab = Vocabulary.load(config.paths.src_vocab_file)
    target_vocab = Vocabulary.load(config.paths.tgt_vocab_file)
    train_examples = load_examples_from_jsonl(config.paths.train_file)
    val_examples = load_examples_from_jsonl(config.paths.val_file)

    train_dataset = VisualEffectDataset(train_examples, source_vocab, target_vocab)
    val_dataset = VisualEffectDataset(val_examples, source_vocab, target_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = build_seq2seq_model(config, source_vocab, target_vocab)
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = build_loss_function(target_vocab.pad_index)

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        target_vocab=target_vocab,
        device=device,
    )
    history = trainer.train(train_loader, val_loader)
    history_dict = history.to_dict()
    history.save(config.paths.train_history_assignment_file)
    plot_loss_curves(history_dict, config.paths.loss_curve_file)
    plot_training_loss(history_dict, config.paths.training_loss_plot_file)
    plot_validation_loss(history_dict, config.paths.validation_loss_plot_file)
    plot_loss_curves(history_dict, config.paths.combined_loss_plot_file)

    print(f"Training complete. Best epoch: {history.best_epoch}")
    print(f"Best checkpoint: {config.paths.best_checkpoint_file}")
    print(f"History saved to: {config.paths.train_history_file}")
    print(f"Assignment history saved to: {config.paths.train_history_assignment_file}")
    print(f"Loss curve saved to: {config.paths.loss_curve_file}")
    print(f"Training loss plot saved to: {config.paths.training_loss_plot_file}")
    print(f"Validation loss plot saved to: {config.paths.validation_loss_plot_file}")
    print(f"Combined loss plot saved to: {config.paths.combined_loss_plot_file}")


if __name__ == "__main__":
    main()
