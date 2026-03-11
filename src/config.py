"""
Project-wide configuration for data, model, training, and output paths.

This file keeps default settings in one place so the rest of the project can
stay small and easy to reference in the report. The values here are meant to
be simple starting points rather than final tuned hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORT_DIR = ROOT_DIR / "report"
REPORT_ASSETS_DIR = ROOT_DIR / "report_assets"


@dataclass
class DataConfig:
    """Settings for synthetic data generation and preprocessing."""

    effect_types: tuple[str, ...] = ("smoke", "sparks", "fire", "mist")
    velocity_levels: tuple[str, ...] = ("low", "medium", "high")
    directions: tuple[str, ...] = ("up", "down")
    spread_levels: tuple[str, ...] = ("wide", "narrow", "compact", "large")
    drift_values: tuple[str, ...] = ("false", "true")
    burst_values: tuple[str, ...] = ("false", "true")
    flicker_values: tuple[str, ...] = ("low", "high")
    density_values: tuple[str, ...] = ("low", "high")
    paraphrases_per_combination: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    source_max_length: int = 32
    target_max_length: int = 20


@dataclass
class ModelConfig:
    """Settings for the encoder, attention module, and decoder."""

    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.10
    encoder_layers: int = 1
    decoder_layers: int = 1


@dataclass
class TrainingConfig:
    """Settings for optimization and experiment control."""

    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 25
    teacher_forcing_ratio: float = 0.50
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5
    device: str = "cpu"


@dataclass
class PathConfig:
    """Filesystem locations used by the project."""

    data_dir: Path = DATA_DIR
    all_file: Path = DATA_DIR / "all.jsonl"
    train_file: Path = DATA_DIR / "train.jsonl"
    val_file: Path = DATA_DIR / "val.jsonl"
    test_file: Path = DATA_DIR / "test.jsonl"
    dataset_stats_file: Path = DATA_DIR / "dataset_stats.json"
    src_vocab_file: Path = DATA_DIR / "src_vocab.json"
    tgt_vocab_file: Path = DATA_DIR / "tgt_vocab.json"
    checkpoint_dir: Path = OUTPUTS_DIR / "checkpoints"
    best_checkpoint_file: Path = OUTPUTS_DIR / "checkpoints" / "best_model.pt"
    figures_dir: Path = OUTPUTS_DIR / "figures"
    loss_curve_file: Path = OUTPUTS_DIR / "figures" / "loss_curves.png"
    plots_dir: Path = OUTPUTS_DIR / "plots"
    training_loss_plot_file: Path = OUTPUTS_DIR / "plots" / "training_loss.png"
    validation_loss_plot_file: Path = OUTPUTS_DIR / "plots" / "validation_loss.png"
    combined_loss_plot_file: Path = OUTPUTS_DIR / "plots" / "loss_curves.png"
    evaluation_loss_plot_file: Path = OUTPUTS_DIR / "plots" / "test_vs_validation_loss.png"
    metrics_dir: Path = OUTPUTS_DIR / "metrics"
    train_history_file: Path = OUTPUTS_DIR / "metrics" / "train_history.json"
    train_history_assignment_file: Path = OUTPUTS_DIR / "train_history.json"
    run_config_file: Path = OUTPUTS_DIR / "metrics" / "run_config.json"
    test_metrics_file: Path = OUTPUTS_DIR / "metrics" / "test_metrics.json"
    test_metrics_assignment_file: Path = OUTPUTS_DIR / "test_metrics.json"
    predictions_dir: Path = OUTPUTS_DIR / "predictions"
    test_predictions_file: Path = OUTPUTS_DIR / "predictions" / "test_predictions.csv"
    test_predictions_assignment_file: Path = OUTPUTS_DIR / "test_predictions.csv"
    test_examples_md_file: Path = OUTPUTS_DIR / "predictions" / "test_examples.md"
    report_dir: Path = REPORT_DIR
    report_assets_dir: Path = REPORT_ASSETS_DIR
    dataset_summary_md_file: Path = REPORT_ASSETS_DIR / "dataset_summary.md"
    architecture_summary_md_file: Path = REPORT_ASSETS_DIR / "architecture_summary.md"
    hyperparameters_md_file: Path = REPORT_ASSETS_DIR / "hyperparameters.md"
    report_test_examples_md_file: Path = REPORT_ASSETS_DIR / "test_examples.md"
    testing_checklist_md_file: Path = REPORT_ASSETS_DIR / "testing_checklist.md"
    attention_snippet_file: Path = REPORT_ASSETS_DIR / "attention_snippet.py"


@dataclass
class ProjectConfig:
    """Container that groups all project configuration sections together."""

    project_name: str = "csc-5991-hw1-seq2seq-attention"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def get_default_config() -> ProjectConfig:
    """Return a fresh config object for scripts and tests."""

    return ProjectConfig()
