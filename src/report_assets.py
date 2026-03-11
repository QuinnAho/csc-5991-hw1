"""Generate report-ready plots, markdown summaries, and code snippets.

This script turns the saved dataset, training, and evaluation artifacts into
the assignment deliverables that are easiest to paste into a university report.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.config import ProjectConfig, get_default_config
from src.data.dataset import VisualEffectDataset, collate_batch, load_examples_from_jsonl
from src.data.generate_dataset import build_and_save_dataset
from src.data.vocab import Vocabulary
from src.models import build_seq2seq_model
from src.training.evaluate import evaluate_model, save_metrics, save_prediction_records
from src.training.plotting import (
    plot_evaluation_loss_summary,
    plot_loss_curves,
    plot_training_loss,
    plot_validation_loss,
)
from src.training.trainer import build_loss_function, load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command-line options for report-asset generation."""

    parser = argparse.ArgumentParser(description="Export report-ready homework assets.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of representative test examples to include in markdown.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use if test predictions need to be regenerated.",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a Python dictionary."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text to disk, creating parent directories if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple GitHub-flavored markdown table."""

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    """Convert config dataclasses into plain JSON-like data."""

    return json.loads(json.dumps(asdict(config), default=str))


def ensure_dataset_artifacts(config: ProjectConfig) -> None:
    """Build the dataset if the expected dataset files do not exist yet."""

    required_paths = [
        config.paths.dataset_stats_file,
        config.paths.train_file,
        config.paths.val_file,
        config.paths.test_file,
        config.paths.src_vocab_file,
        config.paths.tgt_vocab_file,
    ]
    if not all(path.exists() for path in required_paths):
        build_and_save_dataset(config)


def ensure_test_artifacts(
    config: ProjectConfig,
    device_name: str,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Load or regenerate test metrics and prediction rows from the best checkpoint."""

    if not config.paths.best_checkpoint_file.exists():
        if config.paths.test_metrics_file.exists() and config.paths.test_predictions_file.exists():
            return _load_json(config.paths.test_metrics_file), load_prediction_records(
                config.paths.test_predictions_file
            )
        raise FileNotFoundError(
            f"Checkpoint not found at {config.paths.best_checkpoint_file}. "
            "Run `python -m src.train` before exporting report assets."
        )

    source_vocab = Vocabulary.load(config.paths.src_vocab_file)
    target_vocab = Vocabulary.load(config.paths.tgt_vocab_file)
    model = build_seq2seq_model(config, source_vocab, target_vocab)
    device = resolve_device(device_name)
    load_checkpoint(config.paths.best_checkpoint_file, model=model, device=device)
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
    save_metrics(metrics, config.paths.test_metrics_file)
    save_metrics(metrics, config.paths.test_metrics_assignment_file)
    save_prediction_records(prediction_records, config.paths.test_predictions_file, limit=None)
    save_prediction_records(
        prediction_records,
        config.paths.test_predictions_assignment_file,
        limit=None,
    )
    return metrics.to_dict(), prediction_records


def load_prediction_records(path: str | Path) -> list[dict[str, object]]:
    """Load exported prediction comparisons from CSV."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_exact_match(record: dict[str, object]) -> bool:
    """Return True when a prediction record represents an exact match."""

    return str(record["exact_match"]).strip().lower() in {"1", "true", "yes"}


def _parse_slot_values(target_text: str) -> dict[str, str]:
    """Parse a canonical target sequence into slot-value pairs."""

    tokens = target_text.split()
    return {
        tokens[index]: tokens[index + 1]
        for index in range(0, len(tokens) - 1, 2)
    }


def select_representative_examples(
    prediction_records: list[dict[str, object]],
    num_examples: int,
) -> tuple[list[dict[str, object]], str | None]:
    """Choose a diverse set of held-out examples for the report."""

    if not prediction_records or num_examples <= 0:
        return [], None

    incorrect = [record for record in prediction_records if not _is_exact_match(record)]
    correct = [record for record in prediction_records if _is_exact_match(record)]

    selected: list[dict[str, object]] = []
    used_sources: set[str] = set()
    covered_keys: set[tuple[str, str, str]] = set()

    def append_record(record: dict[str, object]) -> None:
        source_text = str(record["source_text"])
        if source_text in used_sources or len(selected) >= num_examples:
            return
        slot_values = _parse_slot_values(str(record["target_text"]))
        diversity_key = (
            slot_values.get("effect", ""),
            slot_values.get("direction", ""),
            slot_values.get("spread", ""),
        )
        selected.append(record)
        used_sources.add(source_text)
        covered_keys.add(diversity_key)

    for record in sorted(incorrect, key=lambda row: float(row["token_match_fraction"])):
        append_record(record)

    for effect_name in ("smoke", "sparks", "fire", "mist"):
        for record in correct:
            slot_values = _parse_slot_values(str(record["target_text"]))
            diversity_key = (
                slot_values.get("effect", ""),
                slot_values.get("direction", ""),
                slot_values.get("spread", ""),
            )
            if slot_values.get("effect") == effect_name and diversity_key not in covered_keys:
                append_record(record)
                break

    for record in correct:
        slot_values = _parse_slot_values(str(record["target_text"]))
        diversity_key = (
            slot_values.get("effect", ""),
            slot_values.get("direction", ""),
            slot_values.get("spread", ""),
        )
        if diversity_key not in covered_keys:
            append_record(record)

    for record in prediction_records:
        append_record(record)

    note = None
    if not incorrect:
        note = (
            "No incorrect held-out examples were available for the final checkpoint because it "
            f"achieved exact-match accuracy on all {len(prediction_records)} test examples."
        )

    return selected[:num_examples], note


def export_dataset_summary(dataset_stats: dict[str, Any], output_path: str | Path) -> None:
    """Write a markdown dataset summary for the report."""

    split_counts = dataset_stats["split_counts"]
    vocab_sizes = dataset_stats["vocabulary_sizes"]
    schema_rows = [
        [name, ", ".join(values)]
        for name, values in dataset_stats["schema"].items()
    ]
    sample_records = dataset_stats["sample_examples"]["train"][:3]

    content = [
        "# Dataset Summary",
        "",
        "## Counts",
        "",
        _markdown_table(
            ["Item", "Value"],
            [
                ["Total examples", str(dataset_stats["total_examples"])],
                ["Train examples", str(split_counts["train"])],
                ["Validation examples", str(split_counts["validation"])],
                ["Test examples", str(split_counts["test"])],
                ["Source vocabulary size", str(vocab_sizes["source"])],
                ["Target vocabulary size", str(vocab_sizes["target"])],
                ["Paraphrases per combination", str(dataset_stats["paraphrases_per_combination"])],
            ],
        ),
        "",
        "## Controlled Schema",
        "",
        _markdown_table(["Field", "Allowed values"], schema_rows),
        "",
        "## Sample Training Examples",
        "",
    ]

    for index, sample in enumerate(sample_records, start=1):
        content.extend(
            [
                f"### Example {index}",
                "",
                f"- Source: `{sample['source_text']}`",
                f"- Target: `{sample['target_text']}`",
                "",
            ]
        )

    _write_text(output_path, "\n".join(content).strip() + "\n")


def export_architecture_summary(
    config: ProjectConfig,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    output_path: str | Path,
) -> None:
    """Write a markdown architecture summary based on the current model config."""

    model = build_seq2seq_model(config, source_vocab, target_vocab)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    content = [
        "# Architecture Summary",
        "",
        "## Model Overview",
        "",
        "- Encoder: token embedding followed by a GRU.",
        "- Attention: Bahdanau additive attention over all encoder outputs.",
        "- Decoder: target embedding plus attention context fed into a GRU.",
        "- Wrapper: teacher forcing during training and greedy autoregressive decoding during inference.",
        "",
        "## Key Dimensions",
        "",
        _markdown_table(
            ["Component", "Value"],
            [
                ["Source vocabulary size", str(len(source_vocab))],
                ["Target vocabulary size", str(len(target_vocab))],
                ["Embedding dimension", str(config.model.embedding_dim)],
                ["Hidden dimension", str(config.model.hidden_dim)],
                ["Encoder GRU layers", str(config.model.encoder_layers)],
                ["Decoder GRU layers", str(config.model.decoder_layers)],
                ["Dropout", str(config.model.dropout)],
                ["Total parameters", f"{total_params:,}"],
                ["Trainable parameters", f"{trainable_params:,}"],
            ],
        ),
        "",
        "## Tensor Flow",
        "",
        "- Encoder outputs: `[batch_size, source_length, hidden_dim]`",
        "- Encoder final hidden state: `[num_layers, batch_size, hidden_dim]`",
        "- Attention weights: `[batch_size, source_length]`",
        "- Decoder step logits: `[batch_size, target_vocab_size]`",
        "- Training logits: `[batch_size, target_length - 1, target_vocab_size]`",
        "",
        "## Attention Type",
        "",
        "This project uses additive Bahdanau attention: the decoder state and each encoder state are projected into a shared space, combined with `tanh`, scored, masked over padded tokens, and normalized with `softmax`.",
        "",
    ]
    _write_text(output_path, "\n".join(content))


def export_hyperparameters(
    config_dict: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a markdown summary of dataset, model, and training hyperparameters."""

    data_rows = [[key, str(value)] for key, value in config_dict["data"].items()]
    model_rows = [[key, str(value)] for key, value in config_dict["model"].items()]
    training_rows = [[key, str(value)] for key, value in config_dict["training"].items()]

    content = [
        "# Hyperparameter Summary",
        "",
        "## Data Settings",
        "",
        _markdown_table(["Hyperparameter", "Value"], data_rows),
        "",
        "## Model Settings",
        "",
        _markdown_table(["Hyperparameter", "Value"], model_rows),
        "",
        "## Training Settings",
        "",
        _markdown_table(["Hyperparameter", "Value"], training_rows),
        "",
    ]
    _write_text(output_path, "\n".join(content))


def export_test_examples(
    prediction_records: list[dict[str, object]],
    output_path: str | Path,
    num_examples: int,
) -> None:
    """Write representative held-out predictions to a markdown table."""

    selected_records, note = select_representative_examples(prediction_records, num_examples)
    rows = [
        [
            str(record["source_text"]),
            str(record["predicted_text"]),
            str(record["target_text"]),
            "Yes" if _is_exact_match(record) else "No",
        ]
        for record in selected_records
    ]

    content = [
        "# Representative Test Examples",
        "",
        f"Selected {len(selected_records)} held-out examples for the final report.",
        "",
        _markdown_table(
            ["Input", "Model Output", "Ground Truth", "Correct?"],
            rows,
        ),
        "",
    ]
    if note is not None:
        content.extend([note, ""])
    _write_text(output_path, "\n".join(content))


def export_testing_checklist(output_path: str | Path) -> None:
    """Write a report-readiness testing checklist as markdown."""

    content = [
        "# Testing Checklist",
        "",
        "Use this checklist before treating the project as report-ready.",
        "",
        "- [ ] `outputs/plots/training_loss.png` exists and opens.",
        "- [ ] `outputs/plots/validation_loss.png` exists and opens.",
        "- [ ] `outputs/plots/loss_curves.png` exists and shows both train and validation loss.",
        "- [ ] `outputs/plots/test_vs_validation_loss.png` exists and shows held-out loss comparison.",
        "- [ ] `outputs/predictions/test_predictions.csv` exists and contains input, prediction, and target columns.",
        "- [ ] `outputs/predictions/test_examples.md` contains the representative held-out examples you want in the report.",
        "- [ ] `report_assets/test_examples.md` matches the curated example table used in the report.",
        "- [ ] `report_assets/dataset_summary.md` includes total counts, split counts, schema, and sample examples.",
        "- [ ] `report_assets/architecture_summary.md` matches the final model you trained.",
        "- [ ] `report_assets/hyperparameters.md` matches the run you want to report.",
        "- [ ] `report_assets/attention_snippet.py` is the exact attention block you want to annotate in the report.",
        "- [ ] `outputs/metrics/train_history.json` exists and matches the plots.",
        "- [ ] `outputs/metrics/test_metrics.json` exists and matches the checkpoint you are reporting.",
        "- [ ] `outputs/checkpoints/best_model.pt` exists and is the checkpoint used for final evaluation.",
        "- [ ] After any new training run, `python -m src.report_assets` has been rerun so all assets are current.",
        "",
        "## Suggested Verification Order",
        "",
        "1. Run `python -m src.train` with the final hyperparameters.",
        "2. Run `python -m src.test` to refresh held-out metrics and predictions.",
        "3. Run `python -m src.report_assets --num-examples 10` to regenerate report artifacts.",
        "4. Open the plots and markdown files in `report_assets/` and confirm they match the run you intend to submit.",
        "",
    ]
    _write_text(output_path, "\n".join(content))


def export_attention_snippet(output_path: str | Path) -> None:
    """Write a line-by-line commented attention snippet for the report."""

    snippet = '''"""Bahdanau attention snippet with line-by-line report comments."""

import torch

def forward(
    self,
    decoder_hidden: torch.Tensor,
    encoder_outputs: torch.Tensor,
    source_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Project the current decoder state into the shared attention space.
    query = self.query_layer(decoder_hidden).unsqueeze(1)

    # 2. Project every encoder output vector into the same attention space.
    keys = self.key_layer(encoder_outputs)

    # 3. Combine query and keys with tanh to create additive attention energy.
    energy = torch.tanh(query + keys)

    # 4. Collapse each source position to one unnormalized attention score.
    scores = self.energy_layer(energy).squeeze(-1)

    # 5. Mask padding positions so the decoder cannot attend to padded tokens.
    scores = scores.masked_fill(~source_mask, float("-inf"))

    # 6. Normalize scores across the source sequence.
    attention_weights = torch.softmax(scores, dim=-1)

    # 7. Compute a weighted sum of encoder outputs to get the context vector.
    context_vector = torch.bmm(
        attention_weights.unsqueeze(1),
        encoder_outputs,
    ).squeeze(1)

    # 8. Return both the context vector and the attention weights.
    return context_vector, attention_weights
'''
    _write_text(output_path, snippet)


def main() -> None:
    """Generate plots, markdown summaries, and the annotated attention snippet."""

    args = parse_args()
    config = get_default_config()
    if args.device is not None:
        config.training.device = args.device

    ensure_dataset_artifacts(config)
    if not config.paths.train_history_file.exists():
        raise FileNotFoundError(
            f"Training history not found at {config.paths.train_history_file}. "
            "Run `python -m src.train` before exporting report assets."
        )

    history = _load_json(config.paths.train_history_file)
    dataset_stats = _load_json(config.paths.dataset_stats_file)
    run_config = (
        _load_json(config.paths.run_config_file)
        if config.paths.run_config_file.exists()
        else _config_to_dict(config)
    )
    test_metrics, prediction_records = ensure_test_artifacts(config, config.training.device)

    plot_training_loss(history, config.paths.training_loss_plot_file)
    plot_validation_loss(history, config.paths.validation_loss_plot_file)
    plot_loss_curves(history, config.paths.combined_loss_plot_file)
    plot_evaluation_loss_summary(history, test_metrics, config.paths.evaluation_loss_plot_file)

    source_vocab = Vocabulary.load(config.paths.src_vocab_file)
    target_vocab = Vocabulary.load(config.paths.tgt_vocab_file)
    export_dataset_summary(dataset_stats, config.paths.dataset_summary_md_file)
    export_architecture_summary(
        config=config,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        output_path=config.paths.architecture_summary_md_file,
    )
    export_hyperparameters(run_config, config.paths.hyperparameters_md_file)
    export_test_examples(prediction_records, config.paths.test_examples_md_file, args.num_examples)
    export_test_examples(
        prediction_records,
        config.paths.report_test_examples_md_file,
        args.num_examples,
    )
    export_testing_checklist(config.paths.testing_checklist_md_file)
    export_attention_snippet(config.paths.attention_snippet_file)

    print(f"Generated plot: {config.paths.training_loss_plot_file}")
    print(f"Generated plot: {config.paths.validation_loss_plot_file}")
    print(f"Generated plot: {config.paths.combined_loss_plot_file}")
    print(f"Generated plot: {config.paths.evaluation_loss_plot_file}")
    print(f"Generated markdown: {config.paths.test_examples_md_file}")
    print(f"Generated markdown: {config.paths.report_test_examples_md_file}")
    print(f"Generated markdown: {config.paths.dataset_summary_md_file}")
    print(f"Generated markdown: {config.paths.architecture_summary_md_file}")
    print(f"Generated markdown: {config.paths.hyperparameters_md_file}")
    print(f"Generated markdown: {config.paths.testing_checklist_md_file}")
    print(f"Generated snippet: {config.paths.attention_snippet_file}")


if __name__ == "__main__":
    main()
