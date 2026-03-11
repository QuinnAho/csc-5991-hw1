"""Synthetic dataset generation for the visual-effect translation task.

Each example pairs:
- a natural-language visual-effect description
- a deterministic structured target token sequence

The dataset is intentionally controlled and fully reproducible so it is easy to
explain in a report and easy for a student team to debug.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.config import ProjectConfig, get_default_config
from src.data.vocab import (
    SPECIAL_TOKENS,
    build_vocabularies_from_records,
    normalize_text,
    tokenize,
)


@dataclass(frozen=True)
class ExampleSpec:
    """Canonical structured representation for one target example."""

    effect: str
    velocity: str
    direction: str
    spread: str
    drift: str
    burst: str
    flicker: str
    density: str


EFFECT_PHRASES = {
    "smoke": ["smoke", "smoke effect", "smoky plume"],
    "sparks": ["sparks", "spark effect", "spark shower"],
    "fire": ["fire", "fire effect", "flame burst"],
    "mist": ["mist", "mist effect", "misty cloud"],
}

VELOCITY_ADVERBS = {
    "low": ["slowly", "at low speed", "with low velocity"],
    "medium": ["at medium speed", "with medium velocity", "moderately"],
    "high": ["quickly", "at high speed", "with high velocity"],
}

VELOCITY_PHRASES = {
    "low": ["a low speed", "low velocity", "slow motion"],
    "medium": ["a medium speed", "medium velocity", "steady motion"],
    "high": ["a high speed", "high velocity", "fast motion"],
}

DIRECTION_VERBS = {
    "up": ["rise upward", "move up", "travel upward"],
    "down": ["fall downward", "move down", "drop downward"],
}

DIRECTION_WORDS = {
    "up": ["up", "upward"],
    "down": ["down", "downward"],
}

SPREAD_CLAUSES = {
    "wide": ["spread wide", "open into a wide shape", "fan out wide"],
    "narrow": ["stay narrow", "keep a narrow spread", "spread in a narrow shape"],
    "compact": ["stay compact", "keep a compact spread", "remain compact"],
    "large": ["spread large", "open into a large shape", "grow into a large spread"],
}

DRIFT_CLAUSES = {
    "true": ["drift enabled", "drifting motion", "drift active"],
    "false": ["drift disabled", "no drifting", "no drift"],
}

BURST_CLAUSES = {
    "true": ["burst enabled", "burst active", "bursting behavior"],
    "false": ["burst disabled", "no burst", "burst inactive"],
}

FLICKER_CLAUSES = {
    "low": ["low flicker", "soft flicker", "gentle flicker"],
    "high": ["high flicker", "strong flicker", "intense flicker"],
}

DENSITY_CLAUSES = {
    "low": ["low density", "light density", "thin density"],
    "high": ["high density", "dense look", "heavy density"],
}


def build_target_tokens(spec: ExampleSpec) -> list[str]:
    """Convert a structured example into the canonical target token sequence."""

    return [
        "effect",
        spec.effect,
        "velocity",
        spec.velocity,
        "direction",
        spec.direction,
        "spread",
        spec.spread,
        "drift",
        spec.drift,
        "burst",
        spec.burst,
        "flicker",
        spec.flicker,
        "density",
        spec.density,
    ]


def _stable_choice(options: list[str], key: str) -> str:
    """Select one option deterministically from a list using a stable hash."""

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def _spec_key(spec: ExampleSpec) -> str:
    """Return a stable string key for one structured example."""

    return "|".join(
        [
            spec.effect,
            spec.velocity,
            spec.direction,
            spec.spread,
            spec.drift,
            spec.burst,
            spec.flicker,
            spec.density,
        ]
    )


def _render_template(spec: ExampleSpec, template_id: int) -> str:
    """Render one natural-language paraphrase for a structured example."""

    key_prefix = f"{_spec_key(spec)}|template:{template_id}"
    effect = _stable_choice(EFFECT_PHRASES[spec.effect], f"{key_prefix}|effect")
    velocity_adverb = _stable_choice(
        VELOCITY_ADVERBS[spec.velocity], f"{key_prefix}|velocity_adverb"
    )
    velocity_phrase = _stable_choice(
        VELOCITY_PHRASES[spec.velocity], f"{key_prefix}|velocity_phrase"
    )
    direction_verb = _stable_choice(DIRECTION_VERBS[spec.direction], f"{key_prefix}|direction_verb")
    direction_word = _stable_choice(DIRECTION_WORDS[spec.direction], f"{key_prefix}|direction_word")
    spread = _stable_choice(SPREAD_CLAUSES[spec.spread], f"{key_prefix}|spread")
    drift = _stable_choice(DRIFT_CLAUSES[spec.drift], f"{key_prefix}|drift")
    burst = _stable_choice(BURST_CLAUSES[spec.burst], f"{key_prefix}|burst")
    flicker = _stable_choice(FLICKER_CLAUSES[spec.flicker], f"{key_prefix}|flicker")
    density = _stable_choice(DENSITY_CLAUSES[spec.density], f"{key_prefix}|density")

    if template_id % 4 == 0:
        sentence = (
            f"make the {effect} {direction_verb} {velocity_adverb} and {spread}, "
            f"{drift}, {burst}, {flicker}, and {density}"
        )
    elif template_id % 4 == 1:
        sentence = (
            f"create {effect} that moves {direction_word} at {velocity_phrase}, "
            f"{spread}, {drift}, {burst}, {flicker}, and {density}"
        )
    elif template_id % 4 == 2:
        sentence = (
            f"design {effect} to travel {direction_word} with {velocity_phrase}, "
            f"{spread}, {drift}, {burst}, {flicker}, and {density}"
        )
    else:
        sentence = (
            f"build a {effect} effect that will {direction_verb}, {spread}, use "
            f"{velocity_phrase}, {drift}, {burst}, {flicker}, and {density}"
        )
    return normalize_text(sentence)


def generate_natural_language(spec: ExampleSpec, num_variants: int) -> list[str]:
    """Generate deterministic paraphrases for one structured example."""

    paraphrases: list[str] = []
    seen: set[str] = set()
    template_id = 0

    while len(paraphrases) < num_variants:
        candidate = _render_template(spec, template_id)
        template_id += 1
        if candidate not in seen:
            paraphrases.append(candidate)
            seen.add(candidate)
        if template_id > num_variants * 8 and len(paraphrases) < num_variants:
            raise ValueError(f"Unable to generate enough unique paraphrases for spec: {spec}")

    return paraphrases


def _enumerate_specs(config: ProjectConfig) -> Iterable[ExampleSpec]:
    """Yield every structured combination in the controlled dataset schema."""

    data_config = config.data
    for values in itertools.product(
        data_config.effect_types,
        data_config.velocity_levels,
        data_config.directions,
        data_config.spread_levels,
        data_config.drift_values,
        data_config.burst_values,
        data_config.flicker_values,
        data_config.density_values,
    ):
        yield ExampleSpec(*values)


def _build_record(spec: ExampleSpec, source_text: str) -> dict[str, object]:
    """Create one JSON-serializable dataset record."""

    target_tokens = build_target_tokens(spec)
    target_text = " ".join(target_tokens)
    return {
        "source_text": source_text,
        "target_text": target_text,
        "source_tokens": tokenize(source_text),
        "target_tokens": target_tokens,
        **asdict(spec),
    }


def generate_examples(config: ProjectConfig) -> list[dict[str, object]]:
    """Build the full synthetic dataset in memory."""

    examples: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    source_to_target: dict[str, str] = {}

    for spec in _enumerate_specs(config):
        paraphrases = generate_natural_language(spec, config.data.paraphrases_per_combination)
        for source_text in paraphrases:
            record = _build_record(spec, source_text)
            pair_key = (str(record["source_text"]), str(record["target_text"]))
            if pair_key in seen_pairs:
                continue
            previous_target = source_to_target.get(str(record["source_text"]))
            if previous_target is not None and previous_target != str(record["target_text"]):
                raise ValueError(
                    "Ambiguous dataset generation produced the same source text for different targets: "
                    f"{record['source_text']}"
                )
            seen_pairs.add(pair_key)
            source_to_target[str(record["source_text"])] = str(record["target_text"])
            examples.append(record)

    examples.sort(key=lambda item: (str(item["source_text"]), str(item["target_text"])))
    return examples


def split_examples(
    examples: Iterable[dict[str, object]], config: ProjectConfig
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Split examples into train, validation, and test partitions deterministically."""

    all_examples = list(examples)
    ratios_total = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if abs(ratios_total - 1.0) > 1e-9:
        raise ValueError("Train/validation/test ratios must sum to 1.0.")

    shuffled = list(all_examples)
    random.Random(config.training.seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * config.data.train_ratio)
    val_count = int(total * config.data.val_ratio)
    test_count = total - train_count - val_count

    train_examples = shuffled[:train_count]
    val_examples = shuffled[train_count : train_count + val_count]
    test_examples = shuffled[train_count + val_count : train_count + val_count + test_count]
    return train_examples, val_examples, test_examples


def save_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> None:
    """Write records to disk in JSONL format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def export_dataset_summary(
    all_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    val_examples: list[dict[str, object]],
    test_examples: list[dict[str, object]],
    source_vocab_size: int,
    target_vocab_size: int,
    config: ProjectConfig,
) -> dict[str, object]:
    """Create a report-friendly dataset summary dictionary."""

    def sample_records(records: list[dict[str, object]], count: int = 3) -> list[dict[str, object]]:
        return records[: min(count, len(records))]

    return {
        "project_name": config.project_name,
        "total_examples": len(all_examples),
        "split_counts": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        },
        "schema": {
            "effect_types": list(config.data.effect_types),
            "velocity_levels": list(config.data.velocity_levels),
            "directions": list(config.data.directions),
            "spread_levels": list(config.data.spread_levels),
            "drift_values": list(config.data.drift_values),
            "burst_values": list(config.data.burst_values),
            "flicker_values": list(config.data.flicker_values),
            "density_values": list(config.data.density_values),
        },
        "paraphrases_per_combination": config.data.paraphrases_per_combination,
        "special_tokens": list(SPECIAL_TOKENS),
        "vocabulary_sizes": {
            "source": source_vocab_size,
            "target": target_vocab_size,
        },
        "sample_examples": {
            "train": sample_records(train_examples),
            "validation": sample_records(val_examples),
            "test": sample_records(test_examples),
        },
    }


def save_dataset_summary(summary: dict[str, object], path: str | Path) -> None:
    """Write the dataset summary dictionary to disk as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_and_save_dataset(config: ProjectConfig) -> dict[str, object]:
    """Generate the dataset, vocabularies, and summary files on disk."""

    all_examples = generate_examples(config)
    train_examples, val_examples, test_examples = split_examples(all_examples, config)
    source_vocab, target_vocab = build_vocabularies_from_records(train_examples)

    save_jsonl(all_examples, config.paths.all_file)
    save_jsonl(train_examples, config.paths.train_file)
    save_jsonl(val_examples, config.paths.val_file)
    save_jsonl(test_examples, config.paths.test_file)
    source_vocab.save(config.paths.src_vocab_file)
    target_vocab.save(config.paths.tgt_vocab_file)

    summary = export_dataset_summary(
        all_examples=all_examples,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        config=config,
    )
    save_dataset_summary(summary, config.paths.dataset_stats_file)
    return summary


def main() -> None:
    """Generate dataset artifacts and print a short summary."""

    config = get_default_config()
    summary = build_and_save_dataset(config)
    print(f"Generated {summary['total_examples']} examples.")
    print(
        "Split counts:",
        summary["split_counts"]["train"],
        summary["split_counts"]["validation"],
        summary["split_counts"]["test"],
    )
    print(
        "Vocabulary sizes:",
        summary["vocabulary_sizes"]["source"],
        summary["vocabulary_sizes"]["target"],
    )


if __name__ == "__main__":
    main()
