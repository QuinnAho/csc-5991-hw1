# CSC 5991 Homework 1

This repository contains a complete PyTorch implementation for `CSC 5991 - Winter 2026`, `Homework 1`, `Option 2: Sequence to Sequence Network and Attention`.

The task is to translate natural-language visual-effect descriptions into canonical parameter token sequences. Example:

- Input: `make the smoke rise slowly and spread wide`
- Output: `effect smoke velocity low direction up spread wide drift false burst false flicker low density low`

## Dataset

The project uses a synthetic, fully reproducible dataset generated from a controlled slot schema:

- effect: `smoke`, `sparks`, `fire`, `mist`
- velocity: `low`, `medium`, `high`
- direction: `up`, `down`
- spread: `wide`, `narrow`, `compact`, `large`
- drift: `false`, `true`
- burst: `false`, `true`
- flicker: `low`, `high`
- density: `low`, `high`

Every structured combination is rendered into four deterministic paraphrases, producing `6144` total examples split into train, validation, and test sets.

## Architecture

The model is a sequence-to-sequence GRU with Bahdanau additive attention:

- GRU encoder over the tokenized source sentence
- Bahdanau attention over encoder outputs
- GRU decoder that predicts the canonical target token sequence
- Teacher forcing during training and greedy decoding during evaluation

Default model dimensions:

- embedding dimension: `64`
- hidden dimension: `128`
- encoder layers: `1`
- decoder layers: `1`
- dropout: `0.1`

## Project Layout

```text
csc-5991-hw1/
  README.md
  requirements.txt
  src/
  data/processed/
  outputs/
    checkpoints/
    metrics/
    plots/
    predictions/
  report/
    report.md
  report_assets/
```

## Training

Create a virtual environment, install dependencies, and run the final training configuration:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --regenerate-data --epochs 25 --batch-size 128 --learning-rate 0.001 --teacher-forcing 0.5 --gradient-clip 1.0 --seed 42
```

Training writes:

- checkpoint: `outputs/checkpoints/best_model.pt`
- history: `outputs/metrics/train_history.json`, `outputs/train_history.json`
- plots: `outputs/plots/training_loss.png`, `outputs/plots/validation_loss.png`, `outputs/plots/loss_curves.png`

## Evaluation

Run held-out evaluation with:

```powershell
python -m src.test --batch-size 128 --num-examples 0
```

Evaluation writes:

- metrics: `outputs/metrics/test_metrics.json`, `outputs/test_metrics.json`
- predictions: `outputs/predictions/test_predictions.csv`, `outputs/test_predictions.csv`

## Report Assets

Generate report-ready markdown and figures with:

```powershell
python -m src.report_assets --num-examples 10
```

This refreshes:

- `outputs/plots/training_loss.png`
- `outputs/plots/loss_curves.png`
- `outputs/plots/test_vs_validation_loss.png`
- `outputs/predictions/test_examples.md`
- `report_assets/dataset_summary.md`
- `report_assets/architecture_summary.md`
- `report_assets/hyperparameters.md`
- `report_assets/attention_snippet.py`

## Final Report

The completed assignment writeup is in `report/report.md`. Supporting assets used by the report are stored in `report_assets/` and `outputs/`.
