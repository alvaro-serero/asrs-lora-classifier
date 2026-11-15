# ASRS LoRA Classifier

Fine-tuning a small open LLM (Qwen3-0.6B) with LoRA to tag NASA ASRS aviation incident reports with their anomaly categories, and comparing it to a prompted frontier model and a TF-IDF baseline.

## Task

- **Input:** the free-text narrative written by the person reporting the incident (`Report 1 | Narrative`).
- **Output:** the set of anomaly labels the report describes (multi-label, 2 to 3 per report on average), e.g. `ATC Issue All Types`, `Conflict Ground Conflict Critical`, `Deviation - Altitude Overshoot`.

This is the same kind of task as the [SIAM 2007 Text Mining Competition](docs/siam2007/), which gave participants only the report text and a label matrix. Here the pipeline starts from the raw database exports instead of the preprocessed contest files.

## Data

- Source: NASA [Aviation Safety Reporting System](https://asrs.arc.nasa.gov/) database exports (CSV).
- 32 half-year files, 2010 H1 to 2025 H2, about 2,800 to 3,000 reports each (~90k total), 126 columns.
- Files are expected in `data/raw/nasa/asrs_<year>_<H1|H2>.csv`. The `data/` folder is not committed.

### What the raw files need

- **Two header rows.** Row 1 is a group (`Time`, `Events`, `Report 1`...), row 2 is the field (`Date`, `Anomaly`, `Narrative`...). They are merged into one name, e.g. `Events | Anomaly`.
- **Anomaly field.** One string with values separated by `;`. The severity of a ground conflict is exported as a separate value (`Conflict Ground Conflict; Critical`), so `Critical` and `Less Severe` are attached back to the value before them.
- **Missing values.** Reports with no anomaly or no narrative are dropped (12 reports without an anomaly in 2010 H1).
- **Duplicates.** The loader checks that no report ID (ACN) appears in two files.
- **Label set.** 46 labels: values with at least 200 train reports and 20 validation reports, minus the catch-all `Other ...` values. Reports left with no label are dropped.

### Why only the narrative

The other columns are mostly filled in by ASRS analysts after reading the report. Some of them (`Synopsis`, `Primary Problem`, `Contributing Factors`, `Result`) partly describe the anomaly itself, so using them as input would leak the label. The model only sees what the reporter wrote, which is also what the frontier baseline gets.

`Date` is kept for the split and `ACN` (report ID) for deduplication.

## Setup

Split by date so the test set is newer than anything seen in training:

| Split | Period | Use |
| --- | --- | --- |
| Train | 2010 to 2022 | training |
| Validation | 2023 | early stopping, per-label thresholds |
| Test | 2024 to 2025 | final numbers, used once |

Contenders:

| Model | Description |
| --- | --- |
| TF-IDF + logistic regression | one-vs-rest baseline |
| Frontier model, prompted | Claude Opus 5.5, label list + 5 train examples, JSON output, no training |
| LoRA model | Qwen3-0.6B-Base, LoRA r=8 on attention, linear head on the last token, sigmoid + BCE, 1 epoch on a T4 |

TF-IDF and LoRA output probabilities, so each label gets its own threshold, tuned on validation. The frontier model outputs labels directly.

Main metric: macro-F1 over the 46 labels, so rare categories count as much as frequent ones. Also micro-F1, per-label F1, and cost and latency per 1,000 reports.

## Results

Test set, 2024 to 2025. The frontier model is too expensive to run on all 11,542 test reports, so all three models are compared on the same fixed random sample of 1,000 test reports.

| Model | Macro-F1 | Micro-F1 | Exact match | Cost per 1,000 reports |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.502 | 0.617 | 0.078 | ~0 (CPU) |
| Qwen3-0.6B + LoRA r=8 | 0.536 | **0.637** | 0.090 | ~0 (one T4 GPU) |
| Claude Opus 5.5, 5-shot | **0.565** | 0.632 | **0.242** | $3.75 |

On the full test set (11,542 reports): TF-IDF 0.523 / 0.627, LoRA 0.545 / 0.639 (macro / micro-F1).

What the numbers say:

- LoRA beats TF-IDF by about 3 points of macro-F1 and 2 of micro-F1.
- Claude gets the best macro-F1 and three times the exact-match rate, but LoRA matches it on micro-F1 (the frequent labels) at almost no cost per report.
- The models fail on different labels. Claude does well on rare ground events but scores 0.06 on `Deviation - Altitude Overshoot` (LoRA 0.55): from the narrative alone it does not follow the way ASRS analysts apply that label. A fine-tuned model learns the annotation conventions, a prompted one only has the label names.
- Run 2 (LoRA r=16 on attention + MLP) is in progress.

Raw scores, per-label F1 and thresholds are in [`results/`](results/).

## Repo layout

```
configs/                    training configs (debug on a laptop, full runs on a Kaggle T4)
data/raw/nasa/              raw ASRS CSV exports (not committed)
data/processed/             splits and label set built by prepare.py (not committed)
docs/siam2007/              SIAM 2007 contest description and scoring software
notebook.ipynb              exploration, prototypes and error analysis
results/                    scores, per-label F1 and thresholds for every model
src/asrs_lora_classifier/
    loader.py               reads and cleans the raw exports
    prepare.py              date split, label set, 1,000-report test sample
    baseline_tfidf.py       TF-IDF + logistic regression baseline
    model.py                Qwen3 + LoRA + classification head
    train.py                training loop (PyTorch, W&B logging)
    evaluate.py             scores a checkpoint on the test set
    frontier.py             prompted Claude baseline
    metrics.py, data.py     shared helpers
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
uv sync
```

Then, from the repo root:

```
uv run python -m asrs_lora_classifier.prepare           # build splits from data/raw/nasa/
uv run python -m asrs_lora_classifier.baseline_tfidf
uv run python -m asrs_lora_classifier.train configs/kaggle.yaml
uv run python -m asrs_lora_classifier.evaluate configs/kaggle.yaml
uv run python -m asrs_lora_classifier.frontier          # needs ANTHROPIC_API_KEY
```
