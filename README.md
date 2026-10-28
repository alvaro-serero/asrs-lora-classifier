# ASRS LoRA Classifier

> Draft. Work in progress (week 1 of 4). Numbers and sections marked TODO are not filled in yet.

Fine-tuning a small open LLM (0.5B to 1.5B) with LoRA to tag NASA ASRS aviation incident reports with their anomaly categories, and comparing it to a prompted frontier model and a TF-IDF baseline.

## Task

- **Input:** the free-text narrative written by the person reporting the incident (`Report 1 | Narrative`).
- **Output:** the set of anomaly labels the report describes (multi-label, 2 to 3 per report on average), e.g. `ATC Issue All Types`, `Conflict Ground Conflict Critical`, `Deviation - Altitude Overshoot`.

This is the same kind of task as the [SIAM 2007 Text Mining Competition](docs/siam2007/), which gave participants only the report text and a label matrix. Here the pipeline starts from the raw database exports instead of the preprocessed contest files.

## Data

- Source: NASA [Aviation Safety Reporting System](https://asrs.arc.nasa.gov/) database exports (CSV).
- 32 half-year files, 2010 H1 to 2025 H2, about 2,800 to 3,000 reports each (~90k total), 126 columns.
- Files are expected in `data/raw/nasa/asrs_<year>_<H1|H2>.csv`. The `data/` folder is not committed.
- TODO: document exactly how the exports were made (query, fields) so they can be reproduced.

### What the raw files need

- **Two header rows.** Row 1 is a group (`Time`, `Events`, `Report 1`...), row 2 is the field (`Date`, `Anomaly`, `Narrative`...). They are merged into one name, e.g. `Events | Anomaly`.
- **Anomaly field.** One string with values separated by `;`. The severity of a ground conflict is exported as a separate value (`Conflict Ground Conflict; Critical`), so `Critical` and `Less Severe` are attached back to the value before them.
- **Missing values.** Reports with no anomaly or no narrative are dropped (12 reports without an anomaly in 2010 H1).
- TODO: duplicate ACNs across files, final label set (values with at least ~200 reports).

### Why only the narrative

The other columns are mostly filled in by ASRS analysts after reading the report. Some of them (`Synopsis`, `Primary Problem`, `Contributing Factors`, `Result`) partly describe the anomaly itself, so using them as input would leak the label. The model only sees what the reporter wrote, which is also what the frontier baseline gets.

`Date` is kept for the split and `ACN` (report ID) for deduplication.

## Planned setup

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
| Frontier model, prompted | label definitions + a few examples, JSON output |
| LoRA model | small open decoder, LoRA on attention, linear head on the last token, sigmoid + BCE |

Main metric: macro-F1 over the final label set, so rare categories count as much as frequent ones. Also micro-F1, per-label F1, and cost and latency per 1,000 reports.

## Results

TODO

## Repo layout

```
data/raw/nasa/          raw ASRS CSV exports (not committed)
docs/siam2007/          SIAM 2007 contest description and scoring software
notebook.ipynb          exploration (loader prototype for now)
src/asrs_lora_classifier/   package code (loader in progress)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
uv sync
```
