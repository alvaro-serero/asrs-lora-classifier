"""Build the processed dataset: date splits, frozen label set, baseline sample.

Run with: uv run python -m asrs_lora_classifier.prepare
Writes to data/processed/: train/val/test.parquet, labels.json, baseline_acns.json
"""

import json

from asrs_lora_classifier.loader import REPO_ROOT, load_all

OUT_DIR = REPO_ROOT / "data" / "processed"
COLS = ["acn", "date", "year", "narrative", "labels"]
SEED = 42
MIN_TRAIN = 200
MIN_VAL = 20
BASELINE_SIZE = 1000


def split_by_year(df):
    """Train 2010-2022, val 2023, test 2024+."""
    train = df[df["year"] <= 2022].copy()
    val = df[df["year"] == 2023].copy()
    test = df[df["year"] >= 2024].copy()
    assert len(train) + len(val) + len(test) == len(df), "split lost rows"
    return train, val, test


def build_label_set(train, val, min_train=MIN_TRAIN, min_val=MIN_VAL):
    """Labels frequent enough to learn (train) and to measure (val), sorted."""
    train_counts = train["labels"].explode().value_counts()
    val_counts = val["labels"].explode().value_counts()
    val_aligned = val_counts.reindex(train_counts.index, fill_value=0)

    keep = (train_counts >= min_train) & (val_aligned >= min_val)
    keep &= ~train_counts.index.str.startswith("Other ")
    return sorted(train_counts[keep].index)


def filter_labels(split, labels):
    """Keep only labels in the set; drop reports left with none."""
    keep = set(labels)
    split = split.copy()
    split["labels"] = split["labels"].apply(lambda ls: [l for l in ls if l in keep])
    empty = split["labels"].str.len() == 0
    return split[~empty].reset_index(drop=True), int(empty.sum())


def check(splits, labels, baseline_acns):
    """Guarantees: no empty targets, no unknown labels, no leakage."""
    label_set = set(labels)
    for name, s in splits.items():
        assert (s["labels"].str.len() > 0).all(), f"empty labels in {name}"
        assert s["labels"].explode().isin(label_set).all(), f"unknown label in {name}"
    train, val, test = (set(splits[k]["acn"]) for k in ("train", "val", "test"))
    assert not (train & val), "train/val overlap"
    assert not (train & test), "train/test overlap"
    assert not (val & test), "val/test overlap"
    assert set(baseline_acns) <= test, "baseline not in test"


def main():
    df = load_all()
    train, val, test = split_by_year(df)
    labels = build_label_set(train, val)

    splits = {}
    for name, s in [("train", train), ("val", val), ("test", test)]:
        splits[name], dropped = filter_labels(s, labels)
        print(f"{name}: dropped {dropped} reports with no kept label")

    baseline_acns = sorted(
        splits["test"].sample(n=BASELINE_SIZE, random_state=SEED)["acn"].tolist()
    )
    check(splits, labels, baseline_acns)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        s[COLS].to_parquet(OUT_DIR / f"{name}.parquet", index=False)
    (OUT_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    (OUT_DIR / "baseline_acns.json").write_text(json.dumps(baseline_acns))

    sizes = " | ".join(f"{k} {len(v)}" for k, v in splits.items())
    print(f"{sizes} | {len(labels)} labels -> {OUT_DIR}")


if __name__ == "__main__":
    main()
