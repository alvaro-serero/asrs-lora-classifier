"""TF-IDF + logistic regression baseline, evaluated on the 2023 validation set.

Run with: uv run python -m asrs_lora_classifier.baseline_tfidf
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from asrs_lora_classifier.loader import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"


def load_split(name):
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def tune_thresholds(y_true, probs, grid=np.arange(0.05, 0.96, 0.05)):
    """For each label, the threshold that maximises its F1."""
    best = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        best[j] = grid[int(np.argmax(scores))]
    return best


def scores(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
    }


def main():
    labels = json.loads((DATA_DIR / "labels.json").read_text())
    train, val = load_split("train"), load_split("val")

    mlb = MultiLabelBinarizer(classes=labels)
    y_train = mlb.fit_transform(train["labels"])
    y_val = mlb.transform(val["labels"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train["narrative"])
    x_val = vectorizer.transform(val["narrative"])

    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000), n_jobs=-1)
    start = time.time()
    clf.fit(x_train, y_train)
    print(f"trained in {time.time() - start:.0f}s")

    probs = clf.predict_proba(x_val)
    at_05 = scores(y_val, (probs >= 0.5).astype(int))
    thresholds = tune_thresholds(y_val, probs)
    tuned = scores(y_val, (probs >= thresholds).astype(int))
    per_label = f1_score(y_val, (probs >= thresholds).astype(int), average=None, zero_division=0)

    print("val @0.5 :", {k: round(v, 4) for k, v in at_05.items()})
    print("val tuned:", {k: round(v, 4) for k, v in tuned.items()})

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "model": "tfidf_logreg",
        "split": "val_2023",
        "at_0.5": at_05,
        "tuned_on_val": tuned,
        "per_label_f1": dict(zip(labels, per_label.round(4).tolist())),
        "thresholds": dict(zip(labels, thresholds.round(2).tolist())),
    }
    (RESULTS_DIR / "tfidf_val.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
