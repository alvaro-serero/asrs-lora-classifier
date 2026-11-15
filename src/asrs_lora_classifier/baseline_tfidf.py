"""TF-IDF + logistic regression baseline: thresholds tuned on 2023 val, then scored once on 2024-25 test.

Run with: uv run python -m asrs_lora_classifier.baseline_tfidf
"""

import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from asrs_lora_classifier.data import DATA_DIR, load_baseline_acns, load_split
from asrs_lora_classifier.loader import REPO_ROOT
from asrs_lora_classifier.metrics import scores, test_report, tune_thresholds

RESULTS_DIR = REPO_ROOT / "results"


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

    # Test (2024-25), read once, with the thresholds tuned on val above
    test = load_split("test").reset_index(drop=True)
    y_test = mlb.transform(test["labels"])
    test_probs = clf.predict_proba(vectorizer.transform(test["narrative"]))
    report = test_report(test, y_test, test_probs, thresholds, labels, load_baseline_acns())
    print("test (val thresholds):", {k: round(v, 4) for k, v in report["full_test"]["val_thresholds"].items()})
    (RESULTS_DIR / "tfidf_test.json").write_text(json.dumps({"model": "tfidf_logreg", **report}, indent=2))


if __name__ == "__main__":
    main()
