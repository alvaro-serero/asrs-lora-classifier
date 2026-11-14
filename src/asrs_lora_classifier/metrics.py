"""Metrics shared by every model (TF-IDF, LoRA, frontier): F1 scores and per-label thresholds."""

import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds(y_true, probs, grid=np.arange(0.05, 0.96, 0.05)):
    """For each label, the threshold that maximises its F1."""
    best = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        scores_j = [f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        best[j] = grid[int(np.argmax(scores_j))]
    return best


def scores(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
    }


def evaluate(probs, y):
    """Scores at a fixed 0.5 threshold and with thresholds tuned per label on the same data."""
    thresholds = tune_thresholds(y, probs)
    res = {
        "at_0.5": scores(y, (probs >= 0.5).astype(int)),
        "tuned": scores(y, (probs >= thresholds).astype(int)),
    }
    return res, thresholds
