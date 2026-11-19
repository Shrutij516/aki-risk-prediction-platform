"""Evaluation metrics, validation-only threshold selection, and bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

# Same grid as the notebooks (np.arange(0.05, 0.96, 0.01)).
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def select_threshold(y_val: np.ndarray, p_val: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> tuple[float, float]:
    """Threshold maximising F1 on VALIDATION data (first maximum, as in the notebooks)."""
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def threshold_at_fpr(y_val: np.ndarray, p_val: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose validation FPR is <= target_fpr."""
    neg = np.sort(p_val[y_val == 0])[::-1]
    k = int(np.floor(target_fpr * len(neg)))
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else float(neg[-1])


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y)), "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 0,
                 keys: tuple[str, ...] = ("roc_auc", "pr_auc", "f1", "sensitivity", "specificity")) -> dict:
    """Percentile 95% intervals from patient-level bootstrap resamples of the test set."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(idx)):
            continue
        m = compute_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}
