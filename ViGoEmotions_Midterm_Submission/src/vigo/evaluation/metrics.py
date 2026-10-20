"""Common multi-label evaluation used by every model (same split, same code, same format)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_recall_fscore_support,
)


def binarize(probs: np.ndarray, thresholds) -> np.ndarray:
    thr = np.asarray(thresholds, dtype=float)
    return (probs >= thr).astype(np.int8)


def evaluate(y_true: np.ndarray, probs: np.ndarray, thresholds=0.5, label_names=None) -> dict:
    """All metrics required by the brief.

    Primary: Macro-F1 and Micro-F1 (over all C labels incl. neutral, as in the paper).
    Also: macro-F1 over the 27 emotions only, weighted/sample F1, Hamming loss,
    LRAP, macro mAP, subset accuracy (reported but not a headline metric).
    Threshold-free metrics (LRAP, mAP) use the continuous probabilities.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = binarize(probs, thresholds)
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "lrap": label_ranking_average_precision_score(y_true, probs),
        "map_macro": average_precision_score(y_true, probs, average="macro"),
        "map_micro": average_precision_score(y_true, probs, average="micro"),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
    }
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out["macro_precision"], out["macro_recall"] = p, r
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    out["micro_precision"], out["micro_recall"] = p, r
    if label_names is not None and "neutral" in label_names:
        emo = [i for i, n in enumerate(label_names) if n != "neutral"]
        out["macro_f1_27emo"] = f1_score(y_true[:, emo], y_pred[:, emo], average="macro", zero_division=0)
    out["avg_pred_cardinality"] = float(y_pred.sum(1).mean())
    out["empty_prediction_rate"] = float((y_pred.sum(1) == 0).mean())
    return {k: float(v) for k, v in out.items()}


def per_label_report(y_true, probs, thresholds, label_names) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = binarize(probs, thresholds)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    ap = []
    for c in range(y_true.shape[1]):
        ap.append(average_precision_score(y_true[:, c], probs[:, c]) if y_true[:, c].any() else np.nan)
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    return pd.DataFrame(
        {
            "label": label_names,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s,
            "n_pred": y_pred.sum(0),
            "ap": ap,
            "threshold": thr,
        }
    )


def f1_per_label_fast(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tp = (y_true & y_pred).sum(0).astype(float)
    fp = ((1 - y_true) & y_pred).sum(0).astype(float)
    fn = (y_true & (1 - y_pred)).sum(0).astype(float)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def paired_bootstrap_delta(
    y_true, pred_a, pred_b, metric: str = "macro", n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap 95% CI of F1(b) - F1(a) on the same samples (used for Q4 subsets)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = f1_score(y_true[idx], pred_a[idx], average=metric, zero_division=0)
        fb = f1_score(y_true[idx], pred_b[idx], average=metric, zero_division=0)
        deltas.append(fb - fa)
    deltas = np.asarray(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
