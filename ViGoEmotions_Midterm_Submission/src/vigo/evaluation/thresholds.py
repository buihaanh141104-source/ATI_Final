"""Threshold decoding strategies (fitted on VALIDATION probabilities only, then frozen).

  fixed      tau = 0.5 for every label
  global     tau* = argmax_tau MacroF1(D_val; tau)     (one scalar)
  per_label  tau*_c = argmax_tau F1_c(D_val; tau)      (one per label)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from vigo.evaluation.metrics import f1_per_label_fast

DEFAULT_GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def sweep_global(y_val, p_val, grid=DEFAULT_GRID) -> dict:
    y_val = np.asarray(y_val).astype(np.int8)
    macro, micro = [], []
    for t in grid:
        pred = (p_val >= t).astype(np.int8)
        macro.append(f1_score(y_val, pred, average="macro", zero_division=0))
        micro.append(f1_score(y_val, pred, average="micro", zero_division=0))
    return {"grid": np.asarray(grid), "macro_f1": np.asarray(macro), "micro_f1": np.asarray(micro)}


def tune_global(y_val, p_val, grid=DEFAULT_GRID, metric: str = "macro_f1") -> float:
    sw = sweep_global(y_val, p_val, grid)
    scores = sw[metric]
    best = np.flatnonzero(scores == scores.max())
    # tie-break: the candidate closest to 0.5
    return float(sw["grid"][best[np.argmin(np.abs(sw["grid"][best] - 0.5))]])


def tune_per_label(y_val, p_val, grid=DEFAULT_GRID) -> np.ndarray:
    y_val = np.asarray(y_val).astype(np.int8)
    C = y_val.shape[1]
    scores = np.zeros((len(grid), C))
    for i, t in enumerate(grid):
        scores[i] = f1_per_label_fast(y_val, (p_val >= t).astype(np.int8))
    taus = np.zeros(C)
    for c in range(C):
        best = np.flatnonzero(scores[:, c] == scores[:, c].max())
        taus[c] = grid[best[np.argmin(np.abs(np.asarray(grid)[best] - 0.5))]]
    return taus


def fit_all(y_val, p_val, grid=DEFAULT_GRID) -> dict:
    C = y_val.shape[1]
    return {
        "fixed": np.full(C, 0.5),
        "global": np.full(C, tune_global(y_val, p_val, grid)),
        "per_label": tune_per_label(y_val, p_val, grid),
    }
