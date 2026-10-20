"""Probability-averaging ensemble of saved runs (no retraining, no test tuning).

    python -m vigo ensemble --runs phobert visobert visobert14gb --name ensemble_3
    python -m vigo ensemble-select        # validation-only comparison of member compositions

Weights: uniform by default; `--weights val` fits non-negative weights on VALIDATION
macro-F1 (coarse simplex grid) and freezes them before touching test.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from vigo.data.dataset import load_label_names
from vigo.evaluation.metrics import evaluate
from vigo.evaluation.runs import finalize_run, load_probs, load_summary
from vigo.evaluation.thresholds import tune_global
from vigo.utils import TABLES_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vigo ensemble")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--weights", default="uniform", choices=["uniform", "val"])
    args = ap.parse_args(argv)
    name = args.name or "ensemble_" + "+".join(args.runs)
    P = {s: [load_probs(r, s) for r in args.runs] for s in ("val", "test")}
    for s in P:
        ids0 = P[s][0]["ids"]
        assert all((p["ids"] == ids0).all() for p in P[s]), "runs must share the same split/order"
    y = {s: P[s][0]["y_true"] for s in P}

    k = len(args.runs)
    if args.weights == "uniform" or k == 1:
        w = np.full(k, 1 / k)
    else:
        best, w = -1, None
        grid = np.arange(0, 1.01, 0.1)
        for cand in itertools.product(grid, repeat=k):
            if abs(sum(cand) - 1) > 1e-6:
                continue
            pv = sum(c * p["probs"] for c, p in zip(cand, P["val"]))
            f = f1_score(y["val"], (pv >= tune_global(y["val"], pv)).astype(int), average="macro", zero_division=0)
            if f > best:
                best, w = f, np.asarray(cand)
    probs = {s: sum(wi * p["probs"] for wi, p in zip(w, P[s])) for s in P}
    sums = [load_summary(r) for r in args.runs]
    params = {"total_params": int(sum(x.get("total_params", 0) for x in sums)),
              "trainable_params": int(sum(x.get("trainable_params", 0) for x in sums))}
    train_time = float(sum(x.get("train_time_s", 0) for x in sums))
    finalize_run(
        name, {"type": "ensemble", "members": args.runs, "weights": w.tolist(), "weighting": args.weights},
        load_label_names(), {s: P[s][0]["ids"] for s in P}, y, probs, params, train_time,
        extra={"members": args.runs, "weights": w.tolist()},
    )


def select_main(argv=None):
    """Compare every subset of candidate runs on VALIDATION only (test is never read).
    Writes results/tables/ensemble_composition_val.csv."""
    ap = argparse.ArgumentParser(prog="vigo ensemble-select")
    ap.add_argument("--runs", nargs="+", default=["phobert", "visobert", "visobert14gb", "tfidf_lr"])
    args = ap.parse_args(argv)
    names = load_label_names()
    V = {r: load_probs(r, "val") for r in args.runs}
    y = V[args.runs[0]]["y_true"]
    rows = []
    for k in range(1, len(args.runs) + 1):
        for combo in itertools.combinations(args.runs, k):
            p = np.mean([V[r]["probs"] for r in combo], 0)
            t = tune_global(y, p)
            rows.append({"members": "+".join(combo), "n": k, "val_macro_f1@0.5": evaluate(y, p, 0.5, names)["macro_f1"],
                         "val_global_tau": t, "val_macro_f1@tau*": evaluate(y, p, t, names)["macro_f1"]})
    df = pd.DataFrame(rows).sort_values("val_macro_f1@tau*", ascending=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "ensemble_composition_val.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
