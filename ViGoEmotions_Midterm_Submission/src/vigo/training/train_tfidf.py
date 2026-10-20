"""Train M1 - TF-IDF + One-vs-Rest Logistic Regression.

    python -m vigo train-tfidf --config configs/tfidf_lr.yaml

Features: word 1-2-grams on normalised, word-segmented text + char_wb 1-5-grams, sublinear TF
(char n-grams make the model robust to teencode / spelling variation and capture emoji).
C and class_weight are selected on the VALIDATION set (Macro-F1 @ tau = 0.5); test is scored
once with the frozen configuration.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from joblib import parallel_backend
from sklearn.metrics import f1_score

from vigo.data.dataset import load_data
from vigo.data.preprocessing import preprocess_texts
from vigo.evaluation.runs import finalize_run
from vigo.models.tfidf import build_features, make_classifier
from vigo.utils import ensure_dirs, load_config, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vigo train-tfidf")
    ap.add_argument("--config", default="configs/tfidf_lr.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    ensure_dirs()
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    name = cfg.get("run_name", "tfidf_lr")

    data = load_data(args.data_dir, cfg.get("split", "official"))
    texts = {s: preprocess_texts(data.texts(s), cfg["preprocess"]) for s in ("train", "val", "test")}
    other = {"val": texts["val"], "test": texts["test"]}

    t0 = time.perf_counter()
    Xtr, X, vecs = build_features(cfg["features"], texts["train"], other)
    feat_time = time.perf_counter() - t0
    print(f"features: {Xtr.shape[1]:,} (word {len(vecs[0].vocabulary_):,} + char {len(vecs[1].vocabulary_):,})")

    # ---- model selection on validation
    grid = cfg["search"]
    ytr = np.ascontiguousarray(data.y["train"], dtype=np.int64)
    results = []
    for C, cw in itertools.product(grid["C"], grid["class_weight"]):
        clf = make_classifier(C, cw, seed, cfg.get("n_jobs", -1))
        with parallel_backend("loky", max_nbytes=None):  # no read-only memmaps (liblinear needs writable arrays)
            clf.fit(Xtr, ytr)
        pv = clf.predict_proba(X["val"])
        f = f1_score(data.y["val"], (pv >= 0.5).astype(int), average="macro", zero_division=0)
        results.append({"C": C, "class_weight": str(cw), "val_macro_f1@0.5": f})
        print(f"  C={C:<6} class_weight={str(cw):<9} val macro-F1@0.5={f:.4f}")
    best = max(results, key=lambda r: r["val_macro_f1@0.5"])
    best_cw = None if best["class_weight"] == "None" else best["class_weight"]
    print("selected:", best)

    # ---- final fit with frozen hyper-parameters (timed)
    t0 = time.perf_counter()
    Xtr, X, vecs = build_features(cfg["features"], texts["train"], other)
    clf = make_classifier(best["C"], best_cw, seed, cfg.get("n_jobs", -1))
    with parallel_backend("loky", max_nbytes=None):
        clf.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    probs = {s: clf.predict_proba(X[s]).astype(np.float32) for s in ("val", "test")}
    n_feat, n_labels = Xtr.shape[1], data.n_labels
    params = {"total_params": int(n_labels * (n_feat + 1)), "trainable_params": int(n_labels * (n_feat + 1)),
              "n_features": int(n_feat)}
    cfg_out = {**cfg, "selected": {"C": best["C"], "class_weight": best_cw}, "search_results": results}
    finalize_run(name, cfg_out, data.label_names, {s: data.ids(s) for s in ("val", "test")}, data.y, probs, params,
                 train_time, extra={"feature_time_s": feat_time})


if __name__ == "__main__":
    main()
