"""M1 - TF-IDF features (word 1-2-grams + char_wb 1-5-grams) and One-vs-Rest Logistic Regression."""
from __future__ import annotations

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def build_features(cfg: dict, train_texts: list[str], other_texts: dict):
    """Fit both vectorisers on TRAIN only; return (X_train, {split: X}, (word_vec, char_vec))."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)[^\s]+",
        ngram_range=tuple(cfg["word_ngram"]),
        min_df=cfg["min_df"],
        max_df=cfg.get("max_df", 1.0),
        sublinear_tf=True,
        max_features=cfg.get("word_max_features"),
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(cfg["char_ngram"]),
        min_df=cfg["min_df"],
        sublinear_tf=True,
        max_features=cfg.get("char_max_features"),
    )
    Xtr = sp.hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)]).tocsr()
    others = {k: sp.hstack([word_vec.transform(v), char_vec.transform(v)]).tocsr() for k, v in other_texts.items()}
    return Xtr, others, (word_vec, char_vec)


def make_classifier(C: float, class_weight, seed: int, n_jobs: int) -> OneVsRestClassifier:
    """28 independent L2-regularised logistic regressions."""
    return OneVsRestClassifier(
        LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=2000, random_state=seed),
        n_jobs=n_jobs,
    )
