"""WP4-WP6 - common evaluation tables, the five Midterm questions and error analysis.

    python -m vigo analyze                                   # defaults below
    python -m vigo analyze --tfidf tfidf_lr --rnn bilstm --transformer phobert \
        --extra visobert visobert14gb ensemble_3 --error_model phobert

Everything is computed from the saved probabilities in results/runs/*/{val,test}.npz.
Thresholds other than 0.5 are fitted on VALIDATION and frozen before scoring test.
Public tables never contain raw comment text; the text-bearing audit file is written to
results/private/ (gitignored) for the team's manual review only.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score

from vigo.data.dataset import frequency_tiers, load_data
from vigo.evaluation.metrics import binarize, evaluate, paired_bootstrap_delta, per_label_report
from vigo.evaluation.runs import load_probs, load_summary, run_dir
from vigo.analysis.text_styles import RULES, style_flags
from vigo.evaluation.thresholds import DEFAULT_GRID, fit_all, sweep_global
from vigo.utils import FIGURES_DIR, PRED_DIR, PRIVATE_DIR, TABLES_DIR, ensure_dirs, save_json

DISPLAY = {
    "tfidf_lr": "TF-IDF + OvR LR",
    "bilstm": "BiLSTM",
    "bigru": "BiGRU",
    "phobert": "PhoBERT-base-v2",
    "phobert_large": "PhoBERT-large",
    "visobert": "ViSoBERT",
    "visobert14gb": "ViSoBERT-14GB",
    "cafebert": "CafeBERT",
    "xlmr": "XLM-R-base",
}
# Test-set numbers reported by the ViGoEmotions authors (official notebooks, tau=0.5,
# BCE with pos_weight, same official split) - for context only.
PAPER = [
    {"model": "PhoBERT-base-v2 (paper, emoji kept)", "macro_f1": 0.5633, "micro_f1": 0.5740},
    {"model": "PhoBERT-base-v2 (paper, emoji->text)", "macro_f1": 0.5931, "micro_f1": 0.6042},
    {"model": "CafeBERT (paper)", "macro_f1": 0.6117, "micro_f1": 0.6247},
    {"model": "ViSoBERT (paper, best)", "macro_f1": 0.6150, "micro_f1": 0.6346},
]
MAIN_COLS = ["macro_f1", "micro_f1", "weighted_f1", "samples_f1", "macro_f1_27emo", "hamming_loss", "lrap",
             "map_macro", "subset_accuracy", "macro_precision", "macro_recall"]


def dname(run: str) -> str:
    if run in DISPLAY:
        return DISPLAY[run]
    if run.startswith("ensemble"):
        return "Ensemble (" + run.replace("ensemble_", "") + ")"
    return run


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=160)
    plt.close(fig)


def _md(df: pd.DataFrame, path, floatfmt=".4f"):
    try:
        path.write_text(df.to_markdown(index=False, floatfmt=floatfmt), encoding="utf-8")
    except ImportError:  # tabulate not installed
        path.write_text(df.to_string(index=False), encoding="utf-8")


# ----------------------------------------------------------------------------- loading
def load_runs(runs: list[str]) -> dict:
    out = {}
    for r in runs:
        if not (run_dir(r) / "summary.json").exists():
            print(f"[skip] run '{r}' not found")
            continue
        out[r] = {"val": load_probs(r, "val"), "test": load_probs(r, "test"), "summary": load_summary(r)}
    return out


# ----------------------------------------------------------------------------- tables
def main_tables(R, label_names):
    rows_t, rows_v, rows_thr = [], [], []
    for r, d in R.items():
        s = d["summary"]
        base = {"model": dname(r), "run": r, "params": s.get("total_params"),
                "train_time_s": round(s.get("train_time_s", 0), 1), "device": s.get("gpu") or s.get("device", "cpu")}
        mt = evaluate(d["test"]["y_true"], d["test"]["probs"], 0.5, label_names)
        mv = evaluate(d["val"]["y_true"], d["val"]["probs"], 0.5, label_names)
        rows_t.append({**base, **{k: mt[k] for k in MAIN_COLS}})
        rows_v.append({**base, **{k: mv[k] for k in MAIN_COLS}})
        th = fit_all(d["val"]["y_true"], d["val"]["probs"])
        d["thresholds"] = th
        for strat, tau in th.items():
            m_te = evaluate(d["test"]["y_true"], d["test"]["probs"], tau, label_names)
            m_va = evaluate(d["val"]["y_true"], d["val"]["probs"], tau, label_names)
            rows_thr.append({"model": dname(r), "run": r, "strategy": strat,
                             "tau": "0.50" if strat == "fixed" else (f"{tau[0]:.2f}" if strat == "global" else "per-label"),
                             "val_macro_f1": m_va["macro_f1"], "test_macro_f1": m_te["macro_f1"],
                             "val_micro_f1": m_va["micro_f1"], "test_micro_f1": m_te["micro_f1"],
                             "test_hamming_loss": m_te["hamming_loss"],
                             "gap_macro_f1(val-test)": m_va["macro_f1"] - m_te["macro_f1"]})
    t, v, thr = pd.DataFrame(rows_t), pd.DataFrame(rows_v), pd.DataFrame(rows_thr)
    t.to_csv(TABLES_DIR / "main_results_test.csv", index=False)
    v.to_csv(TABLES_DIR / "main_results_val.csv", index=False)
    thr.to_csv(TABLES_DIR / "threshold_strategies.csv", index=False)
    show = t[["model", "macro_f1", "micro_f1", "hamming_loss", "lrap", "map_macro", "params", "train_time_s"]]
    _md(show, TABLES_DIR / "main_results_test.md")
    _md(thr.drop(columns=["run"]), TABLES_DIR / "threshold_strategies.md")

    # context: paper numbers vs ours
    best = thr[thr.strategy.isin(["fixed", "per_label"])].copy()
    comp = [{"model": p["model"], "decoding": "tau=0.5, pos_weight BCE", "test_macro_f1": p["macro_f1"],
             "test_micro_f1": p["micro_f1"], "source": "Hung et al. (EACL 2026) official notebooks"} for p in PAPER]
    for _, rw in best.iterrows():
        comp.append({"model": rw["model"] + " (ours)",
                     "decoding": "tau=0.5" if rw["strategy"] == "fixed" else "per-label tau (val-tuned)",
                     "test_macro_f1": rw["test_macro_f1"], "test_micro_f1": rw["test_micro_f1"], "source": "this work"})
    comp = pd.DataFrame(comp)
    comp.to_csv(TABLES_DIR / "comparison_with_paper.csv", index=False)
    _md(comp, TABLES_DIR / "comparison_with_paper.md")
    return t, v, thr


def per_label_tables(R, data, tiers):
    names = data.label_names
    train_counts = data.y["train"].sum(0)
    df = pd.DataFrame({"label": names, "train_frequency": train_counts, "tier": [tiers[n] for n in names],
                       "test_support": data.y["test"].sum(0)})
    long_rows = []
    for r, d in R.items():
        rep = per_label_report(d["test"]["y_true"], d["test"]["probs"], 0.5, names)
        df[f"f1_{r}"] = rep["f1"].round(4)
        rep.insert(0, "model", dname(r))
        long_rows.append(rep)
    df.to_csv(TABLES_DIR / "per_label_f1.csv", index=False)
    pd.concat(long_rows).to_csv(TABLES_DIR / "per_label_report_long.csv", index=False)
    return df


# ----------------------------------------------------------------------------- Q1 / Q2
def q1_q2(per_label: pd.DataFrame, R, order_run: str):
    runs = list(R)
    # Q1: highest / lowest
    rows = []
    for r in runs:
        s = per_label.sort_values(f"f1_{r}", ascending=False)
        for rank, (_, rw) in enumerate(s.iterrows(), 1):
            if rank <= 5 or rank > len(s) - 5:
                rows.append({"model": dname(r), "position": "top-5" if rank <= 5 else "bottom-5", "rank": rank,
                             "label": rw["label"], "f1": rw[f"f1_{r}"], "train_frequency": rw["train_frequency"],
                             "tier": rw["tier"]})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "q1_best_worst_labels.csv", index=False)

    s = per_label.sort_values(f"f1_{order_run}")
    fig, ax = plt.subplots(figsize=(8, 10))
    yy = np.arange(len(s))
    h = 0.8 / len(runs)
    for i, r in enumerate(runs):
        ax.barh(yy + i * h - 0.4 + h / 2, s[f"f1_{r}"], height=h, label=dname(r))
    ax.set_yticks(yy)
    ax.set_yticklabels(s["label"])
    ax.set_xlabel("test F1 (tau = 0.5)")
    ax.set_title(f"Per-label F1 (sorted by {dname(order_run)})")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "q1_per_label_f1")

    # slide-friendly wide variant (labels on the x axis, sorted by the transformer's F1)
    s = per_label.sort_values(f"f1_{order_run}", ascending=False)
    fig, ax = plt.subplots(figsize=(16, 5.2))
    xx = np.arange(len(s))
    w = 0.8 / len(runs)
    for i, r in enumerate(runs):
        ax.bar(xx + i * w - 0.4 + w / 2, s[f"f1_{r}"], width=w, label=dname(r))
    ax.set_xticks(xx)
    ax.set_xticklabels(s["label"], rotation=60, ha="right", fontsize=11)
    for lab, tier in zip(ax.get_xticklabels(), s["tier"]):
        lab.set_color({"head": "#1B4FA8", "mid": "#8A5A12", "tail": "#B02A3A"}[tier])
    ax.set_ylabel("test F1 (tau = 0.5)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=len(runs), fontsize=11, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "q1_per_label_f1_wide")

    # Q2: frequency vs F1
    rows = []
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        x, y = per_label["train_frequency"].to_numpy(), per_label[f"f1_{r}"].to_numpy()
        pr, pp = stats.pearsonr(x, y)
        sr, sp_ = stats.spearmanr(x, y)
        lr_, lp = stats.pearsonr(np.log(x), y)
        rows.append({"model": dname(r), "pearson_r": pr, "pearson_p": pp, "pearson_r_logfreq": lr_,
                     "pearson_p_logfreq": lp, "spearman_rho": sr, "spearman_p": sp_,
                     **{f"mean_f1_{t}": per_label.loc[per_label.tier == t, f"f1_{r}"].mean() for t in ("head", "mid", "tail")}})
        colors = per_label["tier"].map({"head": "#2a6fdb", "mid": "#f2a541", "tail": "#d64550"})
        ax.scatter(x, y, c=colors, s=28)
        for xi, yi, lab in zip(x, y, per_label["label"]):
            ax.annotate(lab, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
        b = np.polyfit(np.log(x), y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(b, np.log(xs)), "k--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{dname(r)}\nSpearman rho={sr:.2f} (p={sp_:.3f})", fontsize=9)
        ax.set_xlabel("training positives (log)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test F1 (tau = 0.5)")
    _save(fig, "q2_frequency_vs_f1")
    q2 = pd.DataFrame(rows)
    q2.to_csv(TABLES_DIR / "q2_frequency_correlation.csv", index=False)
    return q2


# ----------------------------------------------------------------------------- Q3
def q3_cooccurrence(R, data, top_k: int = 15):
    names = data.label_names
    ytr = data.y["train"].astype(int)
    co_tr = ytr.T @ ytr
    C = len(names)
    pairs = sorted(((co_tr[i, j], i, j) for i in range(C) for j in range(i + 1, C)), reverse=True)[:top_k]
    yte = data.y["test"].astype(int)
    rows = []
    iu = np.triu_indices(C, 1)
    true_co = (yte.T @ yte)[iu]
    glob = []
    for r, d in R.items():
        pred = binarize(d["test"]["probs"], 0.5).astype(int)
        pco = pred.T @ pred
        both = yte.T @ yte
        for cnt, i, j in pairs:
            t_mask = (yte[:, i] == 1) & (yte[:, j] == 1)
            rows.append({
                "model": dname(r), "pair": f"{names[i]} + {names[j]}", "train_count": int(cnt),
                "test_true_count": int(both[i, j]), "test_pred_count": int(pco[i, j]),
                "pred/true": round(pco[i, j] / max(both[i, j], 1), 3),
                "pair_recall": round(float((pred[t_mask, i] & pred[t_mask, j]).mean()) if t_mask.any() else np.nan, 3),
            })
        # global dependency preservation: correlation between true & predicted co-occurrence / phi
        with np.errstate(invalid="ignore", divide="ignore"):
            phi_true = np.nan_to_num(np.corrcoef(yte.T))[iu]
            phi_pred = np.nan_to_num(np.corrcoef(pred.T))[iu]
        glob.append({"model": dname(r),
                     "pearson(true_cooc, pred_cooc)": stats.pearsonr(true_co, pco[iu])[0],
                     "pearson(true_phi, pred_phi)": stats.pearsonr(phi_true, phi_pred)[0],
                     "spearman(true_phi, pred_phi)": stats.spearmanr(phi_true, phi_pred)[0],
                     "total_true_pairs": int(true_co.sum()), "total_pred_pairs": int(pco[iu].sum()),
                     "mean_true_cardinality": yte.sum(1).mean(), "mean_pred_cardinality": pred.sum(1).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "q3_top_pairs_true_vs_pred.csv", index=False)
    g = pd.DataFrame(glob)
    g.to_csv(TABLES_DIR / "q3_dependency_preservation.csv", index=False)

    # figure: grouped bars for top pairs
    piv = df.pivot_table(index="pair", columns="model", values="test_pred_count", sort=False)
    truth = df.drop_duplicates("pair").set_index("pair")["test_true_count"]
    piv.insert(0, "Ground truth (test)", truth)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    piv.plot.bar(ax=ax, width=0.85)
    ax.set_ylabel("# test comments with both labels")
    ax.set_title(f"Top-{top_k} training co-occurring pairs: true vs predicted co-occurrence on test (tau = 0.5)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "q3_cooccurrence_true_vs_pred")
    return df, g


# ----------------------------------------------------------------------------- Q4
def q4_text_style(R, data, rnn: str, transformer: str, tfidf: str | None):
    flags = style_flags(data.texts("test"))
    y = data.y["test"]
    rows = []
    preds = {r: binarize(R[r]["test"]["probs"], 0.5) for r in R}
    for subset in ["all"] + list(RULES):
        mask = np.ones(len(y), bool) if subset == "all" else flags[subset].to_numpy()
        if mask.sum() < 20:
            continue
        row = {"subset": subset, "rule": RULES.get(subset, "all test comments"), "n_samples": int(mask.sum())}
        for r in R:
            row[f"macro_f1_{r}"] = f1_score(y[mask], preds[r][mask], average="macro", zero_division=0)
            row[f"micro_f1_{r}"] = f1_score(y[mask], preds[r][mask], average="micro", zero_division=0)
        if rnn in R and transformer in R:
            row["delta_macro_f1"] = row[f"macro_f1_{transformer}"] - row[f"macro_f1_{rnn}"]
            row["delta_micro_f1"] = row[f"micro_f1_{transformer}"] - row[f"micro_f1_{rnn}"]
            _, lo, hi = paired_bootstrap_delta(y[mask], preds[rnn][mask], preds[transformer][mask], "micro", n_boot=500)
            row["delta_micro_f1_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "q4_text_style_breakdown.csv", index=False)
    if rnn in R and transformer in R:
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = np.arange(len(df))
        runs = [x for x in (tfidf, rnn, transformer) if x in R]
        w = 0.8 / len(runs)
        for i, r in enumerate(runs):
            ax.bar(xs + (i - (len(runs) - 1) / 2) * w, df[f"micro_f1_{r}"], w, label=dname(r))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(df.subset, df.n_samples)], fontsize=8)
        ax.set_ylabel("test micro-F1 (tau = 0.5)")
        ax.set_title("Performance by text style (test)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, "q4_text_style")
    return df


# ----------------------------------------------------------------------------- Q5
def q5_thresholds(R, label_names, plot_runs=None):
    grid = DEFAULT_GRID
    rows, fig_data = [], {}
    for r, d in R.items():
        sw = sweep_global(d["val"]["y_true"], d["val"]["probs"], grid)
        fig_data[r] = sw
        i05 = int(np.argmin(np.abs(grid - 0.5)))
        ib = int(np.argmax(sw["macro_f1"]))
        # local sensitivity: F1 change when tau moves 0.5 +- 0.1 / 0.2
        def at(t):
            return sw["macro_f1"][int(np.argmin(np.abs(grid - t)))]
        rows.append({"model": dname(r), "val_macro_f1@0.5": sw["macro_f1"][i05], "val_micro_f1@0.5": sw["micro_f1"][i05],
                     "best_global_tau(val)": grid[ib], "val_macro_f1@best": sw["macro_f1"][ib],
                     "gain_vs_0.5": sw["macro_f1"][ib] - sw["macro_f1"][i05],
                     "val_macro_f1@0.3": at(0.3), "val_macro_f1@0.4": at(0.4), "val_macro_f1@0.6": at(0.6),
                     "val_macro_f1@0.7": at(0.7),
                     "range_macro_f1_tau_in_[0.3,0.7]": float(np.ptp(sw["macro_f1"][(grid >= 0.3) & (grid <= 0.7)]))})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "q5_threshold_sensitivity_summary.csv", index=False)
    sweep = pd.concat([pd.DataFrame({"model": dname(r), "tau": s["grid"], "val_macro_f1": s["macro_f1"],
                                     "val_micro_f1": s["micro_f1"]}) for r, s in fig_data.items()])
    sweep.to_csv(TABLES_DIR / "q5_threshold_sweep_val.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for r, s in fig_data.items():
        if plot_runs and r not in plot_runs:
            continue
        for ax, key in zip(axes, ("macro_f1", "micro_f1")):
            ax.plot(s["grid"], s[key], label=dname(r))
    for ax, key in zip(axes, ("Macro-F1", "Micro-F1")):
        ax.axvline(0.5, color="k", ls=":", lw=1)
        ax.set_xlabel("global decision threshold tau")
        ax.set_title(f"Validation {key} vs threshold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1 (validation)")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, "q5_threshold_sensitivity")

    # per-label threshold profile (val-tuned)
    prof = pd.DataFrame({"label": label_names})
    for r, d in R.items():
        prof[f"tau_{r}"] = d["thresholds"]["per_label"]
    prof.to_csv(TABLES_DIR / "q5_per_label_thresholds.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for r in R:
        if plot_runs and r not in plot_runs:
            continue
        ax.plot(label_names, prof[f"tau_{r}"], marker="o", ms=3, label=dname(r))
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylabel("val-optimal tau_c")
    ax.set_title("Per-label optimal thresholds (validation)")
    ax.tick_params(axis="x", rotation=70)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "q5_per_label_thresholds")
    return df


# ----------------------------------------------------------------------------- predictions CSV
def export_predictions(R):
    for r, d in R.items():
        th = d["thresholds"]
        for split in ("val", "test"):
            p = d[split]
            y, pr = p["y_true"], p["probs"]
            vec = lambda a: [json.dumps([int(x) for x in row]) for row in a]  # noqa: E731
            df = pd.DataFrame({
                "sample_id": p["ids"],
                "true_label_vector": vec(y),
                "probability_vector": [json.dumps([round(float(x), 5) for x in row]) for row in pr],
                "pred_fixed": vec(binarize(pr, th["fixed"])),
                "pred_global": vec(binarize(pr, th["global"])),
                "pred_per_label": vec(binarize(pr, th["per_label"])),
                "label_cardinality": y.sum(1),
            })
            df.to_csv(PRED_DIR / f"{r}_{split}.csv", index=False)
            np.savez_compressed(PRED_DIR / f"{r}_{split}.npz", sample_id=p["ids"], y_true=y, probs=pr,
                                tau_global=th["global"], tau_per_label=th["per_label"])
        save_json({k: np.asarray(v).tolist() for k, v in th.items()}, PRED_DIR / f"{r}_thresholds.json")


# ----------------------------------------------------------------------------- error analysis
def error_analysis(R, data, run: str, n_cases: int = 25):
    names = np.asarray(data.label_names)
    d = R[run]["test"]
    y, pr = d["y_true"].astype(int), d["probs"]
    pred = binarize(pr, 0.5).astype(int)
    fn, fp = (y == 1) & (pred == 0), (y == 0) & (pred == 1)
    flags = style_flags(data.texts("test"))
    card = y.sum(1)

    lab = pd.DataFrame({"label": names, "support": y.sum(0), "false_negatives": fn.sum(0), "false_positives": fp.sum(0),
                        "fn_rate(miss)": (fn.sum(0) / np.maximum(y.sum(0), 1)).round(3),
                        "fp_per_100": (100 * fp.sum(0) / len(y)).round(2)})
    lab.sort_values("false_negatives", ascending=False).to_csv(TABLES_DIR / "error_by_label.csv", index=False)

    # systematic confusion: true label a missed while label b spuriously predicted in the same comment
    C = len(names)
    conf = np.zeros((C, C), int)
    for i in range(len(y)):
        for a in np.flatnonzero(fn[i]):
            conf[a, fp[i]] += 1
    cp = [{"missed_true_label": names[a], "spurious_pred_label": names[b], "count": int(conf[a, b])}
          for a in range(C) for b in range(C) if conf[a, b] > 0]
    cp = pd.DataFrame(cp).sort_values("count", ascending=False)
    cp.to_csv(TABLES_DIR / "error_confusion_pairs.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(conf, xticklabels=names, yticklabels=names, cmap="magma_r", ax=ax, square=True,
                cbar_kws={"shrink": 0.7, "label": "# test comments"})
    ax.set_xlabel("spuriously predicted label (FP)")
    ax.set_ylabel("missed true label (FN)")
    ax.set_title(f"Systematic confusions of {dname(run)} (test, tau = 0.5)")
    _save(fig, "error_confusion_pairs")

    # error profile by style bucket and by cardinality
    def profile(mask):
        return {"n": int(mask.sum()), "exact_match": float((pred[mask] == y[mask]).all(1).mean()),
                "samples_f1": f1_score(y[mask], pred[mask], average="samples", zero_division=0),
                "micro_f1": f1_score(y[mask], pred[mask], average="micro", zero_division=0),
                "avg_FN_per_comment": float(fn[mask].sum(1).mean()), "avg_FP_per_comment": float(fp[mask].sum(1).mean()),
                "empty_prediction_rate": float((pred[mask].sum(1) == 0).mean())}
    rows = [{"bucket": "all", **profile(np.ones(len(y), bool))}]
    for k in RULES:
        rows.append({"bucket": f"style={k}", **profile(flags[k].to_numpy())})
    for k in (1, 2, 3):
        rows.append({"bucket": f"|Y|={k}", **profile(card == k)})
    rows.append({"bucket": "|Y|>=4", **profile(card >= 4)})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "error_by_bucket.csv", index=False)

    # case lists (public = no text; private = with text for manual audit)
    texts = np.asarray(data.texts("test"))
    ids = d["ids"]
    rng = np.random.default_rng(42)

    def fmt(idx):
        return ", ".join(f"{names[c]}({pr[i_, c]:.2f})" for i_, c in idx)

    cases = []
    fn_idx = np.argwhere(fn)
    # most confident misses = lowest prob among FN
    for i, c in sorted(fn_idx.tolist(), key=lambda t: pr[t[0], t[1]])[:n_cases]:
        cases.append(("false_negative", i, f"missed {names[c]} (p={pr[i, c]:.2f})"))
    fp_idx = np.argwhere(fp)
    for i, c in sorted(fp_idx.tolist(), key=lambda t: -pr[t[0], t[1]])[:n_cases]:
        cases.append(("false_positive", i, f"spurious {names[c]} (p={pr[i, c]:.2f})"))
    hc = np.flatnonzero((card >= 3) & ~(pred == y).all(1))
    for i in rng.choice(hc, size=min(15, len(hc)), replace=False):
        cases.append(("high_cardinality_partial", i, f"|Y|={card[i]}, recovered {int((pred[i] & y[i]).sum())}"))
    for bucket in ("short", "emoji_heavy", "teencode", "emoticon"):
        idx = np.flatnonzero(flags[bucket].to_numpy() & ~(pred == y).all(1))
        for i in rng.choice(idx, size=min(8, len(idx)), replace=False):
            cases.append((f"style_{bucket}", i, "exact-match error"))
    rec = []
    for kind, i, note in cases:
        true_l = [names[c] for c in np.flatnonzero(y[i])]
        pred_l = [names[c] for c in np.flatnonzero(pred[i])]
        top = np.argsort(-pr[i])[:4]
        style = [k for k in ("short", "long", "emoji", "emoticon", "teencode") if flags[k].iloc[i]]
        rec.append({"case_type": kind, "sample_id": ids[i], "style_buckets": "|".join(style) or "plain",
                    "true_labels": ", ".join(true_l), "pred_labels": ", ".join(pred_l) or "(none)",
                    "top4_probs": fmt([(i, c) for c in top]), "note": note, "_text": texts[i]})
    df = pd.DataFrame(rec)
    df.drop(columns=["_text"]).to_csv(TABLES_DIR / "error_cases_public.csv", index=False)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    df.rename(columns={"_text": "text"}).to_csv(PRIVATE_DIR / f"error_cases_with_text_{run}.csv", index=False)
    return lab, cp


def cardinality_tier_breakdown(R, data, tiers):
    names = data.label_names
    y = data.y["test"]
    card = y.sum(1)
    rows = []
    for r, d in R.items():
        for strat, tau in d["thresholds"].items():
            pred = binarize(d["test"]["probs"], tau)
            row = {"model": dname(r), "strategy": strat}
            f1s = f1_score(y, pred, average=None, zero_division=0)
            for t in ("head", "mid", "tail"):
                row[f"macro_f1_{t}"] = float(np.mean([f for n, f in zip(names, f1s) if tiers[n] == t]))
            for k, m in (("|Y|=1", card == 1), ("|Y|=2", card == 2), ("|Y|=3", card == 3), ("|Y|>=4", card >= 4)):
                row[f"micro_f1_{k}"] = f1_score(y[m], pred[m], average="micro", zero_division=0)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tier_and_cardinality_breakdown.csv", index=False)
    return df


def update_readme(t: pd.DataFrame, thr: pd.DataFrame, required: list[str]):
    """Write the headline tables between the RESULTS markers of README.md (generated, not hand-typed)."""
    from vigo.utils import PROJECT_ROOT

    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        return
    main = t[["model", "macro_f1", "micro_f1", "hamming_loss", "lrap", "map_macro", "params", "train_time_s"]].copy()
    main["params"] = main["params"].map(lambda x: f"{x / 1e6:.1f}M")
    main["train_time_s"] = main["train_time_s"].map(lambda x: f"{x / 60:.1f} min")
    main.columns = ["Model", "Macro-F1", "Micro-F1", "Hamming ↓", "LRAP", "mAP", "Params", "Train time"]
    piv = thr.pivot_table(index="model", columns="strategy", values="test_macro_f1", sort=False)[["fixed", "global", "per_label"]]
    piv = piv.reset_index().rename(columns={"model": "Model", "fixed": "τ=0.5", "global": "global τ* (val)",
                                             "per_label": "per-label τ*_c (val)"})
    block = ["## Results (test set, generated by `python -m vigo analyze`)", "",
             "**All required metrics at the default threshold τ = 0.5** (first three rows = required models):", "",
             main.to_markdown(index=False, floatfmt=".4f"), "",
             "**Test Macro-F1 under the three decoding strategies** (thresholds fitted on validation only, then frozen):", "",
             piv.to_markdown(index=False, floatfmt=".4f"), "",
             "Published reference (same split, τ = 0.5): ViSoBERT 0.6150, CafeBERT 0.6117, PhoBERT 0.5633 / 0.5931 "
             "(emoji kept / emoji→text) Macro-F1 — `results/tables/comparison_with_paper.csv`."]
    text = readme.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in text and b in text:
        text = text.split(a)[0] + a + "\n" + "\n".join(block) + "\n" + b + text.split(b)[1]
        readme.write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tfidf", default="tfidf_lr")
    ap.add_argument("--rnn", default="bilstm")
    ap.add_argument("--transformer", default="phobert")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--error_model", default=None, help="run used for error analysis (default: --transformer)")
    args = ap.parse_args(argv)
    ensure_dirs()
    sns.set_theme(style="white", font_scale=0.9)
    data = load_data(args.data_dir)
    tiers = frequency_tiers(data.y["train"].sum(0), data.label_names)
    required = [args.tfidf, args.rnn, args.transformer]
    R = load_runs(required + [e for e in args.extra if e not in required])
    for r, d in R.items():  # guard: every run must be on the same test ids
        assert list(d["test"]["ids"]) == data.ids("test"), f"{r}: test ids differ from the fixed split"

    t, v, thr = main_tables(R, data.label_names)
    per_label = per_label_tables(R, data, tiers)
    req = {k: R[k] for k in required if k in R}
    q2 = q1_q2(per_label, req, args.transformer if args.transformer in R else list(R)[0])
    q3, q3g = q3_cooccurrence(req, data)
    q4 = q4_text_style(req, data, args.rnn, args.transformer, args.tfidf)
    q5 = q5_thresholds(R, data.label_names, plot_runs=required)
    export_predictions(R)
    cardinality_tier_breakdown(R, data, tiers)
    em = args.error_model or args.transformer
    if em in R:
        error_analysis(R, data, em)
    update_readme(t, thr, required)

    pd.set_option("display.width", 200)
    print("\n== Test results (tau = 0.5)\n", t[["model", "macro_f1", "micro_f1", "hamming_loss", "lrap", "map_macro",
                                             "params", "train_time_s"]].round(4).to_string(index=False))
    print("\n== Threshold strategies (fitted on val)\n",
          thr[["model", "strategy", "tau", "val_macro_f1", "test_macro_f1", "test_micro_f1"]].round(4).to_string(index=False))
    print("\n== Q2 frequency correlation\n", q2.round(3).to_string(index=False))
    print("\n== Q3 dependency preservation\n", q3g.round(3).to_string(index=False))
    print("\n== Q4 text style\n", q4.round(4).to_string(index=False))
    print("\n== Q5 threshold sensitivity (val)\n", q5.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
