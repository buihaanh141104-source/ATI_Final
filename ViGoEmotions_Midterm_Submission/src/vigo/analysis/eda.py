"""WP1 - Dataset analysis: every table/figure required by the brief, generated from code.

    python -m vigo eda

Outputs (results/tables, results/figures). No raw comment text is written to tables.
Co-occurrence / correlation are computed on TRAINING labels only (reproducibility rule).
"""
from __future__ import annotations

import argparse
import collections
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vigo.data.dataset import SPLITS, frequency_tiers, load_data
from vigo.data.preprocessing import _emoticon_regex, count_emoji, has_diacritics, strip_variation, teencode_hits
from vigo.analysis.text_styles import RULES, style_flags
from vigo.utils import FIGURES_DIR, TABLES_DIR, ensure_dirs

TIER_COLORS = {"head": "#2a6fdb", "mid": "#f2a541", "tail": "#d64550"}
SOURCE_MAP = {"tik": "TikTok", "you": "YouTube", "x": "X (Twitter)", "red": "Reddit", "thr": "Threads",
              "": "Facebook (numeric id, incl. UIT-VSMEC)"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=160)
    plt.close(fig)


def label_tables(data):
    names = data.label_names
    ytr = data.y["train"]
    counts = ytr.sum(0)
    tiers = frequency_tiers(counts, names)
    inv = pd.DataFrame({"index": range(len(names)), "label": names})
    for s in SPLITS:
        inv[f"{s}_count"] = data.y[s].sum(0)
        inv[f"{s}_pct"] = (100 * data.y[s].mean(0)).round(2)
    inv["all_count"] = sum(data.y[s].sum(0) for s in SPLITS)
    inv["tier"] = inv["label"].map(tiers)
    inv["train_rank"] = (-counts).argsort().argsort() + 1
    inv.to_csv(TABLES_DIR / "label_inventory.csv", index=False)

    freq = inv.sort_values("train_count", ascending=False)[["label", "train_count", "train_pct", "tier", "train_rank"]]
    freq.to_csv(TABLES_DIR / "label_frequency.csv", index=False)
    order = freq["train_count"].to_numpy()
    tier_tab = freq.groupby("tier").agg(n_labels=("label", "size"), min_count=("train_count", "min"),
                                        max_count=("train_count", "max"),
                                        labels=("label", lambda x: ", ".join(x))).reindex(["head", "mid", "tail"])
    tier_tab["rule"] = "tertiles of training positive counts (9 head / 10 mid / 9 tail)"
    tier_tab.to_csv(TABLES_DIR / "label_tiers.csv")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(freq["label"], freq["train_count"], color=[TIER_COLORS[t] for t in freq["tier"]])
    for i, v in enumerate(order):
        ax.text(i, v + 15, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("# training comments")
    ax.set_title("Label frequency (train) - head / mid / tail tiers")
    ax.tick_params(axis="x", rotation=70)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLORS.values()]
    ax.legend(handles, [f"{k} ({(freq.tier == k).sum()})" for k in TIER_COLORS], frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "label_frequency")
    imbalance = float(order.max() / order.min())
    return inv, tiers, imbalance


def cardinality(data):
    rows = []
    for s in SPLITS:
        card = data.y[s].sum(1)
        vc = pd.Series(card).value_counts().sort_index()
        row = {"split": s, "n": len(card), "mean": card.mean(), "median": np.median(card), "max": card.max(),
               "density": card.mean() / data.n_labels}
        for k in range(1, 6):
            row[f"pct_|Y|={k}"] = round(100 * (card == k).mean(), 2)
        row["neutral_only_pct"] = round(100 * ((card == 1) & (data.y[s][:, -1] == 1)).mean(), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "cardinality_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    width = 0.27
    for i, s in enumerate(SPLITS):
        card = data.y[s].sum(1)
        ks = np.arange(1, card.max() + 1)
        ax.bar(ks + (i - 1) * width, [(card == k).mean() * 100 for k in ks], width, label=s)
    ax.set_xlabel("|Y| (labels per comment)")
    ax.set_ylabel("% of comments")
    ax.set_title("Label cardinality distribution")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "cardinality_distribution")
    return df


def cooccurrence(data):
    names = data.label_names
    y = data.y["train"].astype(float)
    co = y.T @ y
    n = len(y)
    p = y.mean(0)
    # phi coefficient (Pearson correlation of binary indicators)
    corr = np.corrcoef(y.T)
    pairs = []
    C = len(names)
    for i in range(C):
        for j in range(i + 1, C):
            if co[i, j] == 0:
                continue
            pij = co[i, j] / n
            pairs.append({
                "label_a": names[i], "label_b": names[j], "count": int(co[i, j]),
                "pct_of_train": round(100 * pij, 2),
                "P(b|a)": round(co[i, j] / co[i, i], 3), "P(a|b)": round(co[i, j] / co[j, j], 3),
                "phi": round(corr[i, j], 3), "pmi": round(np.log2(pij / (p[i] * p[j])), 3),
            })
    top = pd.DataFrame(pairs).sort_values("count", ascending=False)
    top.to_csv(TABLES_DIR / "cooccurrence_pairs_train.csv", index=False)
    top.head(20).to_csv(TABLES_DIR / "top_pairs.csv", index=False)
    pd.DataFrame(co.astype(int), index=names, columns=names).to_csv(TABLES_DIR / "cooccurrence_matrix_train.csv")
    pd.DataFrame(corr, index=names, columns=names).round(4).to_csv(TABLES_DIR / "correlation_matrix_train.csv")

    co_plot = co.copy()
    np.fill_diagonal(co_plot, np.nan)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(co_plot, xticklabels=names, yticklabels=names, cmap="viridis", ax=ax, square=True,
                cbar_kws={"label": "# co-occurring training comments", "shrink": 0.7})
    ax.set_title("Label co-occurrence (train, diagonal masked)")
    _save(fig, "cooccurrence")
    corr_plot = corr.copy()
    np.fill_diagonal(corr_plot, np.nan)
    fig, ax = plt.subplots(figsize=(11, 9))
    lim = np.nanmax(np.abs(corr_plot))
    sns.heatmap(corr_plot, xticklabels=names, yticklabels=names, cmap="RdBu_r", vmin=-lim, vmax=lim, center=0,
                ax=ax, square=True, cbar_kws={"label": "phi coefficient", "shrink": 0.7})
    ax.set_title("Pairwise label correlation (phi, train)")
    _save(fig, "correlation")
    return top


def text_length(data):
    rows = []
    lens = {}
    for s in SPLITS:
        t = data.texts(s)
        tok = np.array([len(x.split()) for x in t])
        ch = np.array([len(x) for x in t])
        lens[s] = (tok, ch)
        for kind, arr in (("whitespace_tokens", tok), ("characters", ch)):
            rows.append({"split": s, "unit": kind, "min": arr.min(), "p25": np.percentile(arr, 25),
                         "median": np.median(arr), "mean": round(arr.mean(), 2), "p90": np.percentile(arr, 90),
                         "p95": np.percentile(arr, 95), "p99": np.percentile(arr, 99), "max": arr.max()})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "text_length_stats.csv", index=False)
    for idx, (unit, fname, clip) in enumerate((("whitespace tokens", "token_length", 100),
                                               ("characters", "char_length", 500))):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        arr = lens["train"][idx]
        ax.hist(np.clip(arr, 0, clip), bins=60, color="#2a6fdb", alpha=0.85)
        for q, ls in ((50, "--"), (95, ":"), (99, "-.")):
            v = np.percentile(arr, q)
            ax.axvline(v, color="k", ls=ls, lw=1, label=f"p{q} = {v:.0f}")
        ax.set_xlabel(f"{unit} per comment (train; clipped at {clip})")
        ax.set_ylabel("# comments")
        ax.set_title(f"Comment length in {unit}")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, fname)
    return df


def informal_audit(data):
    """Aggregate (non-reconstructable) statistics of informal Vietnamese phenomena."""
    pooled = pd.concat([data.frames[s].assign(split=s) for s in SPLITS], ignore_index=True)
    texts = pooled["text"].tolist()
    emo_re = _emoticon_regex()
    feats = {
        "contains emoji": [count_emoji(t) > 0 for t in texts],
        ">= 2 emoji": [count_emoji(t) >= 2 for t in texts],
        "ASCII emoticon (:)) =)) :v ^^)": [bool(emo_re.search(t.lower())) for t in texts],
        "teencode / abbreviation token (dictionary)": [len(teencode_hits(t)) > 0 for t in texts],
        "multi-char teencode token": [any(len(h) > 1 for h in teencode_hits(t)) for t in texts],
        "elongated word (letter x3+, e.g. 'quáaaa')": [bool(re.search(r"([^\W\d_])\1{2,}", t)) for t in texts],
        "repeated punctuation (!!, ??, ...)": [bool(re.search(r"([!?.])\1{1,}", t)) for t in texts],
        "laughter token (haha, hihi, kkk, =))": [bool(re.search(r"\b(ha){2,}|\b(hi){2,}|\bk{3,}\b|=\)\)", t.lower())) for t in texts],
        "@mention": [bool(re.search(r"@\w+", t)) for t in texts],
        "URL": [bool(re.search(r"https?://|www\.", t)) for t in texts],
        "hashtag": [bool(re.search(r"#\w+", t)) for t in texts],
        "ALL-CAPS word (>= 3 letters)": [bool(re.search(r"\b[A-ZÀ-Ỹ]{3,}\b", t)) for t in texts],
        "unaccented Vietnamese (>= 4 tokens, no diacritics)": [len(t.split()) >= 4 and not has_diacritics(t) for t in texts],
        "anonymised person placeholder 'per'": [bool(re.search(r"\bper\b", t.lower())) for t in texts],
        "<br> / line break": [("<br>" in t.lower()) or ("\n" in t) for t in texts],
        "profanity / vulgar slang (dm, vl, vcl, đm, cc...)": [
            bool(re.search(r"\b(đm|dm|vl|vcl|vkl|cc|clgt|đéo|đ[ée]o|địt|dit|lồn|loz|cmm|cmn|đcm|dcm|vãi)\b", t.lower()))
            for t in texts],
    }
    rows = [{"phenomenon": k, "n_comments": int(np.sum(v)), "pct": round(100 * np.mean(v), 2)} for k, v in feats.items()]
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "informal_text_audit.csv", index=False)

    tc = collections.Counter(h for t in texts for h in teencode_hits(t))
    from vigo.data.preprocessing import load_teencode

    table = load_teencode()
    pd.DataFrame([{"token": k, "normalised": table[k], "count": v} for k, v in tc.most_common(40)]).to_csv(
        TABLES_DIR / "teencode_top.csv", index=False)
    import emoji as emoji_lib

    ec = collections.Counter(strip_variation(m["emoji"]) for t in texts for m in emoji_lib.emoji_list(t))
    pd.DataFrame([{"emoji": k, "name": emoji_lib.demojize(k).strip(":"), "count": v} for k, v in ec.most_common(40)]
                 ).to_csv(TABLES_DIR / "emoji_top.csv", index=False)

    # emoji usage per emotion (train) - which emotions are emoji-heavy
    tr = data.frames["train"]
    has_e = np.array([count_emoji(t) > 0 for t in tr["text"]])
    per_label = pd.DataFrame({
        "label": data.label_names,
        "pct_with_emoji": [round(100 * has_e[data.y["train"][:, c] == 1].mean(), 1) for c in range(data.n_labels)],
    }).sort_values("pct_with_emoji", ascending=False)
    per_label.to_csv(TABLES_DIR / "emoji_rate_by_label.csv", index=False)

    # source platforms (from id prefix)
    prefix = pooled["id"].str.extract(r"^([a-z]*)")[0].fillna("")
    src = prefix.map(SOURCE_MAP).fillna("other")
    src_tab = pd.crosstab(src, pooled["split"]).reindex(columns=list(SPLITS))
    src_tab["total"] = src_tab.sum(1)
    src_tab["pct"] = (100 * src_tab["total"] / src_tab["total"].sum()).round(2)
    src_tab.sort_values("total", ascending=False).to_csv(TABLES_DIR / "source_distribution.csv")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    a = audit.sort_values("pct")
    ax.barh(a["phenomenon"], a["pct"], color="#2a6fdb")
    for i, v in enumerate(a["pct"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of all 20,664 comments")
    ax.set_title("Informal Vietnamese phenomena")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "informal_text_audit")
    return audit


def style_subsets(data):
    rows = []
    for s in SPLITS:
        f = style_flags(data.texts(s))
        for k, rule in RULES.items():
            rows.append({"subset": k, "rule": rule, "split": s, "n": int(f[k].sum()), "pct": round(100 * f[k].mean(), 2)})
    df = pd.DataFrame(rows).pivot_table(index=["subset", "rule"], columns="split", values=["n", "pct"])
    df.to_csv(TABLES_DIR / "text_style_subsets.csv")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args(argv)
    ensure_dirs()
    sns.set_theme(style="white", font_scale=0.9)
    data = load_data(args.data_dir, "official")
    n_total = sum(len(data.frames[s]) for s in SPLITS)
    inv, tiers, imbalance = label_tables(data)
    card = cardinality(data)
    top = cooccurrence(data)
    lengths = text_length(data)
    audit = informal_audit(data)
    style_subsets(data)
    y_all = np.concatenate([data.y[s] for s in SPLITS])
    summary = pd.DataFrame([
        {"item": "total comments", "value": n_total},
        *[{"item": f"{s} comments", "value": len(data.frames[s])} for s in SPLITS],
        {"item": "labels (27 emotions + neutral)", "value": data.n_labels},
        {"item": "mean labels per comment |Y|", "value": round(float(y_all.sum(1).mean()), 3)},
        {"item": "label density", "value": round(float(y_all.mean()), 4)},
        {"item": "distinct label sets", "value": len({tuple(r) for r in y_all})},
        {"item": "most frequent label (train)", "value": inv.loc[inv.train_count.idxmax(), "label"]},
        {"item": "rarest label (train)", "value": inv.loc[inv.train_count.idxmin(), "label"]},
        {"item": "imbalance ratio max/min (train)", "value": round(imbalance, 2)},
        {"item": "median whitespace tokens (train)", "value": float(lengths.query("split=='train' and unit=='whitespace_tokens'")["median"].iloc[0])},
    ])
    summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    assert n_total == 20664, n_total
    assert data.n_labels == 28
    print(summary.to_string(index=False))
    print("\ncardinality:\n", card.round(3).to_string(index=False))
    print("\ntop pairs:\n", top.head(10).to_string(index=False))
    print("\ninformal audit:\n", audit.to_string(index=False))


if __name__ == "__main__":
    main()
