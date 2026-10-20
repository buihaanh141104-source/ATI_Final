"""Fixed split: write ID files and audit multi-label stratification quality.

    python -m vigo split                 # write splits/official/*.txt + audit table
    python -m vigo split --make iterative  # also create an iterative-stratified 80/10/10 split

Audit metrics (Sechidis et al., 2011): for each split, the Label Distribution (LD) and
the mean absolute deviation of per-label positive rates from the pooled rates; plus the
minimum number of positives per label in val/test. Compared against a random split and an
iterative-stratified split generated with the same seed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from vigo.data.dataset import SPLITS, load_label_names, load_pooled, to_multihot
from vigo.utils import SPLITS_DIR, TABLES_DIR, data_dir, ensure_dirs

RATIOS = (0.8, 0.1, 0.1)


def iterative_split(y: np.ndarray, seed: int = 42, ratios=RATIOS) -> dict[str, np.ndarray]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    idx = np.arange(len(y))
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=ratios[1] + ratios[2], random_state=seed)
    tr, rest = next(s1.split(idx, y))
    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=ratios[2] / (ratios[1] + ratios[2]),
                                          random_state=seed)
    va_rel, te_rel = next(s2.split(rest, y[rest]))
    return {"train": tr, "val": rest[va_rel], "test": rest[te_rel]}


def random_split(n: int, seed: int = 42, ratios=RATIOS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    return {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}


def audit(y: np.ndarray, parts: dict[str, np.ndarray]) -> dict:
    pooled = y.mean(0)
    pooled_ld = pooled / np.clip(1 - pooled, 1e-9, None)
    out = {}
    for s, idx in parts.items():
        p = y[idx].mean(0)
        ld = p / np.clip(1 - p, 1e-9, None)
        out[f"{s}_n"] = len(idx)
        out[f"{s}_mad_rate"] = float(np.mean(np.abs(p - pooled)))
        out[f"{s}_LD"] = float(np.mean(np.abs(ld - pooled_ld)))
        out[f"{s}_min_pos"] = int(y[idx].sum(0).min())
    out["LD_mean"] = float(np.mean([out[f"{s}_LD"] for s in parts]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--make", default=None, help="'iterative' to also materialise that split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    ensure_dirs()
    names = load_label_names(args.data_dir)
    pooled = load_pooled(args.data_dir)
    y = to_multihot(pooled["label_list"].tolist(), len(names))

    official = {s: np.flatnonzero(pooled["official_split"].to_numpy() == s) for s in SPLITS}
    candidates = {
        "official (released)": official,
        f"random (seed {args.seed})": random_split(len(y), args.seed),
        f"iterative stratified (seed {args.seed})": iterative_split(y, args.seed),
    }
    rows = [{"split_protocol": k, **audit(y, v)} for k, v in candidates.items()]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "split_quality_audit.csv", index=False)
    print(df.round(5).to_string(index=False))

    # official split id files (the split used by every model)
    for name, parts in [("official", official)] + (
        [("iterative", candidates[f"iterative stratified (seed {args.seed})"])] if args.make == "iterative" else []
    ):
        d = SPLITS_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            (d / f"{s}_ids.txt").write_text("\n".join(pooled["id"].iloc[parts[s]].tolist()) + "\n", encoding="utf-8")
        print(f"wrote {d}/{{train,val,test}}_ids.txt")

    # per-label coverage in the official split
    cov = pd.DataFrame({"label": names})
    for s in SPLITS:
        cov[f"{s}_count"] = y[official[s]].sum(0)
        cov[f"{s}_pct"] = (100 * y[official[s]].mean(0)).round(2)
    cov.to_csv(TABLES_DIR / "split_label_coverage.csv", index=False)
    print(f"data dir: {data_dir(args.data_dir)}")


if __name__ == "__main__":
    main()
