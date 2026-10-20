"""Dataset loading, label encoding and the fixed train/validation/test split.

Split protocol (documented in README):
  * Primary: the official ViGoEmotions 80/10/10 partition released by the authors
    (train 16,531 / val 2,066 / test 2,067). Using it keeps our numbers directly
    comparable with the EACL 2026 paper. Its multi-label stratification quality is
    audited in `vigo.data.split` against a random split and an iterative-stratified split.
  * Alternative: `split=iterative` re-partitions the pooled 20,664 comments with
    iterative multi-label stratification (Sechidis et al., 2011; Szymanski & Kajdanowicz 2017)
    using a fixed seed. IDs for both are written to `splits/`.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vigo.utils import SPLITS_DIR, data_dir

SPLITS = ("train", "val", "test")
NEUTRAL = "neutral"


def load_label_names(ddir: str | Path | None = None) -> list[str]:
    path = Path(data_dir(ddir)) / "label_dict.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return [d[str(i)] for i in range(len(d))]
    # Fallback: official ordering (same as label_dict.json of the release).
    return [
        "amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride",
        "admiration", "gratitude", "relief", "approval", "realization", "surprise", "curiosity",
        "confusion", "fear", "nervousness", "remorse", "embarrassment", "disappointment",
        "sadness", "grief", "disgust", "anger", "annoyance", "disapproval", "neutral",
    ]


def parse_labels(raw) -> list[int]:
    """'[2, 8, 3]' -> [2, 8, 3]; duplicates removed (one test row lists label 12 twice)."""
    vals = raw if isinstance(raw, (list, tuple)) else ast.literal_eval(str(raw))
    return list(dict.fromkeys(int(x) for x in vals))


def to_multihot(label_lists: list[list[int]], n_labels: int) -> np.ndarray:
    y = np.zeros((len(label_lists), n_labels), dtype=np.int8)
    for i, labs in enumerate(label_lists):
        y[i, labs] = 1
    return y


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str}, encoding="utf-8-sig", keep_default_na=False)
    df["id"] = df["id"].astype(str).str.strip()
    df["label_list"] = df["labels"].apply(parse_labels)
    return df[["id", "text", "labels", "label_list"]]


def load_pooled(ddir: str | Path | None = None) -> pd.DataFrame:
    """All 20,664 comments with their official split name in column `official_split`."""
    ddir = Path(data_dir(ddir))
    frames = []
    for s in SPLITS:
        df = _read_csv(ddir / f"{s}.csv")
        df["official_split"] = s
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    assert pooled["id"].is_unique, "sample ids must be unique"
    return pooled


@dataclass
class DataBundle:
    label_names: list[str]
    frames: dict[str, pd.DataFrame]  # split -> DataFrame(id, text, label_list)
    y: dict[str, np.ndarray]  # split -> multi-hot (n, C)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)

    def texts(self, split: str) -> list[str]:
        return self.frames[split]["text"].tolist()

    def ids(self, split: str) -> list[str]:
        return self.frames[split]["id"].tolist()


def load_data(ddir: str | Path | None = None, split: str = "official") -> DataBundle:
    """Load the fixed split. `split` = 'official' or 'iterative' (reads splits/<name>/*_ids.txt)."""
    label_names = load_label_names(ddir)
    pooled = load_pooled(ddir)
    frames: dict[str, pd.DataFrame] = {}
    if split == "official":
        for s in SPLITS:
            frames[s] = pooled[pooled["official_split"] == s].reset_index(drop=True)
    else:
        sdir = SPLITS_DIR / split
        if not (sdir / "train_ids.txt").exists():
            raise FileNotFoundError(
                f"{sdir}/train_ids.txt not found. Run `python -m vigo split --make {split}` first."
            )
        by_id = pooled.set_index("id")
        for s in SPLITS:
            ids = (sdir / f"{s}_ids.txt").read_text(encoding="utf-8").split()
            frames[s] = by_id.loc[ids].reset_index()
    y = {s: to_multihot(frames[s]["label_list"].tolist(), len(label_names)) for s in SPLITS}
    return DataBundle(label_names=label_names, frames=frames, y=y)


def frequency_tiers(train_counts: np.ndarray, label_names: list[str]) -> dict[str, str]:
    """Head / mid / tail rule, fixed BEFORE any model is evaluated.

    Labels are ranked by TRAINING positive count. Tertile cut-points of that count
    distribution split them into head (top third), mid, tail (bottom third). With 28 labels
    this gives 9 head / 10 mid / 9 tail. The same rule is reused in the Final.
    """
    order = np.argsort(-train_counts, kind="stable")
    n = len(order)
    n_head = int(round(n / 3))
    n_tail = int(round(n / 3))
    tiers = {}
    for rank, idx in enumerate(order):
        if rank < n_head:
            tiers[label_names[idx]] = "head"
        elif rank >= n - n_tail:
            tiers[label_names[idx]] = "tail"
        else:
            tiers[label_names[idx]] = "mid"
    return tiers
