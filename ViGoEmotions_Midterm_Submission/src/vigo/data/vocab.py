"""Word vocabulary (fitted on TRAIN only) and optional pre-trained word vectors for the BiLSTM."""
from __future__ import annotations

import collections

import numpy as np

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, tokens_per_doc: list[list[str]], min_freq: int = 2, max_size: int | None = 50000):
        cnt = collections.Counter(t for doc in tokens_per_doc for t in doc)
        items = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
        self.itos = [PAD, UNK] + items
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        return ids if ids else [1]


def load_word_vectors(path: str, vocab: Vocab, dim: int, seed: int = 42) -> tuple[np.ndarray, float]:
    """Load word2vec/fastText .vec text file; returns matrix + coverage of the vocabulary."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    mat[0] = 0
    found = 0
    opener = open
    if path.endswith(".gz"):
        import gzip

        opener = gzip.open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        first = f.readline().split()
        if len(first) != 2:  # no header line
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            w = parts[0]
            idx = vocab.stoi.get(w)
            if idx is None:
                idx = vocab.stoi.get(w.lower())
            if idx is not None and idx > 1:
                mat[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
    return mat, found / max(len(vocab) - 2, 1)
