"""PyTorch datasets, samplers and collate functions for the neural models."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# ----------------------------------------------------------------------------- BiLSTM (word ids)
class SeqDataset(Dataset):
    def __init__(self, ids: list[list[int]], y: np.ndarray):
        self.ids, self.y = ids, y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def rnn_collate(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    T = int(lengths.max())
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    mask = (x != 0).long()
    return {"input_ids": x, "lengths": lengths, "attention_mask": mask,
            "labels": torch.tensor(np.stack(ys), dtype=torch.float32)}


# ----------------------------------------------------------------------------- transformers (sub-word ids)
class EncodedDataset(Dataset):
    def __init__(self, enc: list[list[int]], y: np.ndarray):
        self.enc, self.y = enc, y

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return self.enc[i], self.y[i]


class LengthGroupedSampler(Sampler):
    """Shuffle, then sort inside mega-batches by length -> less padding, still random."""

    def __init__(self, lengths, batch_size, seed, mega=50):
        self.lengths, self.bs, self.seed, self.mega, self.epoch = np.asarray(lengths), batch_size, seed, mega, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        chunk = self.bs * self.mega
        batches = []
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s : s + chunk], key=lambda i: -self.lengths[i])
            batches += [part[j : j + self.bs] for j in range(0, len(part), self.bs)]
        rng.shuffle(batches)
        for b in batches:
            yield from b

    def __len__(self):
        return len(self.lengths)


def make_transformer_collate(pad_id: int):
    def collate(batch):
        seqs, ys = zip(*batch)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor(np.stack(ys), dtype=torch.float32)}

    return collate
