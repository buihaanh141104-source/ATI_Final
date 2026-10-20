"""M2 - BiLSTM / BiGRU multi-label classifier.

Embedding -> spatial dropout -> 2-layer bidirectional LSTM (or GRU)
-> [additive attention pooling ; max pooling] -> dropout -> linear (C sigmoid outputs).
Output bias is initialised to the log-odds of each label's training prior so that rare
labels start calibrated (helps convergence under imbalance).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):  # x: (B, T, E) -> drop whole embedding channels
        x = x.permute(0, 2, 1).unsqueeze(3)
        x = super().forward(x)
        return x.squeeze(3).permute(0, 2, 1)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_labels: int,
        emb_dim: int = 300,
        hidden: int = 256,
        layers: int = 2,
        cell: str = "lstm",
        emb_dropout: float = 0.25,
        rnn_dropout: float = 0.25,
        dropout: float = 0.3,
        pretrained: np.ndarray | None = None,
        label_prior: np.ndarray | None = None,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.as_tensor(pretrained, dtype=torch.float32))
        self.emb_drop = SpatialDropout(emb_dropout)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
            dropout=rnn_dropout if layers > 1 else 0.0,
        )
        d = 2 * hidden
        self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * d, n_labels)
        if label_prior is not None:
            p = np.clip(label_prior, 1e-4, 1 - 1e-4)
            self.out.bias.data = torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32)

    def forward(self, input_ids, lengths, attention_mask):
        x = self.emb_drop(self.emb(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=input_ids.size(1))
        mask = attention_mask.unsqueeze(-1).bool()
        scores = self.att(h).masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=1)
        att_pool = (w * h).sum(1)
        max_pool = h.masked_fill(~mask, -1e4).max(1).values
        z = self.drop(torch.cat([att_pool, max_pool], dim=-1))
        return self.out(z)
