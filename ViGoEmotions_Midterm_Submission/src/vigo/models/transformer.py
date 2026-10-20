"""M3 - Pre-trained transformer encoder + independent sigmoid head (multi-label).

encoder (PhoBERT / ViSoBERT / XLM-R / CafeBERT) -> pooled vector (CLS, mean, or both)
-> dropout (optionally multi-sample dropout) -> Linear(hidden, C). One logit per label,
sigmoid at inference, trained with BCE (see losses.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TransformerMultiLabel(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_labels: int,
        pooling: str = "cls",
        dropout: float = 0.1,
        multi_sample_dropout: int = 1,
        hidden_dropout: float | None = None,
        attention_dropout: float | None = None,
        label_prior: np.ndarray | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        kw = {}
        if hidden_dropout is not None:
            kw["hidden_dropout_prob"] = hidden_dropout
        if attention_dropout is not None:
            kw["attention_probs_dropout_prob"] = attention_dropout
        config = AutoConfig.from_pretrained(model_name, **kw)
        self.encoder = AutoModel.from_pretrained(model_name, config=config, add_pooling_layer=False)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.pooling = pooling
        h = config.hidden_size
        in_dim = 2 * h if pooling == "cls_mean" else h
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(max(1, multi_sample_dropout))])
        self.head = nn.Linear(in_dim, n_labels)
        nn.init.normal_(self.head.weight, std=0.02)
        if label_prior is not None:
            p = np.clip(label_prior, 1e-4, 1 - 1e-4)
            self.head.bias.data = torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32)
        else:
            nn.init.zeros_(self.head.bias)

    def pool(self, hidden, mask):
        cls = hidden[:, 0]
        if self.pooling == "cls":
            return cls
        m = mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        if self.pooling == "mean":
            return mean
        return torch.cat([cls, mean], dim=-1)

    def forward(self, input_ids, attention_mask, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        z = self.pool(out.last_hidden_state, attention_mask)
        if len(self.dropouts) == 1:
            return self.head(self.dropouts[0](z))
        return torch.stack([self.head(d(z)) for d in self.dropouts], 0).mean(0)


def llrd_param_groups(model: TransformerMultiLabel, lr: float, head_lr: float, decay: float, weight_decay: float):
    """Layer-wise learning-rate decay: top encoder layer gets `lr`, each layer below x`decay`."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm", "layernorm")
    enc = model.encoder
    n_layers = enc.config.num_hidden_layers
    groups = []

    def add(params_named, group_lr):
        dec = [p for n, p in params_named if not any(nd in n for nd in no_decay) and p.requires_grad]
        nodec = [p for n, p in params_named if any(nd in n for nd in no_decay) and p.requires_grad]
        if dec:
            groups.append({"params": dec, "lr": group_lr, "weight_decay": weight_decay})
        if nodec:
            groups.append({"params": nodec, "lr": group_lr, "weight_decay": 0.0})

    assigned = set()
    for i in range(n_layers):
        named = [(n, p) for n, p in enc.named_parameters() if f"encoder.layer.{i}." in n]
        assigned.update(n for n, _ in named)
        add(named, lr * decay ** (n_layers - 1 - i))
    rest = [(n, p) for n, p in enc.named_parameters() if n not in assigned]  # embeddings (+ misc)
    add(rest, lr * decay ** n_layers)
    add(list(model.head.named_parameters()), head_lr)
    return groups


def load_tokenizer(name: str, use_fast: bool = True):
    try:
        return AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    except Exception as e:  # some Vietnamese checkpoints only ship slow tokenizers
        print(f"fast tokenizer unavailable ({e.__class__.__name__}); falling back to slow")
        return AutoTokenizer.from_pretrained(name, use_fast=False)
