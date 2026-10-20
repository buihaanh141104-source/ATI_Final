"""Multi-label losses. `bce` (default, as required by the brief) + optional variants
for the imbalance extension (RQ3): positive-weighted BCE, focal, asymmetric (ASL)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Lin et al. 2017, sigmoid form."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * (1 - p_t) ** self.gamma
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021 (ASL): separate focusing for negatives + probability shifting."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        los_pos = targets * torch.log(p.clamp(min=self.eps)) * (1 - p) ** self.gamma_pos
        los_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg ** self.gamma_neg
        return -(los_pos + los_neg).sum(1).mean() / targets.shape[1]


def build_loss(cfg: dict, y_train: np.ndarray) -> nn.Module:
    name = (cfg or {}).get("name", "bce")
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_posweight":
        n_pos = y_train.sum(0).clip(min=1)
        pw = (len(y_train) - n_pos) / n_pos
        pw = pw ** cfg.get("power", 1.0)  # power=0.5 -> sqrt re-weighting (milder)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32))
    if name == "focal":
        return FocalLoss(cfg.get("gamma", 2.0), cfg.get("alpha"))
    if name == "asl":
        return AsymmetricLoss(cfg.get("gamma_neg", 4.0), cfg.get("gamma_pos", 0.0), cfg.get("clip", 0.05))
    raise ValueError(f"unknown loss {name}")
