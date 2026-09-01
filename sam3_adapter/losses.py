"""Locked loss contract shared by A, B, and D."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def weighted_bce_dice_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_value: int,
    positive_weight: float = 2.0,
    dice_weight: float = 0.65,
    epsilon: float = 1e-6,
) -> Tensor:
    """Fixed 2:1 foreground-weighted BCE plus unweighted soft Dice."""

    if target.ndim == 3:
        target = target.unsqueeze(1)
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"target/logit shape mismatch: {tuple(target.shape)} vs {tuple(logits.shape)}")
    valid = target != ignore_value
    if not bool(valid.any()):
        return logits.sum() * 0.0
    valid_logits = logits[valid]
    valid_target = target[valid].to(logits.dtype)
    pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(valid_logits, valid_target, pos_weight=pos_weight)
    probability = torch.sigmoid(valid_logits)
    dice = 1.0 - (2.0 * (probability * valid_target).sum() + epsilon) / (
        probability.square().sum() + valid_target.square().sum() + epsilon
    )
    return bce + dice_weight * dice
