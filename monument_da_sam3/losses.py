"""Masked two-class segmentation and DA-MoE router objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    segmentation: Tensor
    per_class: Tensor
    dice: Tensor
    focal: Tensor
    presence: Tensor
    balance: Tensor
    router_z: Tensor


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def masked_weighted_focal(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    positive_weight: float = 2.0,
    gamma: float = 2.0,
) -> Tensor:
    valid = valid.bool()
    if not valid.any():
        return logits.sum() * 0.0
    pos_weight = logits.new_tensor(positive_weight)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probability = logits.sigmoid()
    p_t = torch.where(targets.bool(), probability, 1.0 - probability)
    return (((1.0 - p_t) ** gamma) * bce)[valid].mean()


def masked_positive_dice(logits: Tensor, targets: Tensor, valid: Tensor, *, eps: float = 1e-6) -> Tensor:
    probability = logits.sigmoid()
    valid_float = valid.to(logits.dtype)
    target_valid = targets * valid_float
    present = target_valid.flatten(1).sum(1) > 0
    if not present.any():
        return logits.sum() * 0.0
    intersection = (probability * target_valid).flatten(1).sum(1)
    denominator = (probability * valid_float).flatten(1).sum(1) + target_valid.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps))[present].mean()


def switch_balance_loss(soft_probabilities: Tensor, hard_assignments: Tensor) -> Tensor:
    if soft_probabilities.shape != hard_assignments.shape:
        raise ValueError("soft and hard routing tensors must have identical shapes")
    experts = soft_probabilities.shape[-1]
    reduce_dims = tuple(range(soft_probabilities.ndim - 1))
    soft_usage = soft_probabilities.mean(dim=reduce_dims)
    hard_usage = hard_assignments.float().mean(dim=reduce_dims)
    return experts * (soft_usage * hard_usage).sum()


def router_z_loss(router_logits: Tensor) -> Tensor:
    return torch.logsumexp(router_logits, dim=-1).square().mean()


def multilabel_objective(
    logits: Tensor,
    presence_logits: Tensor,
    targets: Tensor,
    valid_masks: Tensor,
    class_present: Tensor,
    *,
    routing: list[tuple[Tensor, Tensor, Tensor]] | None = None,
    positive_weight: float = 2.0,
    gamma: float = 2.0,
    presence_weight: float = 0.1,
    balance_weight: float = 0.01,
    router_z_weight: float = 0.001,
) -> LossOutput:
    if logits.shape != targets.shape or valid_masks.shape != targets.shape:
        raise ValueError("logits, targets, and valid_masks must have identical [B,2,H,W] shapes")
    if logits.ndim != 4 or logits.shape[1] != 2 or presence_logits.shape != logits.shape[:2]:
        raise ValueError("approved objective requires two masks and two presence logits")
    for name, value in (("logits", logits), ("targets", targets), ("presence_logits", presence_logits)):
        _require_finite(name, value)
    dice_terms, focal_terms, presence_terms = [], [], []
    for class_index in range(2):
        dice_terms.append(masked_positive_dice(logits[:, class_index], targets[:, class_index], valid_masks[:, class_index]))
        focal_terms.append(masked_weighted_focal(
            logits[:, class_index], targets[:, class_index], valid_masks[:, class_index],
            positive_weight=positive_weight, gamma=gamma,
        ))
        presence_terms.append(F.binary_cross_entropy_with_logits(
            presence_logits[:, class_index], class_present[:, class_index].to(logits.dtype)
        ))
    dice = torch.stack(dice_terms)
    focal = torch.stack(focal_terms)
    presence = torch.stack(presence_terms)
    per_class = dice + focal + presence_weight * presence
    segmentation = per_class.mean()
    zero = logits.sum() * 0.0
    if routing:
        balances = [switch_balance_loss(soft, hard) for _, soft, hard in routing]
        z_terms = [router_z_loss(router_logits) for router_logits, _, _ in routing]
        balance = torch.stack(balances).mean()
        router_z = torch.stack(z_terms).mean()
    else:
        balance = zero
        router_z = zero
    total = segmentation + balance_weight * balance + router_z_weight * router_z
    _require_finite("total loss", total)
    return LossOutput(total, segmentation, per_class, dice, focal, presence, balance, router_z)
