"""Batch-independent validation metrics for binary deterioration experts."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from crackseg_common.metrics import ConfusionMeter
from crackseg_common.data_plan import panel_name
from crackseg_common.thresholding import INFERENCE_RULE
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class PanelStats:
    """Sufficient statistics for loss and expert metrics across one panel."""

    ce_numerator: float = 0.0
    ce_denominator: float = 0.0
    intersection: float = 0.0
    probability_sum: float = 0.0
    target_sum: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))

    def add(self, other: PanelStats) -> None:
        self.ce_numerator += other.ce_numerator
        self.ce_denominator += other.ce_denominator
        self.intersection += other.intersection
        self.probability_sum += other.probability_sum
        self.target_sum += other.target_sum
        self.confusion += other.confusion

    def composite_loss(self, ce_weight: float, dice_weight: float) -> float:
        ce = self.ce_numerator / max(self.ce_denominator, 1e-12)
        dice = 1 - (2 * self.intersection + 1) / (
            self.probability_sum + self.target_sum + 1
        )
        return ce_weight * ce + dice_weight * dice


@dataclass
class MulticlassPanelStats:
    ce_numerator: float = 0.0
    ce_denominator: float = 0.0
    intersection: np.ndarray = field(default_factory=lambda: np.zeros(3))
    probability_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cost_numerator: float = 0.0
    cost_denominator: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.int64))

    def add(self, other: MulticlassPanelStats) -> None:
        self.ce_numerator += other.ce_numerator
        self.ce_denominator += other.ce_denominator
        self.intersection += other.intersection
        self.probability_sum += other.probability_sum
        self.target_sum += other.target_sum
        self.cost_numerator += other.cost_numerator
        self.cost_denominator += other.cost_denominator
        self.confusion += other.confusion

    def loss_components(self) -> dict[str, float]:
        ce = self.ce_numerator / max(self.ce_denominator, 1e-12)
        dice = 1 - np.mean(
            (2 * self.intersection[1:] + 1)
            / (self.probability_sum[1:] + self.target_sum[1:] + 1)
        )
        cost = self.cost_numerator / max(self.cost_denominator, 1)
        return {"ce": float(ce), "dice": float(dice), "cost": float(cost)}

    def composite_loss(self, criterion: nn.Module) -> float:
        parts = self.loss_components()
        ce, dice, cost = parts["ce"], parts["dice"], parts["cost"]
        return criterion.ce_weight * ce + criterion.dice_weight * dice + criterion.cost_weight * cost


def binary_metrics(counts: np.ndarray) -> dict[str, float | int | None]:
    tp, fp, fn, _ = counts
    gt_pixels = int(tp + fn)
    if tp + fp + fn == 0:
        return {"iou": None, "dice": None, "gt_pixels": gt_pixels}
    eps = 1e-6
    return {
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "gt_pixels": gt_pixels,
    }


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _multiclass_cross_confusion(confusion: np.ndarray) -> dict[str, dict[str, float | int]]:
    def item(row: int, column: int) -> dict[str, float | int]:
        count = int(confusion[row, column])
        total = int(confusion[row].sum())
        return {"count": count, "rate": count / total if total else 0.0, "gt_pixels": total}

    return {
        "craquelure_to_crack": item(2, 1),
        "crack_to_craquelure": item(1, 2),
        "craquelure_to_background": item(2, 0),
    }


@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ignore_value: int,
) -> dict[str, Any]:
    model.eval()
    class_names = ("background", "crack", "craquelure")
    overall = ConfusionMeter(3)
    panels: dict[str, MulticlassPanelStats] = defaultdict(MulticlassPanelStats)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = F.softmax(logits.float(), dim=1)
        predictions = probabilities.argmax(dim=1)
        overall.update(predictions, targets, ignore_value=ignore_value)
        for index, tile in enumerate(batch["name"]):
            target = targets[index]
            valid = target != ignore_value
            safe = torch.where(valid, target, torch.zeros_like(target))
            probability = probabilities[index] * valid.unsqueeze(0)
            one_hot = F.one_hot(safe, 3).permute(2, 0, 1).to(probability.dtype)
            one_hot *= valid.unsqueeze(0)
            ce_map = F.nll_loss(
                F.log_softmax(logits[index], dim=0).unsqueeze(0),
                target.unsqueeze(0),
                weight=criterion.class_weights,
                ignore_index=ignore_value,
                reduction="none",
            )[0]
            if criterion.class_weights is None:
                ce_denominator = valid.sum()
            else:
                ce_denominator = (criterion.class_weights[safe] * valid).sum()
            craquelure = valid & (target == 2)
            cost_numerator = (
                probability[1][craquelure].sum()
                * criterion.craquelure_to_crack_cost
                / 10.0
            )
            target_valid = target[valid]
            prediction_valid = predictions[index][valid]
            confusion = torch.bincount(
                target_valid * 3 + prediction_valid,
                minlength=9,
            ).reshape(3, 3)
            panels[panel_name(tile)].add(
                MulticlassPanelStats(
                    ce_numerator=float(ce_map.sum()),
                    ce_denominator=float(ce_denominator),
                    intersection=(probability * one_hot).sum((1, 2)).cpu().numpy(),
                    probability_sum=probability.sum((1, 2)).cpu().numpy(),
                    target_sum=one_hot.sum((1, 2)).cpu().numpy(),
                    cost_numerator=float(cost_numerator),
                    cost_denominator=int(craquelure.sum()),
                    confusion=confusion.cpu().numpy(),
                )
            )
    combined = MulticlassPanelStats()
    for stats in panels.values():
        combined.add(stats)
    panel_losses = {panel: stats.composite_loss(criterion) for panel, stats in panels.items()}
    tile_micro = overall.compute(class_names, ignore_index=0)
    panel_ious = []
    for stats in panels.values():
        cm = stats.confusion.astype(float)
        denominator = cm.sum(1) + cm.sum(0) - np.diag(cm)
        present = denominator[1:] > 0
        panel_ious.append(float(np.mean(np.diag(cm)[1:][present] / denominator[1:][present])))
    return {
        "loss": combined.composite_loss(criterion),
        "pixel_micro_loss": combined.composite_loss(criterion),
        "panel_macro_loss": float(np.mean(list(panel_losses.values()))),
        "worst_panel_loss": max(panel_losses.values()),
        "per_panel_loss": panel_losses,
        "tile_micro": tile_micro,
        "expert_panel_macro": {"iou": float(np.mean(panel_ious)), "panel_count": len(panels)},
        "cross_confusion": _multiclass_cross_confusion(overall.cm),
        "loss_components": combined.loss_components(),
        "inference_rule": "softmax_argmax",
        "threshold": None,
    }


def batch_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None,
    ignore_value: int,
    threshold: float = 0.5,
) -> np.ndarray:
    valid = target != ignore_value
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    log_probabilities = F.log_softmax(logits, dim=1)
    ce_map = F.nll_loss(
        log_probabilities,
        target,
        weight=class_weights,
        ignore_index=ignore_value,
        reduction="none",
    )
    if class_weights is None:
        ce_denominator = valid.sum((1, 2))
    else:
        ce_denominator = (class_weights[safe_target] * valid).sum((1, 2))
    probability = log_probabilities.exp()[:, 1]
    foreground = target == 1
    prediction = probability > threshold
    values = torch.stack(
        [
            ce_map.sum((1, 2)),
            ce_denominator,
            (probability * foreground * valid).sum((1, 2)),
            (probability * valid).sum((1, 2)),
            (foreground * valid).sum((1, 2)),
            (prediction & foreground & valid).sum((1, 2)),
            (prediction & ~foreground & valid).sum((1, 2)),
            (~prediction & foreground & valid).sum((1, 2)),
            (~prediction & ~foreground & valid).sum((1, 2)),
        ],
        dim=1,
    )
    return values.double().cpu().numpy()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    expert_name: str,
    ignore_value: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate exact pixel-micro and equal-panel-macro composite losses."""
    if getattr(criterion, "num_classes", 2) == 3:
        return evaluate_multiclass(model, loader, criterion, device, ignore_value)
    model.eval()
    class_names = ("background", expert_name)
    overall = ConfusionMeter(2)
    panels: dict[str, PanelStats] = defaultdict(PanelStats)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        predictions = (F.softmax(logits.float(), dim=1)[:, 1] > threshold).long()
        overall.update(predictions, masks, ignore_value=ignore_value)
        stats_rows = batch_stats(logits, masks, criterion.class_weights, ignore_value, threshold)
        for tile, row in zip(batch["name"], stats_rows, strict=True):
            panels[panel_name(tile)].add(
                PanelStats(
                    ce_numerator=row[0],
                    ce_denominator=row[1],
                    intersection=row[2],
                    probability_sum=row[3],
                    target_sum=row[4],
                    confusion=row[5:].astype(np.int64),
                )
            )

    combined = PanelStats()
    for stats in panels.values():
        combined.add(stats)
    panel_losses = {
        panel: stats.composite_loss(criterion.ce_weight, criterion.dice_weight)
        for panel, stats in panels.items()
    }
    pixel_micro_loss = combined.composite_loss(criterion.ce_weight, criterion.dice_weight)
    panel_macro_loss = float(np.mean(list(panel_losses.values())))
    expert_panel = [binary_metrics(stats.confusion) for stats in panels.values()]
    expert_macro = {
        key: finite_mean([metrics[key] for metrics in expert_panel])
        for key in ("iou", "dice")
    }
    expert_macro["panel_count"] = len(expert_panel)
    expert_macro["positive_panel_count"] = sum(
        int(metrics["gt_pixels"] > 0) for metrics in expert_panel
    )
    expert_macro["scored_panel_count"] = sum(
        int(metrics["iou"] is not None) for metrics in expert_panel
    )
    return {
        "threshold": threshold,
        "inference_rule": INFERENCE_RULE,
        "loss": pixel_micro_loss,
        "pixel_micro_loss": pixel_micro_loss,
        "panel_macro_loss": panel_macro_loss,
        "worst_panel_loss": max(panel_losses.values()),
        "per_panel_loss": panel_losses,
        "tile_micro": overall.compute(class_names, ignore_index=0),
        "expert_panel_macro": expert_macro,
    }
