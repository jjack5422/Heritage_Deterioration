"""Confusion-count-first valid-pixel metrics for two independent channels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MultilabelConfusion:
    tp: Tensor
    fp: Tensor
    fn: Tensor
    tn: Tensor

    @classmethod
    def empty(cls, *, device: torch.device | str = "cpu") -> "MultilabelConfusion":
        return cls(*(torch.zeros(2, dtype=torch.int64, device=device) for _ in range(4)))

    def update(self, logits: Tensor, targets: Tensor, valid_masks: Tensor, *, threshold: float = 0.5) -> None:
        if logits.shape != targets.shape or valid_masks.shape != targets.shape or logits.shape[1] != 2:
            raise ValueError("metrics require identical [B,2,H,W] tensors")
        prediction = logits.sigmoid() >= threshold
        truth = targets.bool()
        valid = valid_masks.bool()
        dims = (0, 2, 3)
        self.tp += (prediction & truth & valid).sum(dims)
        self.fp += (prediction & ~truth & valid).sum(dims)
        self.fn += (~prediction & truth & valid).sum(dims)
        self.tn += (~prediction & ~truth & valid).sum(dims)

    def merge(self, other: "MultilabelConfusion") -> None:
        for field in ("tp", "fp", "fn", "tn"):
            getattr(self, field).add_(getattr(other, field))

    def compute(self) -> dict[str, object]:
        def divide(numerator: Tensor, denominator: Tensor) -> Tensor:
            return torch.where(denominator > 0, numerator.double() / denominator.double(), torch.zeros_like(denominator, dtype=torch.double))

        precision = divide(self.tp, self.tp + self.fp)
        recall = divide(self.tp, self.tp + self.fn)
        f1 = divide(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        iou = divide(self.tp, self.tp + self.fp + self.fn)
        accuracy = divide(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn)
        names = ("crack_craquelure", "loss")
        per_class = {
            name: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "iou": float(iou[index]), "accuracy": float(accuracy[index]),
                "tp": int(self.tp[index]), "fp": int(self.fp[index]),
                "fn": int(self.fn[index]), "tn": int(self.tn[index]),
            }
            for index, name in enumerate(names)
        }
        macro = {key: float(values.mean()) for key, values in (
            ("precision", precision), ("recall", recall), ("f1", f1),
            ("iou", iou), ("accuracy", accuracy),
        )}
        return {"per_class": per_class, "macro": macro}


def _mask_boundary(mask: Tensor) -> Tensor:
    if mask.ndim != 3:
        raise ValueError("boundary masks must have shape [B,H,W]")
    foreground = mask.bool()
    touches_background = F.max_pool2d(
        (~foreground).float().unsqueeze(1), kernel_size=3, stride=1, padding=1
    )[:, 0] > 0
    return foreground & touches_background


@dataclass
class BoundaryConfusion:
    """Aggregate symmetric boundary matches for one binary class."""

    matched_prediction: Tensor
    prediction_total: Tensor
    matched_target: Tensor
    target_total: Tensor
    tolerance: int = 2

    @classmethod
    def empty(
        cls,
        *,
        device: torch.device | str = "cpu",
        tolerance: int = 2,
    ) -> "BoundaryConfusion":
        if tolerance < 0:
            raise ValueError("boundary tolerance must be non-negative")
        values = [torch.zeros((), dtype=torch.int64, device=device) for _ in range(4)]
        return cls(*values, tolerance=tolerance)

    def update(self, prediction: Tensor, target: Tensor, valid: Tensor) -> None:
        if prediction.shape != target.shape or valid.shape != target.shape:
            raise ValueError("prediction, target, and valid must have identical [B,H,W] shapes")
        valid = valid.bool()
        prediction_boundary = _mask_boundary(prediction.bool() & valid) & valid
        target_boundary = _mask_boundary(target.bool() & valid) & valid
        kernel = 2 * self.tolerance + 1
        prediction_neighborhood = F.max_pool2d(
            prediction_boundary.float().unsqueeze(1), kernel, stride=1, padding=self.tolerance
        )[:, 0].bool()
        target_neighborhood = F.max_pool2d(
            target_boundary.float().unsqueeze(1), kernel, stride=1, padding=self.tolerance
        )[:, 0].bool()
        self.matched_prediction += (prediction_boundary & target_neighborhood).sum()
        self.prediction_total += prediction_boundary.sum()
        self.matched_target += (target_boundary & prediction_neighborhood).sum()
        self.target_total += target_boundary.sum()

    def compute(self) -> dict[str, float | int]:
        prediction_total = int(self.prediction_total)
        target_total = int(self.target_total)
        matched_prediction = int(self.matched_prediction)
        matched_target = int(self.matched_target)
        precision = matched_prediction / prediction_total if prediction_total else 0.0
        recall = matched_target / target_total if target_total else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_prediction": matched_prediction,
            "prediction_total": prediction_total,
            "matched_target": matched_target,
            "target_total": target_total,
            "tolerance_pixels": self.tolerance,
        }
