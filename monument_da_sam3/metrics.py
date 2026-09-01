"""Confusion-count-first valid-pixel metrics for two independent channels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
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
