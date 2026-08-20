"""多類別語意分割指標 (含背景的 num_classes 全部統計, 但回報 mIoU 預設不含背景)。

使用方式:
    meter = ConfusionMeter(num_classes=5)
    for batch in loader:
        logits = model(batch["image"])
        pred = logits.argmax(1)               # [B,H,W]
        meter.update(pred, batch["mask"])     # 兩者皆 long tensor
    res = meter.compute(class_names=CLASS_NAMES, ignore_index=0)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_long(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().long().numpy()
    return np.asarray(x).astype(np.int64)


class ConfusionMeter:
    """累計 confusion matrix (rows=gt, cols=pred), shape [C, C]."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: ArrayLike, target: ArrayLike, ignore_value: Optional[int] = None):
        p = _to_numpy_long(pred).reshape(-1)
        t = _to_numpy_long(target).reshape(-1)
        if p.shape != t.shape:
            raise ValueError(f"pred/target shape mismatch: {p.shape} vs {t.shape}")
        valid = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        if ignore_value is not None:
            valid &= (t != ignore_value)
        if not valid.any():
            return
        idx = t[valid] * self.num_classes + p[valid]
        bins = np.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += bins.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self, eps: float = 1e-6) -> np.ndarray:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        return (tp + eps) / (tp + fp + fn + eps)

    def per_class_dice(self, eps: float = 1e-6) -> np.ndarray:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        return (2 * tp + eps) / (2 * tp + fp + fn + eps)

    def per_class_precision_recall(self, eps: float = 1e-6):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        return precision, recall

    def pixel_accuracy(self, eps: float = 1e-6) -> float:
        cm = self.cm.astype(np.float64)
        return float((np.diag(cm).sum() + eps) / (cm.sum() + eps))

    def class_pixel_counts(self) -> np.ndarray:
        """每類 ground-truth 像素數 (gt 維度 sum)。"""
        return self.cm.sum(axis=1).astype(np.int64)

    def compute(self,
                class_names: Optional[Sequence[str]] = None,
                ignore_index: Optional[int] = 0) -> Dict:
        """回傳含 per-class 與 macro 指標的 dict。

        ignore_index: 在 mIoU/mDice 等 macro 平均時忽略此類 (預設背景=0)。
                      仍會在 per_class 區段保留所有類別。
        """
        ious = self.per_class_iou()
        dices = self.per_class_dice()
        prec, rec = self.per_class_precision_recall()
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        counts = self.class_pixel_counts()

        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]

        per_class = {}
        for i, name in enumerate(class_names):
            per_class[name] = {
                "iou": float(ious[i]),
                "dice": float(dices[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "gt_pixels": int(counts[i]),
            }

        # macro 平均: 只對 gt 有出現的類做平均, 並依 ignore_index 排除
        present_mask = counts > 0
        avg_mask = present_mask.copy()
        if ignore_index is not None and 0 <= ignore_index < self.num_classes:
            avg_mask[ignore_index] = False

        def _mean(arr):
            if not avg_mask.any():
                return float("nan")
            return float(arr[avg_mask].mean())

        return {
            "per_class": per_class,
            "miou": _mean(ious),
            "mdice": _mean(dices),
            "mprecision": _mean(prec),
            "mrecall": _mean(rec),
            "mf1": _mean(f1),
            "pixel_accuracy": self.pixel_accuracy(),
            "ignore_index": ignore_index,
            "averaged_classes": [class_names[i] for i in range(self.num_classes) if avg_mask[i]],
            "missing_in_gt": [class_names[i] for i in range(self.num_classes) if not present_mask[i]],
            "confusion_matrix": self.cm.tolist(),
        }


def format_metrics(res: Dict, max_class_name: int = 12) -> str:
    """把 compute() 結果做成簡單表格字串, 方便 print/log。"""
    lines = []
    lines.append(f"{'class':<{max_class_name}}  {'IoU':>7} {'Dice':>7} {'Prec':>7} {'Rec':>7}  {'gt_px':>10}")
    for name, m in res["per_class"].items():
        lines.append(
            f"{name:<{max_class_name}}  "
            f"{m['iou']:7.4f} {m['dice']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f}  "
            f"{m['gt_pixels']:>10d}"
        )
    lines.append(
        f"{'macro(no_bg)':<{max_class_name}}  "
        f"{res['miou']:7.4f} {res['mdice']:7.4f} {res['mprecision']:7.4f} {res['mrecall']:7.4f}"
    )
    lines.append(f"pixel_acc = {res['pixel_accuracy']:.4f}  ignore_index={res['ignore_index']}")
    if res.get("missing_in_gt"):
        lines.append(f"missing_in_gt: {res['missing_in_gt']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 保留二元舊 API (zero-shot binary 預測流程仍會用)
# ---------------------------------------------------------------------------

def _binarize(x, thr=0.5):
    return (np.asarray(x) > thr).astype(np.uint8)


def confusion(pred, target, thr=0.5):
    p = _binarize(pred, thr)
    t = _binarize(target, thr)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    return tp, fp, fn, tn


def iou(pred, target, thr=0.5, eps=1e-6):
    tp, fp, fn, _ = confusion(pred, target, thr)
    return (tp + eps) / (tp + fp + fn + eps)


def dice(pred, target, thr=0.5, eps=1e-6):
    tp, fp, fn, _ = confusion(pred, target, thr)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def precision_recall_f1(pred, target, thr=0.5, eps=1e-6):
    tp, fp, fn, _ = confusion(pred, target, thr)
    p = (tp + eps) / (tp + fp + eps)
    r = (tp + eps) / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return p, r, f1


def evaluate(pred, target, thr=0.5):
    p, r, f1 = precision_recall_f1(pred, target, thr)
    return {
        "iou": iou(pred, target, thr),
        "dice": dice(pred, target, thr),
        "precision": p,
        "recall": r,
        "f1": f1,
    }
