"""Shared threshold policy and validation-only calibration for binary experts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INFERENCE_RULE = "foreground_probability > threshold"


@dataclass(frozen=True)
class ThresholdMetrics:
    """Pixel-micro validation metrics at one threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    iou: float
    tp: int
    fp: int
    fn: int
    constraint_met: bool = True
    tn: int = 0
    fpr: float = 0.0


@dataclass(frozen=True)
class ThresholdPolicy:
    """Frozen inference rule for one deterioration expert."""

    expert: str
    foreground_id: int
    selected_preset: str
    source: str
    presets: dict[str, ThresholdMetrics]
    validation: dict[str, Any]
    schema_version: int = 1
    inference_rule: str = INFERENCE_RULE

    @property
    def threshold(self) -> float:
        return self.presets[self.selected_preset].threshold

    @classmethod
    def default(cls, expert: str, foreground_id: int, threshold: float = 0.5) -> ThresholdPolicy:
        metric = ThresholdMetrics(threshold, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, False)
        return cls(
            expert=expert,
            foreground_id=foreground_id,
            selected_preset="balanced",
            source="default",
            presets={name: metric for name in ("sensitive", "balanced", "clean")},
            validation={},
        )

    def with_manual_threshold(self, threshold: float) -> ThresholdPolicy:
        _validate_threshold(threshold)
        manual = ThresholdMetrics(float(threshold), 0.0, 0.0, 0.0, 0.0, 0, 0, 0, False)
        return replace(
            self,
            selected_preset="manual",
            source="manual",
            presets={**self.presets, "manual": manual},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ThresholdPolicy:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("inference_rule") != INFERENCE_RULE:
            raise ValueError(f"不支援的 threshold policy: {path}")
        presets = {
            name: ThresholdMetrics(**values) for name, values in data["presets"].items()
        }
        policy = cls(
            expert=str(data["expert"]),
            foreground_id=int(data["foreground_id"]),
            selected_preset=str(data["selected_preset"]),
            source=str(data["source"]),
            presets=presets,
            validation=dict(data.get("validation", {})),
        )
        _validate_threshold(policy.threshold)
        return policy


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold 必須介於 0 與 1: {threshold}")


def load_foreground_probability(probability: np.ndarray) -> np.ndarray:
    """Return the foreground plane from HxW or 2xHxW cached probabilities."""

    array = np.asarray(probability)
    if array.ndim == 3 and array.shape[0] == 2:
        array = array[1]
    if array.ndim != 2:
        raise ValueError(f"probability 必須是 HxW 或 2xHxW，收到 {array.shape}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("probability 含 NaN 或 infinite")
    if array.size and (float(array.min()) < -1e-6 or float(array.max()) > 1.000001):
        raise ValueError("probability 必須介於 0 與 1")
    return np.clip(array, 0.0, 1.0)


def apply_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the one canonical binary-expert inference rule."""

    _validate_threshold(threshold)
    return (load_foreground_probability(probability) > threshold).astype(np.uint8)


class ThresholdSweep:
    """Accumulate exact confusion counts for a small fixed threshold grid."""

    def __init__(self, thresholds: np.ndarray | None = None) -> None:
        values = (
            np.round(np.arange(0.05, 0.951, 0.01), 2)
            if thresholds is None
            else np.asarray(thresholds, dtype=np.float64)
        )
        if values.ndim != 1 or values.size == 0 or np.any(np.diff(values) <= 0):
            raise ValueError("thresholds 必須是非空且嚴格遞增的一維陣列")
        for value in values:
            _validate_threshold(float(value))
        self.thresholds = values
        self.tp = np.zeros(values.size, dtype=np.int64)
        self.fp = np.zeros(values.size, dtype=np.int64)
        self.positive_pixels = 0
        self.negative_pixels = 0
        self.image_count = 0

    def update(
        self,
        probability: np.ndarray,
        target: np.ndarray,
        foreground_id: int = 1,
        ignore_value: int = 255,
    ) -> None:
        probability = load_foreground_probability(probability)
        target = np.asarray(target)
        if target.shape != probability.shape:
            raise ValueError(f"probability/target 尺寸不同: {probability.shape} / {target.shape}")
        valid = target != ignore_value
        foreground = target == foreground_id
        positive = probability[valid & foreground]
        negative = probability[valid & ~foreground]
        self.tp += _counts_at_or_above(positive, self.thresholds)
        self.fp += _counts_at_or_above(negative, self.thresholds)
        self.positive_pixels += int(positive.size)
        self.negative_pixels += int(negative.size)
        self.image_count += 1

    def metrics(self) -> list[ThresholdMetrics]:
        result = []
        for threshold, tp, fp in zip(self.thresholds, self.tp, self.fp, strict=True):
            fn = self.positive_pixels - int(tp)
            tn = self.negative_pixels - int(fp)
            precision = int(tp) / (int(tp) + int(fp)) if tp + fp else 0.0
            recall = int(tp) / self.positive_pixels if self.positive_pixels else 0.0
            fpr = int(fp) / self.negative_pixels if self.negative_pixels else 0.0
            f1 = 2 * int(tp) / (2 * int(tp) + int(fp) + fn) if 2 * tp + fp + fn else 0.0
            iou = int(tp) / (int(tp) + int(fp) + fn) if tp + fp + fn else 0.0
            result.append(
                ThresholdMetrics(
                    float(threshold), precision, recall, f1, iou,
                    int(tp), int(fp), fn, True, tn, fpr,
                )
            )
        return result

    def build_policy(
        self,
        expert: str,
        foreground_id: int,
        precision_floor: float = 0.9,
        recall_floor: float = 0.9,
        selected_preset: str = "balanced",
        validation: dict[str, Any] | None = None,
    ) -> ThresholdPolicy:
        if self.image_count == 0:
            raise ValueError("沒有 validation probability 可校正")
        if self.positive_pixels == 0:
            raise ValueError("validation 沒有此 expert 的正樣本，無法校正 threshold")
        _validate_threshold(precision_floor)
        _validate_threshold(recall_floor)
        metrics = self.metrics()
        balanced = max(metrics, key=lambda item: (item.f1, item.precision, item.threshold))
        clean = _select_constrained(
            metrics,
            predicate=lambda item: item.precision >= precision_floor and item.tp + item.fp > 0,
            key=lambda item: (item.recall, item.f1, item.precision, item.threshold),
            fallback_key=lambda item: (item.precision, item.recall, item.threshold),
        )
        sensitive = _select_constrained(
            metrics,
            predicate=lambda item: item.recall >= recall_floor and self.positive_pixels > 0,
            key=lambda item: (item.precision, item.f1, item.threshold),
            fallback_key=lambda item: (item.recall, item.precision, item.threshold),
        )
        presets = {"sensitive": sensitive, "balanced": balanced, "clean": clean}
        if selected_preset not in presets:
            raise ValueError(f"未知 preset: {selected_preset}")
        details = dict(validation or {})
        details.update(
            {
                "split": "validation",
                "aggregation": "pixel_micro",
                "image_count": self.image_count,
                "positive_pixels": self.positive_pixels,
                "negative_pixels": self.negative_pixels,
                "precision_floor": precision_floor,
                "recall_floor": recall_floor,
                "threshold_metrics": [asdict(item) for item in metrics],
            }
        )
        return ThresholdPolicy(
            expert=expert,
            foreground_id=foreground_id,
            selected_preset=selected_preset,
            source="validation_calibrated",
            presets=presets,
            validation=details,
        )


def _counts_at_or_above(probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    crossed = np.searchsorted(thresholds, probability, side="left")
    histogram = np.bincount(crossed, minlength=thresholds.size + 1)
    return np.cumsum(histogram[::-1], dtype=np.int64)[::-1][1:]


def _select_constrained(metrics, predicate, key, fallback_key) -> ThresholdMetrics:
    candidates = [item for item in metrics if predicate(item)]
    if candidates:
        return max(candidates, key=key)
    return replace(max(metrics, key=fallback_key), constraint_met=False)


def calibrate_validation_directory(
    *,
    probability_dir: str | Path,
    mask_dir: str | Path,
    split_plan_path: str | Path,
    thresholds: np.ndarray | None = None,
    precision_floor: float = 0.9,
    recall_floor: float = 0.9,
    selected_preset: str = "balanced",
) -> ThresholdPolicy:
    """Calibrate from the explicit ``val`` list; train/test entries are ignored."""

    probability_dir, mask_dir = Path(probability_dir), Path(mask_dir)
    split_plan_path = Path(split_plan_path)
    plan = json.loads(split_plan_path.read_text(encoding="utf-8"))
    try:
        expert = str(plan["expert"]["name"])
        foreground_id = int(plan["expert"]["source_class_id"])
        ignore_value = int(plan["ignore_value"])
        validation_names = tuple(plan["val"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"split_plan 缺少 expert/ignore_value/val: {split_plan_path}") from exc
    if not validation_names:
        raise ValueError("split_plan 的 validation 清單為空")
    sweep = ThresholdSweep(thresholds)
    for filename in validation_names:
        probability_path = probability_dir / f"{Path(filename).stem}.npy"
        mask_path = mask_dir / filename
        if not probability_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"validation 缺少 probability/mask: {filename}")
        probability = np.load(probability_path, allow_pickle=False)
        with Image.open(mask_path) as image:
            target = np.asarray(image)
        if target.ndim == 3:
            target = target[..., 0]
        sweep.update(probability, target, foreground_id, ignore_value)
    return sweep.build_policy(
        expert=expert,
        foreground_id=foreground_id,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
        selected_preset=selected_preset,
        validation={
            "split_plan": str(split_plan_path.resolve()),
            "probability_dir": str(probability_dir.resolve()),
            "mask_dir": str(mask_dir.resolve()),
        },
    )


def calibrate_model_threshold(
    *,
    model: Any,
    loader: Any,
    device: Any,
    expert: str,
    foreground_id: int,
    ignore_value: int,
    validation: dict[str, Any] | None = None,
) -> ThresholdPolicy:
    """Stream one binary validation loader without retaining its probability maps."""

    import torch
    import torch.nn.functional as functional

    sweep = ThresholdSweep()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = functional.softmax(model(images).float(), dim=1)[:, 1].cpu().numpy()
            targets = batch["mask"].cpu().numpy()
            for probability, target in zip(probabilities, targets, strict=True):
                sweep.update(probability, target, foreground_id=1, ignore_value=ignore_value)
    return sweep.build_policy(
        expert=expert,
        foreground_id=foreground_id,
        validation=validation,
    )


def save_run_threshold_policy(policy: ThresholdPolicy, output_root: str | Path) -> None:
    """Write the policy, sweep table and selected rule into one standard run."""

    output_root = Path(output_root)
    policy.save(output_root / "config" / "threshold_policy.json")
    write_threshold_metrics(policy, output_root / "metrics" / "threshold_metrics.csv")
    run_path = output_root / "config" / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run.update(
            {
                "inference_rule": policy.inference_rule,
                "threshold": policy.threshold,
                "threshold_source": policy.source,
                "threshold_policy": "config/threshold_policy.json",
            }
        )
        run_path.write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def write_threshold_metrics(policy: ThresholdPolicy, path: str | Path) -> Path:
    rows = policy.validation.get("threshold_metrics", [])
    if not rows:
        raise ValueError("policy 沒有 validation threshold metrics")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="只用 split_plan 的 validation 清單校正 binary expert threshold。"
    )
    result.add_argument("--probability-dir", required=True)
    result.add_argument("--mask-dir", required=True)
    result.add_argument("--split-plan", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--precision-floor", type=float, default=0.9)
    result.add_argument("--recall-floor", type=float, default=0.9)
    result.add_argument(
        "--preset", choices=("sensitive", "balanced", "clean"), default="balanced"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    policy = calibrate_validation_directory(
        probability_dir=args.probability_dir,
        mask_dir=args.mask_dir,
        split_plan_path=args.split_plan,
        precision_floor=args.precision_floor,
        recall_floor=args.recall_floor,
        selected_preset=args.preset,
    )
    output_dir = Path(args.output_dir)
    policy_path = policy.save(output_dir / "threshold_policy.json")
    metrics_path = write_threshold_metrics(policy, output_dir / "threshold_metrics.csv")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from crackseg_common.reporting.threshold_curves import render_pr_roc_curve

    curve_path = render_pr_roc_curve(policy, output_dir / "pr_roc_curve.png")
    print(
        f"expert={policy.expert} preset={policy.selected_preset} "
        f"threshold={policy.threshold:.2f}\npolicy={policy_path}\n"
        f"metrics={metrics_path}\ncurve={curve_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
