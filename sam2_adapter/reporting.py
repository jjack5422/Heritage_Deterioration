"""Required durable reporting outputs for each H0 stage/fold run."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


EPOCH_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "f1",
    "precision",
    "recall",
    "iou",
    "accuracy",
    "learning_rate",
)
PER_IMAGE_COLUMNS = (
    "image",
    "split",
    "target_class",
    "f1",
    "precision",
    "recall",
    "iou",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "gt_pixels",
    "pred_pixels",
    "error_reason",
    "input_path",
    "gt_path",
    "prediction_path",
    "overlay_path",
)
REPORTING_SCRIPTS = Path.home() / ".codex" / "skills" / "training-output-reporting" / "scripts"


@dataclass(frozen=True)
class RunLayout:
    """Canonical per-fold output locations required by training-output-reporting."""

    root: Path
    config: Path
    logs: Path
    tensorboard: Path
    metrics: Path
    checkpoints: Path
    qualitative: Path
    reports: Path

    @classmethod
    def create(cls, root: str | Path) -> "RunLayout":
        root = Path(root).resolve()
        layout = cls(
            root=root,
            config=root / "config",
            logs=root / "logs",
            tensorboard=root / "tensorboard",
            metrics=root / "metrics",
            checkpoints=root / "artifacts" / "checkpoints",
            qualitative=root / "artifacts" / "qualitative",
            reports=root / "reports",
        )
        for directory in (
            layout.config,
            layout.logs,
            layout.tensorboard,
            layout.metrics,
            layout.checkpoints,
            layout.qualitative,
            layout.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return layout


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_log(layout: RunLayout, message: str) -> None:
    with (layout.logs / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


class EpochReporter:
    """Write the mandatory TensorBoard tags and canonical CSV rows."""

    def __init__(self, epoch_csv: Path, writer: Any) -> None:
        self.writer = writer
        self.handle = epoch_csv.open("w", encoding="utf-8", newline="")
        self.csv = csv.DictWriter(self.handle, fieldnames=EPOCH_COLUMNS)
        self.csv.writeheader()
        self.handle.flush()

    def record(
        self,
        epoch: int,
        *,
        train_loss: float,
        validation_loss: float,
        f1: float | None,
        precision: float | None,
        recall: float | None,
        iou: float | None,
        accuracy: float | None,
        learning_rate: float,
    ) -> dict[str, int | float | str]:
        row: dict[str, int | float | str] = {
            "epoch": epoch,
            "train_loss": _finite_or_blank(train_loss),
            "val_loss": _finite_or_blank(validation_loss),
            "f1": _finite_or_blank(f1),
            "precision": _finite_or_blank(precision),
            "recall": _finite_or_blank(recall),
            "iou": _finite_or_blank(iou),
            "accuracy": _finite_or_blank(accuracy),
            "learning_rate": _finite_or_blank(learning_rate),
        }
        for tag, column in (
            ("loss/train", "train_loss"),
            ("loss/validation", "val_loss"),
            ("metrics/f1", "f1"),
            ("metrics/precision", "precision"),
            ("metrics/recall", "recall"),
            ("metrics/iou", "iou"),
            ("metrics/accuracy", "accuracy"),
            ("optimizer/lr", "learning_rate"),
        ):
            value = row[column]
            if value != "":
                self.writer.add_scalar(tag, float(value), epoch)
        self.csv.writerow(row)
        self.handle.flush()
        self.writer.flush()
        return row

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def binary_metric_row(target: np.ndarray, prediction: np.ndarray) -> dict[str, int | float | str]:
    """Calculate an unambiguous foreground metric row for one binary label map."""

    target = target.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)
    tp = int(np.logical_and(target, prediction).sum())
    fp = int(np.logical_and(~target, prediction).sum())
    fn = int(np.logical_and(target, ~prediction).sum())
    gt_pixels = int(target.sum())
    pred_pixels = int(prediction.sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn) if gt_pixels else ""
    iou = _ratio(tp, tp + fp + fn) if gt_pixels else ""
    accuracy = _ratio(target.size - fp - fn, target.size)
    if gt_pixels and pred_pixels == 0:
        reason = "false_negative"
    elif not gt_pixels and pred_pixels:
        reason = "false_positive"
    elif gt_pixels and fn > fp and fn > 0:
        reason = "fragmentation"
    elif gt_pixels and (fp or fn):
        reason = "boundary_error"
    else:
        reason = ""
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "error_reason": reason,
    }


def save_qualitative_example(
    layout: RunLayout,
    *,
    image_id: str,
    input_rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    target_class: str,
) -> dict[str, int | float | str]:
    """Save Input / GT / Prediction / Overlay source PNGs and one metric row."""

    if input_rgb.ndim != 3 or input_rgb.shape[-1] != 3:
        raise ValueError("input_rgb must be HxWx3")
    if target.shape != input_rgb.shape[:2] or prediction.shape != input_rgb.shape[:2]:
        raise ValueError("target and prediction must match input image height/width")
    identifier = Path(image_id).stem
    destination = layout.qualitative / identifier
    destination.mkdir(parents=True, exist_ok=True)
    input_rgb = input_rgb.astype(np.uint8, copy=False)
    target_bool = target.astype(bool, copy=False)
    prediction_bool = prediction.astype(bool, copy=False)
    gt_rgb = _mask_rgb(target_bool, color=(244, 63, 94))
    prediction_rgb = _mask_rgb(prediction_bool, color=(6, 182, 212))
    overlay = _overlay_rgb(input_rgb, target_bool, prediction_bool)
    paths = {
        "input_path": destination / "input.png",
        "gt_path": destination / "gt.png",
        "prediction_path": destination / "prediction.png",
        "overlay_path": destination / "overlay.png",
    }
    Image.fromarray(input_rgb, mode="RGB").save(paths["input_path"])
    Image.fromarray(gt_rgb, mode="RGB").save(paths["gt_path"])
    Image.fromarray(prediction_rgb, mode="RGB").save(paths["prediction_path"])
    Image.fromarray(overlay, mode="RGB").save(paths["overlay_path"])
    result = binary_metric_row(target_bool, prediction_bool)
    result.update(
        {
            "image": image_id,
            "split": "validation",
            "target_class": target_class,
            **{key: path.relative_to(layout.root).as_posix() for key, path in paths.items()},
        }
    )
    return result


def write_validation_rows(layout: RunLayout, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist the full required per-validation-image table, in stable image order."""

    materialized = sorted((dict(row) for row in rows), key=lambda row: str(row["image"]))
    with (layout.metrics / "per_image_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return materialized


def _rankable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(_as_float(row.get("f1")), float)]


def _embed_ranked_composites(
    writer: Any,
    layout: RunLayout,
    rows: list[dict[str, Any]],
    *,
    selected_epoch: int,
    top_k: int = 20,
) -> None:
    candidates = _rankable(rows)
    if not candidates:
        raise RuntimeError("validation set has no rankable foreground images for qualitative reporting")
    ordered = sorted(candidates, key=lambda row: (_as_float(row["f1"]) or -1.0, str(row["image"])))
    groups = {"best": list(reversed(ordered[-top_k:])), "worst": ordered[:top_k]}
    for group, ranked in groups.items():
        for rank, row in enumerate(ranked, start=1):
            composite = _load_composite(layout, row)
            safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in Path(str(row["image"])).stem)
            writer.add_image(
                f"qualitative/{group}/{rank:02d}_{safe_id}",
                composite,
                selected_epoch,
                dataformats="HWC",
            )
    writer.flush()


def finalize_reporting(
    layout: RunLayout,
    *,
    writer: Any,
    reporter: EpochReporter,
    validation_rows: Iterable[dict[str, Any]],
    selected_epoch: int,
    outer_test_metrics: dict[str, Any],
) -> None:
    """Finish a run with TensorBoard exports and the required static dashboard."""

    rows = write_validation_rows(layout, validation_rows)
    write_json(layout.metrics / "outer_test_metrics.json", outer_test_metrics)
    try:
        _embed_ranked_composites(writer, layout, rows, selected_epoch=selected_epoch)
    finally:
        reporter.close()
        writer.close()
    _run_reporting_script(
        layout,
        "export_tensorboard_scalars.py",
        [
            "--logdir",
            str(layout.tensorboard),
            "--scalars-out",
            str(layout.metrics / "tensorboard_scalars.csv"),
            "--epochs-out",
            str(layout.metrics / "epochs.csv"),
        ],
    )
    _run_reporting_script(
        layout,
        "export_tensorboard_images.py",
        ["--logdir", str(layout.tensorboard), "--output-dir", str(layout.tensorboard / "images")],
    )
    _run_reporting_script(layout, "build_training_report.py", ["--run-dir", str(layout.root)])
    _validate_required_outputs(layout)


def _run_reporting_script(layout: RunLayout, name: str, arguments: list[str]) -> None:
    script = REPORTING_SCRIPTS / name
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    append_log(layout, result.stdout)
    append_log(layout, result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name} failed: {result.stderr.strip() or result.stdout.strip()}")


def _validate_required_outputs(layout: RunLayout) -> None:
    required = (
        layout.metrics / "epochs.csv",
        layout.metrics / "per_image_validation.csv",
        layout.metrics / "outer_test_metrics.json",
        layout.metrics / "experiment_summary.json",
        layout.tensorboard / "images" / "manifest.csv",
        layout.tensorboard / "images" / "loss_curve.png",
        layout.reports / "index.html",
        layout.reports / "best_20.html",
        layout.reports / "worst_20.html",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required reporting artifacts are missing: {missing}")
    html = (layout.reports / "index.html").read_text(encoding="utf-8")
    if "assets/" not in html:
        raise RuntimeError("report index has no copied qualitative image paths")


def _mask_rgb(mask: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = color
    return canvas


def _overlay_rgb(input_rgb: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    overlay = input_rgb.astype(np.float32).copy()
    # Pink-red = GT; cyan-blue = prediction; violet indicates overlap.
    overlay[target] = 0.52 * overlay[target] + 0.48 * np.array((244, 63, 94), dtype=np.float32)
    overlay[prediction] = 0.52 * overlay[prediction] + 0.48 * np.array((6, 182, 212), dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _load_composite(layout: RunLayout, row: dict[str, Any]) -> np.ndarray:
    panels = []
    for column in ("input_path", "gt_path", "prediction_path", "overlay_path"):
        with Image.open(layout.root / str(row[column])) as image:
            panels.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.concatenate(panels, axis=1)


def _ratio(numerator: int, denominator: int) -> float | str:
    return numerator / denominator if denominator else ""


def _as_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _finite_or_blank(value: float | None) -> float | str:
    number = _as_float(value)
    return number if number is not None else ""
