#!/usr/bin/env python3
"""Backfill canonical pixel-accuracy metrics into completed training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from sam2_adapter.metrics import pixel_accuracy


REPORTING_SCRIPTS = Path.home() / ".codex" / "skills" / "training-output-reporting" / "scripts"


def reconstruct_counts(
    *, precision: float, recall: float, positive_pixels: int
) -> tuple[int, int, int]:
    """Reconstruct integer TP/FP/FN from stored metrics and fixed validation GT."""

    if positive_pixels <= 0:
        raise ValueError("positive_pixels must be positive")
    if not (math.isfinite(precision) and math.isfinite(recall)):
        raise ValueError("precision and recall must be finite")
    if not (0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0):
        raise ValueError("precision and recall must lie in [0, 1]")
    tp = round(recall * positive_pixels)
    fn = positive_pixels - tp
    if precision == 0.0:
        raise ValueError("cannot reconstruct false positives from zero precision")
    fp = round(tp * (1.0 - precision) / precision)
    return tp, fp, fn


def epoch_accuracies(
    rows: list[dict[str, str]], *, background_pixels: int, positive_pixels: int
) -> list[tuple[int, float]]:
    valid_pixels = background_pixels + positive_pixels
    results: list[tuple[int, float]] = []
    for row in rows:
        tp, fp, fn = reconstruct_counts(
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            positive_pixels=positive_pixels,
        )
        results.append((int(row["epoch"]), pixel_accuracy(tp, fp, fn, valid_pixels)))
    return results


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _add_accuracy_to_metric_file(path: Path, *, valid_pixels: int) -> float | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    micro = payload.get("tile_micro")
    if not isinstance(micro, dict) or not all(key in micro for key in ("tp", "fp", "fn")):
        return None
    accuracy = pixel_accuracy(
        int(micro["tp"]), int(micro["fp"]), int(micro["fn"]), valid_pixels
    )
    micro["maccuracy"] = accuracy
    _write_json(path, payload)
    return accuracy


def _backfill_per_image_csv(path: Path, *, dataset_root: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if "accuracy" not in fields:
        position = fields.index("iou") + 1 if "iou" in fields else len(fields)
        fields.insert(position, "accuracy")
    for row in rows:
        with Image.open(dataset_root / "masks" / row["image"]) as mask_image:
            mask = np.asarray(mask_image)
        valid_pixels = int(np.logical_or(mask == 0, mask == 1).sum())
        row["accuracy"] = str(
            pixel_accuracy(
                int(row["tp"]), int(row["fp"]), int(row["fn"]), valid_pixels
            )
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _run_script(name: str, arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(REPORTING_SCRIPTS / name), *arguments],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"{name} failed: {result.stderr.strip() or result.stdout.strip()}")


def backfill_fold(
    fold_dir: Path, *, rebuild_reports: bool, dataset_root_override: Path | None = None
) -> dict[str, Any]:
    dataset = _read_json(fold_dir / "config" / "dataset.json")
    pixel_counts = dataset["pixel_counts"]
    validation = pixel_counts["validation"]
    outer = pixel_counts["outer_test"]
    validation_valid = int(validation["background"]) + int(validation["foreground"])
    outer_valid = int(outer["background"]) + int(outer["foreground"])
    dataset_root = dataset_root_override or Path(dataset["dataset_root"])

    epochs_path = fold_dir / "metrics" / "epochs.csv"
    with epochs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    accuracies = epoch_accuracies(
        rows,
        background_pixels=int(validation["background"]),
        positive_pixels=int(validation["foreground"]),
    )
    if len(accuracies) != len(rows):
        raise RuntimeError(f"epoch reconstruction count mismatch: {fold_dir}")

    writer = SummaryWriter(
        log_dir=str(fold_dir / "tensorboard"), filename_suffix=".accuracy_backfill"
    )
    try:
        for epoch, accuracy in accuracies:
            writer.add_scalar("metrics/accuracy", accuracy, epoch)
        writer.flush()
    finally:
        writer.close()

    validation_accuracy = _add_accuracy_to_metric_file(
        fold_dir / "metrics" / "selected_validation_metrics.json",
        valid_pixels=validation_valid,
    )
    outer_accuracy = _add_accuracy_to_metric_file(
        fold_dir / "metrics" / "outer_test_metrics.json", valid_pixels=outer_valid
    )
    validation_image_count = _backfill_per_image_csv(
        fold_dir / "metrics" / "per_image_validation.csv", dataset_root=dataset_root
    )
    _backfill_per_image_csv(
        fold_dir / "metrics" / "outer_test_per_image.csv", dataset_root=dataset_root
    )
    summary_path = fold_dir / "metrics" / "experiment_summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        summary["selected_validation_accuracy"] = validation_accuracy
        summary["outer_test_accuracy"] = outer_accuracy
        summary["accuracy_definition"] = "(TP + TN) / non-ignored pixels"
        _write_json(summary_path, summary)

    if rebuild_reports:
        _run_script(
            "export_tensorboard_scalars.py",
            [
                "--logdir", str(fold_dir / "tensorboard"),
                "--scalars-out", str(fold_dir / "metrics" / "tensorboard_scalars.csv"),
                "--epochs-out", str(epochs_path),
            ],
        )
        _run_script(
            "export_tensorboard_images.py",
            ["--logdir", str(fold_dir / "tensorboard"), "--output-dir", str(fold_dir / "tensorboard" / "images")],
        )
        _run_script("build_training_report.py", ["--run-dir", str(fold_dir)])

    return {
        "fold_dir": str(fold_dir),
        "epoch_count": len(accuracies),
        "validation_valid_pixels": validation_valid,
        "outer_test_valid_pixels": outer_valid,
        "selected_validation_accuracy": validation_accuracy,
        "outer_test_accuracy": outer_accuracy,
        "validation_image_count": validation_image_count,
        "reconstruction_source": "stored precision/recall plus fixed validation GT pixel totals",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--no-rebuild-reports", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for fold_root in args.fold_root:
        root = fold_root.resolve()
        folds = sorted(path for path in root.glob("fold*") if path.is_dir())
        if len(folds) != 5:
            raise RuntimeError(f"expected five folds under {root}, found {len(folds)}")
        records.extend(
            backfill_fold(
                fold,
                rebuild_reports=not args.no_rebuild_reports,
                dataset_root_override=args.dataset_root.resolve() if args.dataset_root else None,
            )
            for fold in folds
        )

    by_experiment: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        fold_dir = Path(record["fold_dir"])
        experiment = fold_dir.parents[2]
        by_experiment.setdefault(experiment, []).append(record)
    for experiment, experiment_records in by_experiment.items():
        audit = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "accuracy_definition": "(TP + TN) / non-ignored pixels",
            "historical_epoch_method": "integer confusion counts reconstructed from stored precision/recall and fixed GT totals",
            "folds": experiment_records,
        }
        _write_json(experiment / "info" / "accuracy_backfill.json", audit)
    print(json.dumps({"fold_count": len(records), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
