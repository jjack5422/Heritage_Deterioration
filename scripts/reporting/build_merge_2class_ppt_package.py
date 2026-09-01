"""Build the evidence tables and qualitative asset package for the merge 2-class PPT brief.

This script is read-only with respect to training runs.  It normalizes the six
approved nested five-fold experiments, audits their reporting artifacts, and
copies one ranked validation Best/Worst composite per model into a compact
presentation package.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = WORKSPACE_ROOT / "model_report" / "merge_2class_ppt_package"
ASSET_DIR = OUTPUT_DIR / "assets"

REQUIRED_SCALAR_TAGS = {
    "loss/train",
    "loss/validation",
    "metrics/f1",
    "metrics/precision",
    "metrics/recall",
    "metrics/iou",
    "metrics/accuracy",
    "optimizer/lr",
}

RUNS = (
    {
        "model_key": "convnext_unet",
        "model_name": "ConvNeXt-Large U-Net",
        "family": "CNN encoder-decoder",
        "run": "unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_convnext-large_seed42",
        "expert": "foreground",
    },
    {
        "model_key": "resunet50",
        "model_name": "ResUNet-50",
        "family": "CNN encoder-decoder",
        "run": "unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_resunet50_seed42",
        "expert": "foreground",
    },
    {
        "model_key": "segformer_b5",
        "model_name": "SegFormer-B5",
        "family": "Transformer segmentation",
        "run": "segformer/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_segformer-b5_seed42",
        "expert": "foreground",
    },
    {
        "model_key": "sam2_adapter",
        "model_name": "SAM2 Adapter (Hiera-L)",
        "family": "Foundation model adapter",
        "run": "sam2_adapter/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42",
        "expert": "foreground",
    },
    {
        "model_key": "sam2_sac",
        "model_name": "SAM2-SAC (Hiera-L)",
        "family": "Foundation model + SAC",
        "run": "sam2_sac/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-sac-hiera-large_seed42",
        "expert": "foreground",
    },
    {
        "model_key": "sam3_adapter",
        "model_name": "SAM3 Adapter (512)",
        "family": "Foundation model adapter",
        "run": "sam3_adapter/runs/2026-08-28_sam3-adapter-512_seed42",
        "expert": "sam3_adapter",
    },
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def float_or_none(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_image_id(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"^\d+_", "", stem)
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", stem).lower()


def partition_values(dataset: dict, key: str) -> list[str]:
    split = dataset.get("split")
    source = split if isinstance(split, dict) else dataset
    aliases = {
        "train": ("train",),
        "validation": ("validation", "val"),
        "outer_test": ("outer_test", "test"),
    }
    for alias in aliases[key]:
        values = source.get(alias)
        if isinstance(values, list):
            return sorted(str(item) for item in values)
    return []


def partition_signature(dataset: dict) -> str:
    payload = {
        key: partition_values(dataset, key)
        for key in ("train", "validation", "outer_test")
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def fold_directory(run_dir: Path, expert: str, fold: int) -> Path:
    return run_dir / "5fold" / expert / f"fold{fold}"


def extract_counts(metrics: dict) -> tuple[int, int, int, int | None]:
    tile_micro = metrics.get("tile_micro")
    if not isinstance(tile_micro, dict):
        raise ValueError("outer-test metrics do not contain tile_micro")

    direct = tuple(tile_micro.get(key) for key in ("tp", "fp", "fn"))
    if all(isinstance(value, (int, float)) for value in direct):
        return int(direct[0]), int(direct[1]), int(direct[2]), None

    matrix = tile_micro.get("confusion_matrix")
    if (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        tn = int(matrix[0][0])
        fp = int(matrix[0][1])
        fn = int(matrix[1][0])
        tp = int(matrix[1][1])
        return tp, fp, fn, tn
    raise ValueError("could not extract binary confusion counts")


def extract_fold_metrics(metrics: dict) -> dict[str, float | int | None]:
    tile_micro = metrics["tile_micro"]
    tp, fp, fn, tn = extract_counts(metrics)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    accuracy = float_or_none(tile_micro.get("maccuracy"))
    if accuracy is None:
        accuracy = float_or_none(tile_micro.get("pixel_accuracy"))
    if accuracy is None and tn is not None:
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    panel = metrics.get("expert_panel_macro")
    panel_f1 = None
    if isinstance(panel, dict):
        panel_f1 = float_or_none(panel.get("f1"))
        if panel_f1 is None:
            panel_f1 = float_or_none(panel.get("dice"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "panel_macro_f1": panel_f1,
        "loss": float_or_none(metrics.get("loss")),
        "threshold": float_or_none(metrics.get("threshold")),
        "selected_epoch": metrics.get("selected_epoch"),
    }


def selected_epoch(fold_dir: Path, metrics: dict) -> int | None:
    direct = metrics.get("selected_epoch")
    if isinstance(direct, int):
        return direct
    summary_path = fold_dir / "metrics" / "experiment_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        for key in ("selected_epoch", "best_epoch"):
            value = summary.get(key)
            if isinstance(value, int):
                return value
    return None


def broken_report_links(report_path: Path) -> int:
    if not report_path.exists():
        return -1
    content = report_path.read_text(encoding="utf-8", errors="replace")
    refs = re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", content)
    broken = 0
    for raw_ref in refs:
        ref = html.unescape(unquote(raw_ref)).split("#", 1)[0].split("?", 1)[0]
        if not ref or ref.startswith(("http://", "https://", "data:", "mailto:", "#")):
            continue
        if not (report_path.parent / ref).resolve().exists():
            broken += 1
    return broken


def audit_fold(fold_dir: Path) -> dict[str, object]:
    required = (
        "config/args.json",
        "config/dataset.json",
        "config/environment.json",
        "config/run.json",
        "metrics/epochs.csv",
        "metrics/tensorboard_scalars.csv",
        "metrics/per_image_validation.csv",
        "metrics/outer_test_metrics.json",
        "metrics/experiment_summary.json",
        "artifacts/checkpoints/best.pt",
        "artifacts/checkpoints/last.pt",
        "tensorboard/images/loss_curve.png",
        "tensorboard/images/manifest.csv",
        "reports/index.html",
        "reports/best_20.html",
        "reports/worst_20.html",
    )
    missing = [relative for relative in required if not (fold_dir / relative).exists()]

    scalar_tags: set[str] = set()
    scalar_path = fold_dir / "metrics" / "tensorboard_scalars.csv"
    if scalar_path.exists():
        scalar_tags = {row.get("tag", "") for row in read_csv(scalar_path)}
    missing_scalar_tags = sorted(REQUIRED_SCALAR_TAGS - scalar_tags)

    per_image_columns: set[str] = set()
    per_image_path = fold_dir / "metrics" / "per_image_validation.csv"
    if per_image_path.exists():
        rows = read_csv(per_image_path)
        if rows:
            per_image_columns = set(rows[0])
    missing_per_image_columns = sorted(
        {"image", "f1", "accuracy", "input_path", "gt_path", "prediction_path", "overlay_path"}
        - per_image_columns
    )

    manifest_broken_paths = 0
    manifest_path = fold_dir / "tensorboard" / "images" / "manifest.csv"
    if manifest_path.exists():
        for row in read_csv(manifest_path):
            relative = row.get("path", "")
            if not relative or not (manifest_path.parent / relative).resolve().exists():
                manifest_broken_paths += 1

    report_broken_links = sum(
        max(0, broken_report_links(fold_dir / "reports" / name))
        for name in ("index.html", "best_20.html", "worst_20.html")
    )

    run_config_path = fold_dir / "config" / "run.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    selection_metric = str(run_config.get("selection_metric", ""))
    selection_uses_validation = selection_metric.startswith("val_") or "validation" in selection_metric

    strict_complete = not any(
        (missing, missing_scalar_tags, missing_per_image_columns, manifest_broken_paths, report_broken_links)
    ) and selection_uses_validation
    return {
        "missing_files": missing,
        "missing_scalar_tags": missing_scalar_tags,
        "missing_per_image_columns": missing_per_image_columns,
        "manifest_broken_paths": manifest_broken_paths,
        "report_broken_links": report_broken_links,
        "selection_metric": selection_metric,
        "selection_uses_validation": selection_uses_validation,
        "strict_reporting_complete": strict_complete,
    }


def candidate_assets(model: dict, run_dir: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for fold in range(5):
        fold_dir = fold_directory(run_dir, str(model["expert"]), fold)
        per_image_path = fold_dir / "metrics" / "per_image_validation.csv"
        manifest_path = fold_dir / "tensorboard" / "images" / "manifest.csv"
        if not per_image_path.exists() or not manifest_path.exists():
            continue
        image_rows = {
            normalize_image_id(row.get("image", "")): row
            for row in read_csv(per_image_path)
            if row.get("image")
        }
        for manifest_row in read_csv(manifest_path):
            group = manifest_row.get("group", "")
            if group not in {"best", "worst"}:
                continue
            tag_tail = manifest_row.get("tag", "").rsplit("/", 1)[-1]
            image_row = image_rows.get(normalize_image_id(tag_tail))
            if image_row is None:
                continue
            f1 = float_or_none(image_row.get("f1"))
            gt_pixels = float_or_none(image_row.get("gt_pixels"))
            if f1 is None or gt_pixels is None or gt_pixels < 1000:
                continue
            source_path = manifest_path.parent / manifest_row.get("path", "")
            if not source_path.exists():
                continue
            candidates.append(
                {
                    "model_key": model["model_key"],
                    "model_name": model["model_name"],
                    "fold": fold,
                    "group": group,
                    "rank": int(manifest_row.get("rank", 999)),
                    "image": image_row.get("image", ""),
                    "f1": f1,
                    "precision": float_or_none(image_row.get("precision")),
                    "recall": float_or_none(image_row.get("recall")),
                    "iou": float_or_none(image_row.get("iou")),
                    "accuracy": float_or_none(image_row.get("accuracy")),
                    "gt_pixels": int(gt_pixels),
                    "pred_pixels": int(float_or_none(image_row.get("pred_pixels")) or 0),
                    "error_reason": image_row.get("error_reason", ""),
                    "source_path": source_path,
                }
            )
    return candidates


def choose_assets(model: dict, run_dir: Path) -> list[dict[str, object]]:
    candidates = candidate_assets(model, run_dir)
    selected: list[dict[str, object]] = []
    best = [item for item in candidates if item["group"] == "best"]
    worst = [item for item in candidates if item["group"] == "worst"]
    if best:
        selected.append(max(best, key=lambda item: (float(item["f1"]), int(item["gt_pixels"]))))
    if worst:
        selected.append(min(worst, key=lambda item: (float(item["f1"]), -int(item["gt_pixels"]))))
    return selected


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    names = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, object]] = []
    split_signatures: dict[int, str] = {}
    manifest_hashes: set[str] = set()

    for model in RUNS:
        run_dir = WORKSPACE_ROOT / str(model["run"])
        per_model_folds: list[dict[str, object]] = []
        selected_epochs: list[int] = []
        all_split_matches = True

        for fold in range(5):
            fold_dir = fold_directory(run_dir, str(model["expert"]), fold)
            dataset = read_json(fold_dir / "config" / "dataset.json")
            manifest_hash = str(dataset.get("manifest_hash", ""))
            manifest_hashes.add(manifest_hash)
            signature = partition_signature(dataset)
            reference = split_signatures.setdefault(fold, signature)
            split_match = signature == reference
            all_split_matches = all_split_matches and split_match

            outer_metrics = read_json(fold_dir / "metrics" / "outer_test_metrics.json")
            metrics = extract_fold_metrics(outer_metrics)
            epoch = selected_epoch(fold_dir, outer_metrics)
            if epoch is not None:
                selected_epochs.append(epoch)
            metrics["selected_epoch"] = epoch
            row = {
                "model_key": model["model_key"],
                "model_name": model["model_name"],
                "fold": fold,
                "run": model["run"],
                "manifest_hash": manifest_hash,
                "split_signature": signature,
                "split_match_reference": split_match,
                "train_tiles": len(partition_values(dataset, "train")),
                "validation_tiles": len(partition_values(dataset, "validation")),
                "outer_test_tiles": len(partition_values(dataset, "outer_test")),
                **metrics,
            }
            fold_rows.append(row)
            per_model_folds.append(row)

            audit = audit_fold(fold_dir)
            audit_rows.append(
                {
                    "model_key": model["model_key"],
                    "model_name": model["model_name"],
                    "fold": fold,
                    "run_dir": fold_dir.relative_to(WORKSPACE_ROOT).as_posix(),
                    **audit,
                    "missing_files": "; ".join(audit["missing_files"]),
                    "missing_scalar_tags": "; ".join(audit["missing_scalar_tags"]),
                    "missing_per_image_columns": "; ".join(audit["missing_per_image_columns"]),
                }
            )

        tp = sum(int(row["tp"]) for row in per_model_folds)
        fp = sum(int(row["fp"]) for row in per_model_folds)
        fn = sum(int(row["fn"]) for row in per_model_folds)
        fold_f1 = [float(row["f1"]) for row in per_model_folds]
        fold_iou = [float(row["iou"]) for row in per_model_folds]
        accuracies = [float(row["accuracy"]) for row in per_model_folds if row["accuracy"] is not None]
        panel_f1 = [
            float(row["panel_macro_f1"])
            for row in per_model_folds
            if row["panel_macro_f1"] is not None
        ]
        model_rows.append(
            {
                "model_key": model["model_key"],
                "model_name": model["model_name"],
                "family": model["family"],
                "run": model["run"],
                "folds": len(per_model_folds),
                "outer_test_tiles": sum(int(row["outer_test_tiles"]) for row in per_model_folds),
                "pooled_precision": safe_div(tp, tp + fp),
                "pooled_recall": safe_div(tp, tp + fn),
                "pooled_f1": safe_div(2 * tp, 2 * tp + fp + fn),
                "pooled_iou": safe_div(tp, tp + fp + fn),
                "mean_fold_f1": statistics.fmean(fold_f1),
                "population_sd_fold_f1": statistics.pstdev(fold_f1),
                "worst_fold_f1": min(fold_f1),
                "best_fold_f1": max(fold_f1),
                "f1_range": max(fold_f1) - min(fold_f1),
                "mean_fold_iou": statistics.fmean(fold_iou),
                "mean_fold_accuracy": statistics.fmean(accuracies) if accuracies else None,
                "mean_panel_macro_f1": statistics.fmean(panel_f1) if panel_f1 else None,
                "selected_epochs": ",".join(str(value) for value in selected_epochs),
                "manifest_hash": per_model_folds[0]["manifest_hash"],
                "all_splits_match_reference": all_split_matches,
                "strict_reporting_folds": sum(
                    bool(row["strict_reporting_complete"])
                    for row in audit_rows
                    if row["model_key"] == model["model_key"]
                ),
            }
        )

        for selected in choose_assets(model, run_dir):
            group = str(selected["group"])
            filename = (
                f"{model['model_key']}_{group}_fold{selected['fold']}_"
                f"f1_{float(selected['f1']):.3f}.png"
            )
            destination = ASSET_DIR / filename
            shutil.copy2(Path(selected["source_path"]), destination)
            asset_rows.append(
                {
                    **selected,
                    "source_path": Path(selected["source_path"]).relative_to(WORKSPACE_ROOT).as_posix(),
                    "asset_path": destination.relative_to(WORKSPACE_ROOT).as_posix(),
                    "panel_order": "Input | GT | Prediction | Overlay",
                }
            )

    model_rows.sort(key=lambda row: float(row["pooled_f1"]), reverse=True)
    for rank, row in enumerate(model_rows, start=1):
        row["pooled_f1_rank"] = rank

    write_csv(
        OUTPUT_DIR / "model_metrics.csv",
        model_rows,
        (
            "pooled_f1_rank", "model_key", "model_name", "family", "folds",
            "outer_test_tiles", "pooled_precision", "pooled_recall", "pooled_f1",
            "pooled_iou", "mean_fold_f1", "population_sd_fold_f1",
            "worst_fold_f1", "best_fold_f1", "f1_range", "mean_fold_iou",
            "mean_fold_accuracy", "mean_panel_macro_f1", "selected_epochs",
            "manifest_hash", "all_splits_match_reference", "strict_reporting_folds", "run",
        ),
    )
    write_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        fold_rows,
        (
            "model_key", "model_name", "fold", "outer_test_tiles", "precision", "recall",
            "f1", "iou", "accuracy", "panel_macro_f1", "loss", "selected_epoch",
            "tp", "fp", "fn", "tn", "manifest_hash", "split_signature",
            "split_match_reference", "train_tiles", "validation_tiles", "run",
        ),
    )
    write_csv(
        OUTPUT_DIR / "reporting_audit.csv",
        audit_rows,
        (
            "model_key", "model_name", "fold", "strict_reporting_complete",
            "selection_metric", "selection_uses_validation", "missing_files",
            "missing_scalar_tags", "missing_per_image_columns", "manifest_broken_paths",
            "report_broken_links", "run_dir",
        ),
    )
    write_csv(
        OUTPUT_DIR / "asset_manifest.csv",
        asset_rows,
        (
            "model_key", "model_name", "group", "fold", "rank", "image", "f1",
            "precision", "recall", "iou", "accuracy", "gt_pixels", "pred_pixels",
            "error_reason", "panel_order", "asset_path", "source_path",
        ),
    )

    snapshot = {
        "scope": "formal merge 2-class runs only",
        "models": model_rows,
        "folds": fold_rows,
        "reporting_audit": audit_rows,
        "assets": asset_rows,
        "quality_checks": {
            "manifest_hash_count": len(manifest_hashes),
            "manifest_hashes": sorted(manifest_hashes),
            "all_models_have_five_folds": all(int(row["folds"]) == 5 for row in model_rows),
            "all_models_match_reference_splits": all(
                bool(row["all_splits_match_reference"]) for row in model_rows
            ),
            "all_thresholds_are_0_5": all(
                row["threshold"] == 0.5 for row in fold_rows
            ),
            "selected_asset_count": len(asset_rows),
        },
    }
    (OUTPUT_DIR / "analysis_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(snapshot["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
