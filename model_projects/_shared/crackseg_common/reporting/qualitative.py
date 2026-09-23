"""Per-image validation metrics and qualitative artifacts for one expert."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from crackseg_common.augment import IMAGENET_MEAN, IMAGENET_STD
from crackseg_common.evaluation import evaluate
from PIL import Image
from crackseg_common.thresholding import (
    ThresholdPolicy,
    calibrate_model_threshold,
    save_run_threshold_policy,
)
from crackseg_common.reporting.threshold_curves import (
    render_pr_roc_curve,
    write_tensorboard_curve,
)
from torch import nn
from torch.utils.data import DataLoader


PER_IMAGE_COLUMNS = (
    "image",
    "split",
    "target_class",
    "threshold",
    "f1",
    "precision",
    "recall",
    "iou",
    "tp",
    "fp",
    "fn",
    "gt_pixels",
    "pred_pixels",
    "valid_pixels",
    "false_positive_rate",
    "crack_f1",
    "craquelure_f1",
    "craquelure_to_crack",
    "crack_to_craquelure",
    "error_reason",
    "input_path",
    "gt_path",
    "prediction_path",
    "overlay_path",
)
FOREGROUND_RGB = np.array((255, 55, 80), dtype=np.uint8)
CLASS_RGB = np.array(((0, 0, 0), (255, 55, 80), (217, 70, 239)), dtype=np.uint8)
PREDICTION_RGB = np.array(((0, 0, 0), (56, 189, 248), (250, 204, 21)), dtype=np.uint8)


def score_image(prediction: np.ndarray, target: np.ndarray, ignore_value: int) -> dict[str, int | float | str]:
    valid = target != ignore_value
    foreground = target == 1
    predicted = prediction == 1
    tp = int(np.count_nonzero(valid & foreground & predicted))
    fp = int(np.count_nonzero(valid & ~foreground & predicted))
    fn = int(np.count_nonzero(valid & foreground & ~predicted))
    gt_pixels = tp + fn
    pred_pixels = tp + fp
    valid_pixels = int(np.count_nonzero(valid))
    negative_pixels = valid_pixels - gt_pixels
    false_positive_rate = fp / negative_pixels if negative_pixels else ""
    if gt_pixels == 0:
        f1 = precision = recall = iou = ""
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    error_reason = ""
    if gt_pixels > 0 and pred_pixels == 0:
        error_reason = "false_negative"
    elif gt_pixels == 0 and pred_pixels > 0:
        error_reason = "false_positive"
    elif gt_pixels > 0 and isinstance(f1, float) and f1 < 0.5:
        error_reason = "boundary_error"
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "valid_pixels": valid_pixels,
        "false_positive_rate": false_positive_rate,
        "error_reason": error_reason,
    }


def restore_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    rgb = (image.detach().cpu() * std + mean).clamp(0, 1)
    return (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    rendered = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rendered[mask == 1] = FOREGROUND_RGB
    return rendered


def overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32)
    result[target == 1] = result[target == 1] * 0.5 + FOREGROUND_RGB * 0.5
    predicted_colour = np.array((56, 189, 248), dtype=np.float32)
    result[prediction == 1] = result[prediction == 1] * 0.5 + predicted_colour * 0.5
    return result.round().astype(np.uint8)


def _multiclass_score(prediction: np.ndarray, target: np.ndarray, ignore_value: int) -> dict[str, Any]:
    valid = target != ignore_value
    scores: dict[int, dict[str, float | int | str]] = {}
    for class_id in (1, 2):
        positive = valid & (target == class_id)
        predicted = valid & (prediction == class_id)
        tp = int(np.count_nonzero(positive & predicted))
        fp = int(np.count_nonzero(~positive & predicted))
        fn = int(np.count_nonzero(positive & ~predicted))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else ""
        scores[class_id] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1}
    present_f1 = [float(item["f1"]) for item in scores.values() if item["f1"] != ""]
    tp = sum(int(item["tp"]) for item in scores.values())
    fp = sum(int(item["fp"]) for item in scores.values())
    fn = sum(int(item["fn"]) for item in scores.values())
    gt_pixels = int(np.count_nonzero(valid & (target > 0)))
    pred_pixels = int(np.count_nonzero(valid & (prediction > 0)))
    negative = int(np.count_nonzero(valid & (target == 0)))
    background_fp = int(np.count_nonzero(valid & (target == 0) & (prediction > 0)))
    error_reason = ""
    if gt_pixels > 0 and pred_pixels == 0:
        error_reason = "false_negative"
    elif gt_pixels == 0 and pred_pixels > 0:
        error_reason = "false_positive"
    elif present_f1 and np.mean(present_f1) < 0.5:
        error_reason = "boundary_error"
    return {
        "f1": float(np.mean(present_f1)) if present_f1 else "",
        "precision": tp / (tp + fp) if tp + fp else "",
        "recall": tp / (tp + fn) if tp + fn else "",
        "iou": tp / (tp + fp + fn) if tp + fp + fn else "",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "valid_pixels": int(np.count_nonzero(valid)),
        "false_positive_rate": background_fp / negative if negative else "",
        "crack_f1": scores[1]["f1"],
        "craquelure_f1": scores[2]["f1"],
        "craquelure_to_crack": int(np.count_nonzero(valid & (target == 2) & (prediction == 1))),
        "crack_to_craquelure": int(np.count_nonzero(valid & (target == 1) & (prediction == 2))),
        "error_reason": error_reason,
    }


def _save_multiclass_images(
    output_dir: Path,
    image: torch.Tensor,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = restore_image(image)
    target_rgb = CLASS_RGB[np.clip(target, 0, 2)]
    target_rgb[target == 255] = 0
    prediction_rgb = CLASS_RGB[np.clip(prediction, 0, 2)]
    combined = rgb.astype(np.float32)
    for class_id in (1, 2):
        selected = target == class_id
        combined[selected] = combined[selected] * 0.5 + CLASS_RGB[class_id] * 0.5
        selected = prediction == class_id
        combined[selected] = combined[selected] * 0.5 + PREDICTION_RGB[class_id] * 0.5
    paths = {
        "input_path": output_dir / "input.png",
        "gt_path": output_dir / "gt.png",
        "prediction_path": output_dir / "prediction.png",
        "overlay_path": output_dir / "overlay.png",
    }
    for key, array in (
        ("input_path", rgb),
        ("gt_path", target_rgb),
        ("prediction_path", prediction_rgb),
        ("overlay_path", combined.round().astype(np.uint8)),
    ):
        Image.fromarray(array, "RGB").save(paths[key])
    return {key: path.as_posix() for key, path in paths.items()}


def save_images(
    output_dir: Path,
    image: torch.Tensor,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_rgb = restore_image(image)
    saved = {
        "input_path": output_dir / "input.png",
        "gt_path": output_dir / "gt.png",
        "prediction_path": output_dir / "prediction.png",
        "overlay_path": output_dir / "overlay.png",
    }
    Image.fromarray(input_rgb, "RGB").save(saved["input_path"])
    Image.fromarray(mask_rgb(target), "RGB").save(saved["gt_path"])
    Image.fromarray(mask_rgb(prediction), "RGB").save(saved["prediction_path"])
    Image.fromarray(overlay(input_rgb, target, prediction), "RGB").save(saved["overlay_path"])
    return {name: path.as_posix() for name, path in saved.items()}


def ranked_validation_rows(
    rows: list[dict[str, Any]],
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return F1-ranked best and worst validation rows, excluding blank F1."""

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            score = float(row["f1"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(score):
            scored.append((score, row))
    scored.sort(key=lambda item: (item[0], str(item[1]["image"])))
    return [row for _, row in reversed(scored[-top_k:])], [row for _, row in scored[:top_k]]


def write_tensorboard_validation_images(
    writer: Any,
    output_root: Path,
    rows: list[dict[str, Any]],
    global_step: int,
    top_k: int = 20,
) -> None:
    """Embed best/worst four-panel validation images in this run's event file."""

    qualitative_root = (output_root / "artifacts" / "qualitative").resolve()
    best, worst = ranked_validation_rows(rows, top_k)
    for group, selected in (("best", best), ("worst", worst)):
        for rank, row in enumerate(selected, start=1):
            images = []
            for column in ("input_path", "gt_path", "prediction_path", "overlay_path"):
                path = (output_root / str(row[column])).resolve()
                if not path.is_file() or qualitative_root not in path.parents:
                    raise RuntimeError(f"TensorBoard image is outside qualitative artifacts: {path}")
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB")))
            image_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(row["image"])
            )
            writer.add_image(
                f"qualitative/{group}/{rank:02d}_{image_id}",
                np.concatenate(images, axis=1),
                global_step,
                dataformats="HWC",
            )
    writer.flush()


@torch.no_grad()
def write_validation_artifacts(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_root: Path,
    expert_name: str,
    ignore_value: int,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate each validation tile and retain the four report images per tile."""

    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].cpu().numpy()
        probabilities = F.softmax(model(images).float(), dim=1)[:, 1]
        predictions = (probabilities > threshold).long().cpu().numpy()
        for image, target, prediction, filename in zip(
            batch["image"], targets, predictions, batch["name"], strict=True
        ):
            image_id = Path(filename).stem
            relative_dir = Path("artifacts") / "qualitative" / image_id
            paths = save_images(output_root / relative_dir, image, target, prediction)
            row = {
                "image": image_id,
                "split": "validation",
                "target_class": expert_name,
                "threshold": threshold,
                **score_image(prediction, target, ignore_value),
                **{name: (relative_dir / Path(path).name).as_posix() for name, path in paths.items()},
            }
            rows.append(row)
    metrics_path = output_root / "metrics" / "per_image_validation.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PER_IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


@torch.no_grad()
def write_multiclass_validation_artifacts(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_root: Path,
    ignore_value: int,
) -> list[dict[str, Any]]:
    """Write joint crack/craquelure validation metrics and four-panel artifacts."""

    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].cpu().numpy()
        predictions = model(images).float().argmax(dim=1).cpu().numpy()
        for image, target, prediction, filename in zip(
            batch["image"], targets, predictions, batch["name"], strict=True
        ):
            image_id = Path(filename).stem
            relative_dir = Path("artifacts") / "qualitative" / image_id
            paths = _save_multiclass_images(
                output_root / relative_dir,
                image,
                target,
                prediction,
            )
            rows.append(
                {
                    "image": image_id,
                    "split": "validation",
                    "target_class": "crack_craquelure",
                    "threshold": "argmax",
                    **_multiclass_score(prediction, target, ignore_value),
                    **{
                        name: (relative_dir / Path(path).name).as_posix()
                        for name, path in paths.items()
                    },
                }
            )
    metrics_path = output_root / "metrics" / "per_image_validation.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PER_IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _checkpoint_metrics(epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    micro = metrics["tile_micro"]
    cross = metrics["cross_confusion"]["craquelure_to_crack"]
    return {
        "epoch": epoch,
        "panel_macro_loss": metrics["panel_macro_loss"],
        "macro_iou": micro["miou"],
        "macro_f1": micro["mf1"],
        "crack": micro["per_class"]["crack"],
        "craquelure": micro["per_class"]["craquelure"],
        "craquelure_to_crack_count": cross["count"],
        "craquelure_to_crack_rate": cross["rate"],
        "confusion_matrix": micro["confusion_matrix"],
    }


def write_checkpoint_comparison(
    path: Path,
    *,
    best_epoch: int,
    best_metrics: dict[str, Any],
    last_epoch: int,
    last_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Persist validation-only best-vs-last results and signed last-minus-best deltas."""

    best = _checkpoint_metrics(best_epoch, best_metrics)
    last = _checkpoint_metrics(last_epoch, last_metrics)
    fields = ("panel_macro_loss", "macro_iou", "macro_f1", "craquelure_to_crack_rate")
    result = {
        "scope": "validation_only",
        "selection_warning": "outer test is excluded from checkpoint comparison",
        "best": best,
        "last": last,
        "delta_last_minus_best": {name: last[name] - best[name] for name in fields},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = (
        "checkpoint",
        "epoch",
        "panel_macro_loss",
        "macro_iou",
        "macro_f1",
        "craquelure_to_crack_rate",
    )
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for checkpoint, values in (("best", best), ("last", last)):
            writer.writerow({"checkpoint": checkpoint, **{name: values[name] for name in fields[1:]}})
    return result


def prepare_joint_final_evaluation(
    *,
    best_path: Path,
    last_path: Path,
    model: nn.Module,
    criterion: nn.Module,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_root: Path,
    plan: Any,
    evaluate_outer_test: bool,
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    """Compare best/last on validation, then optionally touch outer test with best."""

    from crackseg_common.evaluation import evaluate

    last_data = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(last_data["model"])
    last_metrics = evaluate(
        model, validation_loader, criterion, device, plan.expert_name, plan.ignore_value
    )
    best_data = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_data["model"])
    best_metrics = evaluate(
        model, validation_loader, criterion, device, plan.expert_name, plan.ignore_value
    )
    write_checkpoint_comparison(
        output_root / "metrics" / "checkpoint_comparison.json",
        best_epoch=int(best_data["epoch"]),
        best_metrics=best_metrics,
        last_epoch=int(last_data["epoch"]),
        last_metrics=last_metrics,
    )
    rows = write_multiclass_validation_artifacts(
        model=model,
        loader=validation_loader,
        device=device,
        output_root=output_root,
        ignore_value=plan.ignore_value,
    )
    test_metrics = None
    if evaluate_outer_test:
        test_metrics = evaluate(
            model, test_loader, criterion, device, plan.expert_name, plan.ignore_value
        )
    return int(best_data["epoch"]), test_metrics, rows


def prepare_final_evaluation(
    *,
    best_path: Path,
    model: nn.Module,
    criterion: nn.Module,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_root: Path,
    plan: Any,
    writer: Any,
) -> tuple[int, Any, dict[str, Any], list[dict[str, Any]]]:
    """Calibrate on validation, freeze the policy, then touch the outer test once."""

    best_data = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_data["model"])
    policy = calibrate_model_threshold(
        model=model,
        loader=validation_loader,
        device=device,
        expert=plan.expert_name,
        foreground_id=plan.expert_id,
        ignore_value=plan.ignore_value,
        validation={"outer_fold": plan.outer_fold, "inner_fold": plan.inner_fold},
    )
    save_run_threshold_policy(policy, output_root)
    curve_path = render_pr_roc_curve(
        policy, output_root / "tensorboard" / "images" / "pr_roc_curve.png"
    )
    write_tensorboard_curve(writer, curve_path, int(best_data["epoch"]))
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        plan.expert_name,
        plan.ignore_value,
        policy.threshold,
    )
    rows = write_validation_artifacts(
        model=model,
        loader=validation_loader,
        device=device,
        output_root=output_root,
        expert_name=plan.expert_name,
        ignore_value=plan.ignore_value,
        threshold=policy.threshold,
    )
    return int(best_data["epoch"]), policy, test_metrics, rows


def prepare_fixed_binary_final_evaluation(
    *,
    best_path: Path,
    model: nn.Module,
    criterion: nn.Module,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_root: Path,
    plan: Any,
    threshold: float = 0.5,
) -> tuple[int, ThresholdPolicy, dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the merged binary contract at its fixed, model-independent threshold."""

    best_data = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_data["model"])
    policy = ThresholdPolicy.default(plan.expert_name, plan.expert_id, threshold)
    policy.save(output_root / "config" / "threshold_policy.json")
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        plan.expert_name,
        plan.ignore_value,
        threshold,
    )
    rows = write_validation_artifacts(
        model=model,
        loader=validation_loader,
        device=device,
        output_root=output_root,
        expert_name=plan.expert_name,
        ignore_value=plan.ignore_value,
        threshold=threshold,
    )
    return int(best_data["epoch"]), policy, test_metrics, rows
