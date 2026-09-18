#!/usr/bin/env python3
"""Evaluate three F1-selected SAM3-Adapter experts on Dataset115 expert views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dual_adapter_sam3.metrics import BoundaryConfusion
from sam2_adapter.h0_core import load_trainable_state_dict
from sam2_adapter.reporting import binary_metric_row
from sam2_adapter.runtime import _autocast
from sam3_adapter.sam3_adapter_model import Sam3AdapterModel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset115_filtered"
DEFAULT_RUN = (
    ROOT
    / "sam3_adapter/runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42"
)
DEFAULT_BASE_CHECKPOINT = ROOT / "segment-anything-3/checkpoints/sam3.pt"
EXPERTS = ("scratch_crack", "shrinkage_craquelure", "loss")
MODEL_INPUT_SIZE = 1008
SOURCE_SIZE = 512
THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 1
EXPECTED_IMAGES = 715
_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class ExpertItem:
    expert: str
    temple: str
    source_group: str
    tile: str
    image: Path
    mask: Path
    image_sha256: str
    mask_sha256: str


class ExpertDataset(Dataset[dict[str, Any]]):
    def __init__(self, items: Sequence[ExpertItem]) -> None:
        self.items = tuple(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        with Image.open(item.image) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(item.mask) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        if rgb.shape != (SOURCE_SIZE, SOURCE_SIZE, 3):
            raise ValueError(f"expected 512x512 RGB image: {item.image}")
        if mask.shape != (SOURCE_SIZE, SOURCE_SIZE) or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"expected 512x512 binary mask: {item.mask}")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
        return {
            "image": (image - _MEAN) / _STD,
            "rgb": torch.from_numpy(rgb.copy()),
            "target": torch.from_numpy(mask == 255),
            "temple": item.temple,
            "source_group": item.source_group,
            "tile": item.tile,
            "image_path": str(item.image),
            "mask_path": str(item.mask),
            "image_sha256": item.image_sha256,
            "mask_sha256": item.mask_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(dataset: Path, expert: str) -> tuple[list[ExpertItem], Path]:
    manifest = dataset / "expert_views" / expert / "manifest.csv"
    items: list[ExpertItem] = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["expert"] != expert:
                raise ValueError(f"expert mismatch in {manifest}: {row['expert']}")
            image = (dataset / row["image"]).resolve()
            mask = (dataset / row["mask"]).resolve()
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"missing Dataset115 image or mask: {image}, {mask}")
            items.append(
                ExpertItem(
                    expert=expert,
                    temple=row["temple"],
                    source_group=row["source_group"],
                    tile=row["tile"],
                    image=image,
                    mask=mask,
                    image_sha256=row["image_sha256"],
                    mask_sha256=row["mask_sha256"],
                )
            )
    if len(items) != EXPECTED_IMAGES:
        raise ValueError(f"{expert} manifest requires {EXPECTED_IMAGES} images, found {len(items)}")
    if len({item.tile for item in items}) != len(items):
        raise ValueError(f"duplicate tile identifiers in {manifest}")
    return items, manifest


def _load_checkpoint(path: Path, expert: str, base_hash: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version": 1,
        "expert": expert,
        "selection_metric": "validation_pixel_micro_f1",
        "selection_threshold": THRESHOLD,
        "base_checkpoint_sha256": base_hash,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} mismatch: {payload.get(key)!r} != {expected!r}")
    if not isinstance(payload.get("adaptation_state"), dict) or not payload["adaptation_state"]:
        raise ValueError(f"checkpoint has no adaptation state: {path}")
    return payload


def _mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask] = color
    return result


def _overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[target] = 0.52 * result[target] + 0.48 * np.asarray((244, 63, 94), dtype=np.float32)
    result[prediction] = 0.52 * result[prediction] + 0.48 * np.asarray((6, 182, 212), dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _write_images(
    output: Path,
    *,
    temple: str,
    source_group: str,
    tile: str,
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, str]:
    destination = output / "artifacts" / "qualitative" / temple / source_group / tile
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_path": destination / "input.png",
        "gt_path": destination / "gt.png",
        "prediction_path": destination / "prediction.png",
        "overlay_path": destination / "overlay.png",
    }
    Image.fromarray(rgb, mode="RGB").save(paths["input_path"])
    Image.fromarray(_mask_rgb(target, (244, 63, 94)), mode="RGB").save(paths["gt_path"])
    Image.fromarray(_mask_rgb(prediction, (6, 182, 212)), mode="RGB").save(paths["prediction_path"])
    Image.fromarray(_overlay(rgb, target, prediction), mode="RGB").save(paths["overlay_path"])
    mask_path = output / "mask_tiles" / temple / source_group / f"{tile}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(mask_path)
    relative = {key: path.relative_to(output).as_posix() for key, path in paths.items()}
    relative["mask_output_path"] = mask_path.relative_to(output).as_posix()
    return relative


def _region_summary(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


def _boundary_micro(summaries: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    matched_prediction = sum(int(summary["matched_prediction"]) for summary in summaries)
    prediction_total = sum(int(summary["prediction_total"]) for summary in summaries)
    matched_target = sum(int(summary["matched_target"]) for summary in summaries)
    target_total = sum(int(summary["target_total"]) for summary in summaries)
    precision = matched_prediction / prediction_total if prediction_total else 0.0
    recall = matched_target / target_total if target_total else 0.0
    return {
        "matched_prediction": matched_prediction,
        "prediction_total": prediction_total,
        "matched_target": matched_target,
        "target_total": target_total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tolerance_pixels": BOUNDARY_TOLERANCE,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _evaluate_expert(
    *,
    expert: str,
    dataset: Path,
    run: Path,
    output_root: Path,
    base_checkpoint: Path,
    base_hash: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    output = output_root / expert
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    items, manifest = _read_manifest(dataset, expert)
    checkpoint = run / "1fold" / expert / "fold0/artifacts/checkpoints/best.pt"
    checkpoint_hash = _sha256(checkpoint)
    payload = _load_checkpoint(checkpoint, expert, base_hash)
    model = Sam3AdapterModel(base_checkpoint, device=device, input_size=MODEL_INPUT_SIZE)
    load_trainable_state_dict(model, payload["adaptation_state"])
    model.eval()
    loader = DataLoader(
        ExpertDataset(items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, os.cpu_count() or 1),
        pin_memory=True,
    )
    rows: list[dict[str, Any]] = []
    aggregate_boundary = BoundaryConfusion.empty(device=device, tolerance=BOUNDARY_TOLERANCE)
    tp = fp = fn = 0
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                images: Tensor = batch["image"].to(device, non_blocking=True)
                targets: Tensor = batch["target"].to(device, non_blocking=True)
                with _autocast(device, args.amp):
                    logits = model(images)
                predictions = torch.sigmoid(logits[:, 0]) >= THRESHOLD
                valid = torch.ones_like(targets, dtype=torch.bool)
                aggregate_boundary.update(predictions, targets, valid)
                predictions_cpu = predictions.cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                for index in range(images.shape[0]):
                    target = targets_cpu[index]
                    prediction = predictions_cpu[index]
                    metrics = binary_metric_row(target, prediction)
                    tp += int(metrics["tp"])
                    fp += int(metrics["fp"])
                    fn += int(metrics["fn"])
                    image_boundary = BoundaryConfusion.empty(tolerance=BOUNDARY_TOLERANCE)
                    image_boundary.update(
                        torch.from_numpy(prediction).unsqueeze(0),
                        torch.from_numpy(target).unsqueeze(0),
                        torch.ones((1, SOURCE_SIZE, SOURCE_SIZE), dtype=torch.bool),
                    )
                    boundary = image_boundary.compute()
                    rgb = batch["rgb"][index].numpy().astype(np.uint8, copy=False)
                    paths = _write_images(
                        output,
                        temple=str(batch["temple"][index]),
                        source_group=str(batch["source_group"][index]),
                        tile=str(batch["tile"][index]),
                        rgb=rgb,
                        target=target,
                        prediction=prediction,
                    )
                    rows.append(
                        {
                            "expert": expert,
                            "temple": str(batch["temple"][index]),
                            "source_group": str(batch["source_group"][index]),
                            "tile": str(batch["tile"][index]),
                            "image": str(batch["image_path"][index]),
                            "ground_truth": str(batch["mask_path"][index]),
                            "image_sha256": str(batch["image_sha256"][index]),
                            "ground_truth_sha256": str(batch["mask_sha256"][index]),
                            **metrics,
                            "boundary_precision_at_1": boundary["precision"],
                            "boundary_recall_at_1": boundary["recall"],
                            "boundary_f1_at_1": boundary["f1"],
                            "boundary_matched_prediction": boundary["matched_prediction"],
                            "boundary_prediction_total": boundary["prediction_total"],
                            "boundary_matched_target": boundary["matched_target"],
                            "boundary_target_total": boundary["target_total"],
                            **paths,
                        }
                    )
                print(
                    f"{expert} batch {batch_index}/{len(loader)} ({len(rows)}/{len(items)})",
                    flush=True,
                )
    finally:
        del model
        torch.cuda.empty_cache()

    boundary = aggregate_boundary.compute()
    summary = {
        "expert": expert,
        "image_count": len(rows),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": int(payload["epoch"]),
        "selection_validation_f1": float(payload["best_validation_f1"]),
        "dataset_manifest": str(manifest.resolve()),
        "dataset_manifest_sha256": _sha256(manifest),
        "region": _region_summary(tp, fp, fn),
        "boundary_at_1": boundary,
    }
    _write_csv(output / "metrics" / "per_image.csv", rows)
    _write_json(output / "metrics" / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3-Adapter evaluation requires CUDA")
    dataset = args.dataset.resolve()
    run_root = args.run.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    output = (
        args.output.resolve()
        if args.output
        else run_root / "inference" / "dataset115_filtered"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_hash = _sha256(base_checkpoint)
    device = torch.device("cuda")
    summaries = [
        _evaluate_expert(
            expert=expert,
            dataset=dataset,
            run=run_root,
            output_root=output,
            base_checkpoint=base_checkpoint,
            base_hash=base_hash,
            args=args,
            device=device,
        )
        for expert in EXPERTS
    ]
    region_micro = _region_summary(
        sum(int(summary["region"]["tp"]) for summary in summaries),
        sum(int(summary["region"]["fp"]) for summary in summaries),
        sum(int(summary["region"]["fn"]) for summary in summaries),
    )
    boundary_summaries = [summary["boundary_at_1"] for summary in summaries]
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset),
        "run": str(run_root),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_hash,
        "model_input_size": MODEL_INPUT_SIZE,
        "metric_size": SOURCE_SIZE,
        "threshold": THRESHOLD,
        "boundary_definition": "8-neighbor foreground boundary; Chebyshev tolerance 1 pixel",
        "classes": summaries,
        "macro": {
            "f1": sum(float(summary["region"]["f1"]) for summary in summaries) / len(summaries),
            "boundary_f1_at_1": sum(float(summary["boundary_at_1"]["f1"]) for summary in summaries) / len(summaries),
        },
        "micro": {
            **region_micro,
            "boundary_at_1": _boundary_micro(boundary_summaries),
        },
    }
    _write_json(output / "metrics" / "summary.json", aggregate)
    _write_json(
        output / "config" / "evaluation.json",
        {
            **vars(args),
            "output": output,
            "experts": EXPERTS,
            "expected_images_per_expert": EXPECTED_IMAGES,
            "target_contract": {
                "scratch_crack": ["D-01", "D-11"],
                "shrinkage_craquelure": ["D-03", "D-04"],
                "loss": ["D-02"],
            },
        },
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
