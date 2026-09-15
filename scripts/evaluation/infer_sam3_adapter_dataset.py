#!/usr/bin/env python3
"""Run a trained 1008px SAM3-Adapter checkpoint on paired tile folders."""

from __future__ import annotations

import argparse
import csv
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

from sam2_adapter.h0_core import load_trainable_state_dict
from sam2_adapter.reporting import binary_metric_row
from sam2_adapter.runtime import _autocast, _sha256
from sam3_adapter.sam3_adapter_model import Sam3AdapterModel


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = WORKSPACE / "dataset"
DEFAULT_BASE_CHECKPOINT = WORKSPACE / "segment-anything-3" / "checkpoints" / "sam3.pt"
DEFAULT_ADAPTATION_CHECKPOINT = (
    WORKSPACE
    / "sam3_adapter"
    / "runs"
    / "2026-08-26_sam2-sam3-native-probe-adapter_seed42"
    / "5fold"
    / "sam3_adapter"
    / "fold1"
    / "artifacts"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "sam3_adapter"
    / "runs"
    / "2026-09-15_fold1-1008-dataset-inference-craquelure-only"
    / "1fold"
    / "sam3_adapter"
    / "fold1"
)
MODEL_INPUT_SIZE = 1008
SOURCE_SIZE = 512
THRESHOLD = 0.5
DEFAULT_TARGET_RAW_IDS = (4,)
_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class TilePair:
    source_group: str
    image: Path
    mask: Path


def discover_pairs(dataset: str | Path) -> tuple[TilePair, ...]:
    """Find and validate every image_tiles/mask_tiles pair."""

    root = Path(dataset).resolve()
    pairs: list[TilePair] = []
    for image in sorted(root.glob("*/*/image_tiles/*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask = image.parent.parent / "mask_tiles" / f"{image.stem}.png"
        if not mask.is_file():
            raise FileNotFoundError(f"missing mask for {image}: {mask}")
        pairs.append(TilePair(image.parent.parent.name, image.resolve(), mask.resolve()))
    if not pairs:
        raise FileNotFoundError(f"no paired tiles found under {root}")
    mask_stems = {
        (path.parent.parent.resolve(), path.stem)
        for path in root.glob("*/*/mask_tiles/*.png")
    }
    image_stems = {(pair.image.parent.parent.resolve(), pair.image.stem) for pair in pairs}
    extras = sorted(mask_stems - image_stems)
    if extras:
        raise ValueError(f"found {len(extras)} masks without images; first={extras[0]}")
    return tuple(pairs)


def make_target(mask: np.ndarray, *, raw_ids: Sequence[int]) -> np.ndarray:
    """Map the selected raw class IDs to a binary target."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2-D class-index image, got {mask.shape}")
    if not raw_ids:
        raise ValueError("at least one target raw ID is required")
    return np.isin(mask, tuple(raw_ids))


def _mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask] = color
    return output


def _overlay_rgb(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32).copy()
    output[target] = 0.52 * output[target] + 0.48 * np.array((244, 63, 94), dtype=np.float32)
    output[prediction] = 0.52 * output[prediction] + 0.48 * np.array((6, 182, 212), dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def compose_panels(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return Input | GT | Prediction | Overlay as one RGB image."""

    return np.concatenate(
        (
            image,
            _mask_rgb(target, (244, 63, 94)),
            _mask_rgb(prediction, (6, 182, 212)),
            _overlay_rgb(image, target, prediction),
        ),
        axis=1,
    )


class TileDataset(Dataset[dict[str, Any]]):
    def __init__(self, pairs: Sequence[TilePair]) -> None:
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        with Image.open(pair.image) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(pair.mask) as handle:
            mask = np.asarray(handle, dtype=np.uint8)
        if rgb.shape != (SOURCE_SIZE, SOURCE_SIZE, 3) or mask.shape != (SOURCE_SIZE, SOURCE_SIZE):
            raise ValueError(f"expected 512x512 image/mask: {pair.image}, {pair.mask}")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
        image = (image - _MEAN) / _STD
        return {
            "image": image,
            "raw_rgb": torch.from_numpy(rgb.copy()),
            "mask": torch.from_numpy(mask.copy()),
            "source_group": pair.source_group,
            "tile": pair.image.stem,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = (
        "source_group", "image", "f1", "precision", "recall", "iou", "accuracy",
        "tp", "fp", "fn", "gt_pixels", "pred_pixels", "error_reason", "composite_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    total = len(rows) * SOURCE_SIZE * SOURCE_SIZE
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "image_count": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "accuracy": (total - fp - fn) / total,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--adaptation-checkpoint", type=Path, default=DEFAULT_ADAPTATION_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-raw-ids", type=int, nargs="+", default=list(DEFAULT_TARGET_RAW_IDS))
    parser.add_argument(
        "--reuse-predictions-from",
        type=Path,
        help="Rebuild GT, overlay, composites, and metrics from an existing inference output.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _save_example(
    output: Path,
    *,
    source_group: str,
    tile: str,
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    destination = output / "artifacts" / "qualitative" / source_group / tile
    destination.mkdir(parents=True, exist_ok=True)
    panels = {
        "input.png": rgb,
        "gt.png": _mask_rgb(target, (244, 63, 94)),
        "prediction.png": _mask_rgb(prediction, (6, 182, 212)),
        "overlay.png": _overlay_rgb(rgb, target, prediction),
    }
    for name, array in panels.items():
        Image.fromarray(array, mode="RGB").save(destination / name)
    composite_path = output / "artifacts" / "composites" / source_group / f"{tile}.png"
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(compose_panels(rgb, target, prediction), mode="RGB").save(composite_path)
    return {
        "source_group": source_group,
        "image": tile,
        **binary_metric_row(target, prediction),
        "composite_path": composite_path.relative_to(output).as_posix(),
    }


def _finish_output(
    output: Path,
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    payload: dict[str, Any],
    *,
    base_hash: str,
) -> None:
    raw_ids = tuple(int(value) for value in args.target_raw_ids)
    labels = {1: "Crack", 4: "Craquelure"}
    names = ", ".join(f"{value} ({labels.get(value, 'raw class')})" for value in raw_ids)
    _write_csv(output / "metrics" / "per_image_inference.csv", rows)
    summary = {
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.adaptation_checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": base_hash,
        "model_input_size": MODEL_INPUT_SIZE,
        "source_and_output_size": SOURCE_SIZE,
        "threshold": THRESHOLD,
        "target_raw_ids": list(raw_ids),
        "target_rule": f"dataset raw ID(s) {names} are foreground",
        "prediction_source": str(args.reuse_predictions_from.resolve()) if args.reuse_predictions_from else "model inference",
        "panel_order": ["input", "gt", "prediction", "overlay"],
        "colors": {"gt": [244, 63, 94], "prediction": [6, 182, 212]},
        "metrics": _summary(rows),
    }
    _write_json(output / "config" / "inference.json", vars(args))
    _write_json(output / "metrics" / "summary.json", summary)


def _rebuild_from_predictions(
    args: argparse.Namespace,
    pairs: Sequence[TilePair],
    output: Path,
    payload: dict[str, Any],
    base_hash: str,
) -> Path:
    source = args.reuse_predictions_from.resolve()
    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs, start=1):
        with Image.open(pair.image) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(pair.mask) as handle:
            raw_mask = np.asarray(handle, dtype=np.uint8)
        prediction_path = source / "artifacts" / "qualitative" / pair.source_group / pair.image.stem / "prediction.png"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"missing reusable prediction: {prediction_path}")
        with Image.open(prediction_path) as handle:
            prediction_rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        prediction = prediction_rgb.any(axis=2)
        target = make_target(raw_mask, raw_ids=args.target_raw_ids)
        rows.append(_save_example(
            output,
            source_group=pair.source_group,
            tile=pair.image.stem,
            rgb=rgb,
            target=target,
            prediction=prediction,
        ))
        if index % 50 == 0 or index == len(pairs):
            print(f"rebuild {index}/{len(pairs)}", flush=True)
    _finish_output(output, args, rows, payload, base_hash=base_hash)
    return output


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    adaptation_checkpoint = args.adaptation_checkpoint.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    if not base_checkpoint.is_file() or not adaptation_checkpoint.is_file():
        raise FileNotFoundError("base or adaptation checkpoint is missing")
    pairs = discover_pairs(dataset)
    payload = torch.load(adaptation_checkpoint, map_location="cpu", weights_only=False)
    base_hash = _sha256(base_checkpoint)
    if payload.get("base_checkpoint_sha256") != base_hash:
        raise ValueError("adaptation checkpoint does not match the selected SAM3 base checkpoint")
    if args.reuse_predictions_from:
        return _rebuild_from_predictions(args, pairs, output, payload, base_hash)
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3-Adapter inference requires CUDA")

    device = torch.device("cuda")
    model = Sam3AdapterModel(base_checkpoint, device=device, input_size=MODEL_INPUT_SIZE)
    load_trainable_state_dict(model, payload["adaptation_state"])
    model.eval()
    loader = DataLoader(
        TileDataset(pairs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, os.cpu_count() or 1),
        pin_memory=True,
    )
    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images: Tensor = batch["image"].to(device, non_blocking=True)
            with _autocast(device, args.amp):
                logits = model(images)
            predictions = (torch.sigmoid(logits[:, 0]) >= THRESHOLD).cpu().numpy()
            for index in range(images.shape[0]):
                source_group = str(batch["source_group"][index])
                tile = str(batch["tile"][index])
                rgb = batch["raw_rgb"][index].numpy().astype(np.uint8, copy=False)
                raw_mask = batch["mask"][index].numpy()
                target = make_target(raw_mask, raw_ids=args.target_raw_ids)
                prediction = predictions[index]
                rows.append(_save_example(
                    output,
                    source_group=source_group,
                    tile=tile,
                    rgb=rgb,
                    target=target,
                    prediction=prediction,
                ))
            print(f"inference batch {batch_index}/{len(loader)} ({len(rows)}/{len(pairs)})", flush=True)

    _finish_output(output, args, rows, payload, base_hash=base_hash)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    destination = run(args)
    print(json.dumps({"status": "completed", "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
