"""Multi-concept qualitative artifacts layered on the shared reporting contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam2_adapter.reporting import (
    EpochReporter,
    RunLayout,
    binary_metric_row,
    finalize_reporting,
    write_json,
)


def save_concept_qualitative(
    layout: RunLayout,
    *,
    concept: str,
    image_id: str,
    input_rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    """Save Input/GT/Prediction/Overlay under qualitative/<concept>/<image>."""

    identifier = Path(image_id).stem
    destination = layout.qualitative / concept / identifier
    destination.mkdir(parents=True, exist_ok=True)
    input_rgb = input_rgb.astype(np.uint8, copy=False)
    target = target.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)
    gt = np.zeros_like(input_rgb)
    gt[target] = (244, 63, 94)
    pred = np.zeros_like(input_rgb)
    pred[prediction] = (6, 182, 212)
    overlay = input_rgb.astype(np.float32).copy()
    overlay[target] = 0.52 * overlay[target] + 0.48 * np.array((244, 63, 94), dtype=np.float32)
    overlay[prediction] = 0.52 * overlay[prediction] + 0.48 * np.array((6, 182, 212), dtype=np.float32)
    paths = {
        "input_path": destination / "input.png",
        "gt_path": destination / "gt.png",
        "prediction_path": destination / "prediction.png",
        "overlay_path": destination / "overlay.png",
    }
    Image.fromarray(input_rgb, mode="RGB").save(paths["input_path"])
    Image.fromarray(gt, mode="RGB").save(paths["gt_path"])
    Image.fromarray(pred, mode="RGB").save(paths["prediction_path"])
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(paths["overlay_path"])
    row = binary_metric_row(target, prediction)
    row.update({
        "image": f"{concept}/{image_id}",
        "split": "validation",
        "target_class": concept,
        **{key: path.relative_to(layout.root).as_posix() for key, path in paths.items()},
    })
    return row


__all__ = [
    "EpochReporter",
    "RunLayout",
    "finalize_reporting",
    "save_concept_qualitative",
    "write_json",
]
