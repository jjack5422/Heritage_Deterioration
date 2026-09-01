"""CPU-only dummy segmentation adapter for end-to-end validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from adapters.base import SegmentationAdapter
from imaging.image_processing import make_overlay


class DummyAdapter(SegmentationAdapter):
    """Segment pixels by grayscale intensity without a checkpoint."""

    def __init__(self) -> None:
        self.loaded = False

    def load(self, weight_path: Path | None) -> None:
        if weight_path is not None:
            raise ValueError("Dummy adapter does not accept a checkpoint")
        self.loaded = True

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        if not self.loaded:
            raise RuntimeError("Dummy adapter is not loaded")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        source = image.convert("RGB")
        grayscale = np.asarray(source.convert("L"), dtype=np.uint8)
        cutoff = round(threshold * 255)
        mask = np.where(grayscale >= cutoff, 255, 0).astype(np.uint8)
        return {
            "mask": mask,
            "overlay": make_overlay(source, mask),
            "metadata": {"width": source.width, "height": source.height},
        }

    def unload(self) -> None:
        self.loaded = False
