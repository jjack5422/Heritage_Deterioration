"""Shared segmentation adapter contract."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adapters.tiled_inference import tiled_foreground_probability
from imaging.image_processing import make_overlay


class SegmentationAdapter:
    """Interface implemented by every supported segmentation architecture."""

    def load(self, weight_path: Path | None) -> None:
        raise NotImplementedError

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        raise NotImplementedError

    def unload(self) -> None:
        """Release model resources when switching adapters."""


class TiledTorchAdapter(SegmentationAdapter):
    """Base lifecycle for real binary models operating on normalized tiles."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        tile_size: int = 512,
        stride: int = 384,
        batch_size: int = 1,
    ) -> None:
        self.device = torch.device(device or "cuda")
        self.tile_size = tile_size
        self.stride = stride
        self.batch_size = batch_size
        self.model: torch.nn.Module | None = None
        self.load_metadata: dict[str, object] = {}

    def _require_runtime_device(self) -> None:
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real-model inference")

    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        if self.model is None:
            raise RuntimeError("Model adapter is not loaded")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        source = image.convert("RGB")
        probability, tile_count = tiled_foreground_probability(
            source,
            self._predict_batch,
            device=self.device,
            tile_size=self.tile_size,
            stride=self.stride,
            batch_size=self.batch_size,
        )
        mask = np.where(probability >= threshold, 255, 0).astype(np.uint8)
        return {
            "mask": mask,
            "overlay": make_overlay(source, mask),
            "metadata": {
                **self.load_metadata,
                "width": source.width,
                "height": source.height,
                "tile_count": tile_count,
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
            },
        }

    def unload(self) -> None:
        self.model = None
        self.load_metadata = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
