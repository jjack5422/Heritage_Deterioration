"""Shared arbitrary-size tiled inference for real segmentation adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor


PredictBatch = Callable[[Tensor], Tensor]
TileNormalization = Literal["imagenet", "zero_one"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sliding_positions(length: int, *, tile_size: int, stride: int) -> tuple[int, ...]:
    """Return window origins that cover a dimension and align its far edge."""

    if length <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("tile_size and stride must satisfy 0 < stride <= tile_size")
    if length <= tile_size:
        return (0,)
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return tuple(positions)


def _gaussian_window(tile_size: int, sigma_ratio: float = 0.125) -> np.ndarray:
    sigma = max(1.0, tile_size * sigma_ratio)
    coordinates = np.arange(tile_size, dtype=np.float32) - (tile_size - 1) / 2.0
    axis = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    window = np.outer(axis, axis).astype(np.float32)
    window /= float(window.max())
    return window


def _normalized_tile(
    tile: np.ndarray,
    normalization: TileNormalization = "imagenet",
) -> Tensor:
    tensor = torch.from_numpy(tile.copy()).permute(2, 0, 1).float().div_(255.0)
    if normalization == "zero_one":
        return tensor
    if normalization != "imagenet":
        raise ValueError(f"unsupported tile normalization: {normalization}")
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def tiled_foreground_probability(
    image: Image.Image,
    predict_batch: PredictBatch,
    *,
    device: torch.device,
    tile_size: int = 512,
    stride: int = 384,
    batch_size: int = 1,
    normalization: TileNormalization = "imagenet",
) -> tuple[np.ndarray, int]:
    """Infer and blend a foreground probability map at the original size."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = source.shape[:2]
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded[:height, :width] = source

    y_positions = sliding_positions(
        padded_height,
        tile_size=tile_size,
        stride=stride,
    )
    x_positions = sliding_positions(
        padded_width,
        tile_size=tile_size,
        stride=stride,
    )
    positions = tuple((y, x) for y in y_positions for x in x_positions)
    window = _gaussian_window(tile_size)
    probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)

    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        batch = torch.stack(
            [
                _normalized_tile(
                    padded[y : y + tile_size, x : x + tile_size],
                    normalization,
                )
                for y, x in batch_positions
            ]
        ).to(device, non_blocking=device.type == "cuda")
        probability = predict_batch(batch)
        if probability.ndim == 4 and probability.shape[1] == 1:
            probability = probability[:, 0]
        expected_shape = (len(batch_positions), tile_size, tile_size)
        if tuple(probability.shape) != expected_shape:
            raise RuntimeError(
                "model probability shape mismatch: "
                f"expected {expected_shape}, got {tuple(probability.shape)}"
            )
        probability_array = probability.detach().float().cpu().numpy()
        if not np.isfinite(probability_array).all():
            raise RuntimeError("model produced non-finite foreground probabilities")
        if probability_array.min() < 0.0 or probability_array.max() > 1.0:
            raise RuntimeError("model foreground probabilities must be between 0 and 1")
        for tile_probability, (y, x) in zip(
            probability_array,
            batch_positions,
            strict=True,
        ):
            probability_sum[y : y + tile_size, x : x + tile_size] += (
                tile_probability * window
            )
            weight_sum[y : y + tile_size, x : x + tile_size] += window

    blended = probability_sum / np.maximum(weight_sum, np.finfo(np.float32).tiny)
    return blended[:height, :width].astype(np.float32, copy=False), len(positions)
