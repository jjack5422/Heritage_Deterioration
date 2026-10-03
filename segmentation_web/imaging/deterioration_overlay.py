"""Compose independently predicted deterioration masks without losing overlaps."""

from collections.abc import Mapping

import numpy as np
from PIL import Image


def compose_deterioration_masks(
    image: Image.Image,
    masks: Mapping[str, Image.Image],
    colors: Mapping[str, tuple[int, int, int]],
    *,
    alpha: float = 0.45,
) -> dict:
    """Render class colors, charcoal overlap regions, and opaque white hatching."""

    if not masks or set(masks) != set(colors):
        raise ValueError("Masks and colors must contain the same deterioration classes")
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay opacity must be between 0 and 1")
    source = image.convert("RGB")
    count = np.zeros((source.height, source.width), dtype=np.uint8)
    colored = np.zeros((source.height, source.width, 3), dtype=np.uint8)
    for class_id, mask in masks.items():
        if mask.size != source.size:
            raise ValueError("All deterioration masks must match the source image")
        foreground = np.asarray(mask.convert("L")) > 0
        count += foreground
        colored[foreground] = colors[class_id]
    overlap = count > 1
    colored[overlap] = (70, 70, 70)
    # Broadcast compact row/column vectors instead of allocating coordinate grids.
    rows = np.arange(source.height, dtype=np.uint32)[:, None]
    columns = np.arange(source.width, dtype=np.uint32)[None, :]
    hatch = overlap & ((rows + columns) % 12 < 3)
    colored[hatch] = (255, 255, 255)
    color_mask = Image.fromarray(colored)
    foreground_mask = Image.fromarray(np.where(count > 0, 255, 0).astype(np.uint8))
    overlay = Image.composite(Image.blend(source, color_mask, alpha), source, foreground_mask)
    hatch_mask = Image.fromarray(hatch.astype(np.uint8) * 255)
    overlay = Image.composite(Image.new("RGB", source.size, "white"), overlay, hatch_mask)
    return {
        "mask": color_mask,
        "overlay": overlay,
        "overlap_pixels": int(np.count_nonzero(overlap)),
    }
