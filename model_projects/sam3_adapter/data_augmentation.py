"""Training-only image and mask augmentation for SAM3-Adapter experts."""

from __future__ import annotations

import random

import numpy as np

ROTATION_QUARTER_TURNS = 4
HORIZONTAL_FLIP_PROBABILITY = 0.5
VERTICAL_FLIP_PROBABILITY = 0.5
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.85, 1.15)
GAMMA_RANGE = (0.85, 1.15)
CHANNEL_GAIN_RANGE = (0.95, 1.05)


def photometric_augment(image: np.ndarray) -> np.ndarray:
    """Apply conservative RGB-only intensity and color variation."""

    values = image.astype(np.float32) / 255.0
    mean = values.mean(axis=(0, 1), keepdims=True)
    values = (values - mean) * random.uniform(*CONTRAST_RANGE) + mean
    values *= random.uniform(*BRIGHTNESS_RANGE)
    values = np.clip(values, 0.0, 1.0) ** random.uniform(*GAMMA_RANGE)
    channel_gains = np.asarray(
        [random.uniform(*CHANNEL_GAIN_RANGE) for _ in range(3)],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    return np.rint(np.clip(values * channel_gains, 0.0, 1.0) * 255.0).astype(np.uint8)


def augment_training_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact shared geometry and RGB-only photometric augmentation."""

    quarter_turns = random.randrange(ROTATION_QUARTER_TURNS)
    if quarter_turns:
        image = np.rot90(image, quarter_turns, axes=(0, 1)).copy()
        mask = np.rot90(mask, quarter_turns, axes=(0, 1)).copy()
    if random.random() < HORIZONTAL_FLIP_PROBABILITY:
        image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
    if random.random() < VERTICAL_FLIP_PROBABILITY:
        image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
    return photometric_augment(image), mask
