"""Training augmentation invariants for thin segmentation masks."""

import random

import numpy as np

from model_projects.sam3_adapter.data_augmentation import augment_training_pair


def test_quarter_turn_rotates_image_and_mask_without_interpolation(monkeypatch) -> None:
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    mask = np.array([[0, 3, 255], [4, 0, 3], [255, 4, 0]], dtype=np.uint8)
    monkeypatch.setattr(random, "randrange", lambda stop: 1)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    monkeypatch.setattr(random, "uniform", lambda low, high: 1.0)

    augmented_image, augmented_mask = augment_training_pair(image, mask)

    np.testing.assert_array_equal(augmented_image, np.rot90(image, 1))
    np.testing.assert_array_equal(augmented_mask, np.rot90(mask, 1))
    assert set(np.unique(augmented_mask)) == {0, 3, 4, 255}


def test_photometric_augmentation_never_changes_mask(monkeypatch) -> None:
    image = np.full((3, 3, 3), 200, dtype=np.uint8)
    mask = np.array([[0, 3, 0], [4, 255, 3], [0, 4, 0]], dtype=np.uint8)
    factors = iter((1.0, 0.85, 1.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(random, "randrange", lambda stop: 0)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    monkeypatch.setattr(random, "uniform", lambda low, high: next(factors))

    augmented_image, augmented_mask = augment_training_pair(image, mask)

    np.testing.assert_array_equal(augmented_image, np.full_like(image, 170))
    np.testing.assert_array_equal(augmented_mask, mask)
