from __future__ import annotations

import pytest

from scripts.reporting.backfill_pixel_accuracy import epoch_accuracies, reconstruct_counts


def test_reconstruct_counts_from_fixed_positive_total() -> None:
    assert reconstruct_counts(precision=0.7, recall=0.875, positive_pixels=8) == (7, 3, 1)


def test_epoch_accuracy_includes_true_negatives() -> None:
    rows = [{"epoch": "1", "precision": "0.7", "recall": "0.875"}]
    assert epoch_accuracies(rows, background_pixels=12, positive_pixels=8) == [(1, 0.8)]


def test_zero_precision_is_not_silently_invented() -> None:
    with pytest.raises(ValueError, match="zero precision"):
        reconstruct_counts(precision=0.0, recall=0.0, positive_pixels=8)
