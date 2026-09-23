"""Protect the fixed transfer split and KYT thinning rules."""

from pathlib import Path

import numpy as np

from model_projects.sam3_adapter.scripts.data.prepare_transfer_expert_splits import (
    SPLITS,
    _default_output,
    thin_7px,
    zhang_suen_skeleton,
)


def test_transfer_keeps_benchmark_groups_without_fixed_training_count() -> None:
    assert set(SPLITS["shrinkage_craquelure"]["test"]) == {
        "KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1",
    }
    assert set(SPLITS["shrinkage_craquelure"]["unused_holdout"]) == {
        "KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A3E-3-2",
    }


def test_new_release_gets_release_specific_prepared_output() -> None:
    assert _default_output(Path("dataset20260101")).name == "dataset20260101"


def test_zhang_suen_and_radius_three_disk_on_straight_line() -> None:
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[250:263, 254:257] = 255
    skeleton = zhang_suen_skeleton(mask == 255)
    assert skeleton.sum() < (mask == 255).sum()
    result = thin_7px(mask)
    assert set(np.unique(result)).issubset({0, 255})
    assert result[256, 253] == 255
    assert result[256, 257] == 255
    assert result[256, 250] == 0


def test_unannotated_kyt_tile_stays_background() -> None:
    assert np.count_nonzero(thin_7px(np.zeros((512, 512), dtype=np.uint8))) == 0
