"""Protect the fixed transfer split and KYT thinning rules."""

import numpy as np

from scripts.data.prepare_transfer_expert_splits import SPLITS, thin_7px, zhang_suen_skeleton


def test_transfer_partition_counts_and_kyt_only_test() -> None:
    assert SPLITS["scratch_crack"]["counts"] == (1241, 105, 112)
    assert SPLITS["loss"]["counts"] == (1243, 108, 107)
    assert SPLITS["shrinkage_craquelure"]["counts"] == (1081, 123, 142)
    assert set(SPLITS["shrinkage_craquelure"]["test"]) == {
        "KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1",
    }
    assert set(SPLITS["shrinkage_craquelure"]["unused_holdout"]) == {
        "KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A3E-3-2",
    }


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
