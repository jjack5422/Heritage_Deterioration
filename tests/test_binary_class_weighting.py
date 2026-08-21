from __future__ import annotations

import numpy as np
import pytest
import torch

from crackseg_common.data_plan import (
    binary_positive_weight,
    binary_weighting_record,
    class_weights,
    joint_class_weights,
)


def test_binary_weights_use_a_fixed_background_one_to_foreground_two_ratio() -> None:
    counts = np.array([9_900, 100], dtype=np.int64)

    binary = class_weights(counts)

    assert binary.tolist() == pytest.approx([0.5, 1.0])


def test_binary_weights_do_not_change_with_fold_pixel_prevalence() -> None:
    imbalanced = class_weights(np.array([9_900, 100], dtype=np.int64))
    nearly_balanced = class_weights(np.array([600, 400], dtype=np.int64))

    assert torch.equal(imbalanced, nearly_balanced)


def test_binary_positive_weight_is_the_foreground_to_background_ratio() -> None:
    counts = np.array([9_900, 100], dtype=np.int64)

    assert binary_positive_weight(counts) == pytest.approx(2.0)


def test_binary_weighting_record_names_the_fixed_two_to_one_contract() -> None:
    record = binary_weighting_record(np.array([9_900, 100], dtype=np.int64))

    assert record["scheme"] == "fixed_binary_background_1_foreground_2"
    assert record["cross_entropy_class_weights"] == pytest.approx([0.5, 1.0])
    assert record["bce_positive_weight"] == pytest.approx(2.0)


def test_three_class_inverse_sqrt_contract_is_unchanged() -> None:
    counts = np.array([9_000, 700, 300], dtype=np.int64)

    assert joint_class_weights(counts).tolist() == pytest.approx(
        [0.25, 0.79128784, 1.20871215]
    )
