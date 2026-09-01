from monument_da_sam3.evaluate_cross_validation import _finite_binary_row, _mean_std, _metrics_from_rows

import numpy as np


def test_outer_empty_target_metrics_are_finite() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    row = _finite_binary_row(empty, empty)
    assert row["f1"] == 1.0
    assert row["iou"] == 1.0
    assert row["accuracy"] == 1.0


def test_cross_fold_summary_uses_sample_standard_deviation() -> None:
    summary = _mean_std([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["std_sample"] == 1.0


def test_metrics_are_pooled_from_integer_confusion_counts() -> None:
    rows = [
        {"target_class": "crack_craquelure", "tp": 2, "fp": 1, "fn": 1, "tn": 6},
        {"target_class": "loss", "tp": 1, "fp": 0, "fn": 1, "tn": 8},
    ]
    result = _metrics_from_rows(rows)
    assert result["per_class"]["crack_craquelure"]["f1"] == 4 / 6
    assert result["per_class"]["loss"]["f1"] == 2 / 3
    assert result["macro"]["f1"] == 2 / 3
