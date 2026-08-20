"""Tests for stage and final-H0 metric accounting."""

from __future__ import annotations

import numpy as np

from sam2_adapter.metrics import binary_summary, hierarchy_summary


def test_binary_summary_reports_pixel_micro_and_panel_macro() -> None:
    summary = binary_summary(
        {
            "panel_a": (2, 1, 1),
            "panel_b": (1, 0, 1),
        }
    )

    assert summary["tile_micro"]["tp"] == 3
    assert summary["tile_micro"]["fp"] == 1
    assert summary["tile_micro"]["fn"] == 2
    assert summary["tile_micro"]["mf1"] == 2 * 3 / (2 * 3 + 1 + 2)
    assert summary["expert_panel_macro"]["panel_count"] == 2
    assert summary["expert_panel_macro"]["f1"] == (2 * 2 / 6 + 2 * 1 / 3) / 2


def test_hierarchy_summary_scores_each_foreground_class_without_overlap() -> None:
    target = np.array([[0, 1, 2, 2]], dtype=np.uint8)
    prediction = np.array([[0, 1, 1, 2]], dtype=np.uint8)

    summary = hierarchy_summary(target, prediction, ignore_value=255)

    assert summary["mutually_exclusive"] is True
    assert summary["per_class"]["crack"]["tp"] == 1
    assert summary["per_class"]["crack"]["fp"] == 1
    assert summary["per_class"]["craquelure"]["fn"] == 1
    assert summary["foreground_macro"]["f1"] == (2 / 3 + 2 / 3) / 2
