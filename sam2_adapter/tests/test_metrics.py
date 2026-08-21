"""Tests for binary merged-crack metric accounting."""

from __future__ import annotations

from sam2_adapter.metrics import binary_summary


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
