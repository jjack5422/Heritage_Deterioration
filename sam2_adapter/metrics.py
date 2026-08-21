"""Metric accounting for binary merged-crack segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

def counts_summary(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    """Return standard foreground pixel metrics, retaining raw sufficient counts."""

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mprecision": tp / precision_denominator if precision_denominator else None,
        "mrecall": tp / recall_denominator if recall_denominator else None,
        "mf1": 2 * tp / f1_denominator if f1_denominator else None,
        "miou": tp / iou_denominator if iou_denominator else None,
    }


def binary_summary(panel_counts: Mapping[str, tuple[int, int, int]]) -> dict[str, Any]:
    """Combine group-level TP/FP/FN into tile-micro and panel-macro summaries."""

    total_tp = sum(count[0] for count in panel_counts.values())
    total_fp = sum(count[1] for count in panel_counts.values())
    total_fn = sum(count[2] for count in panel_counts.values())
    rows = [counts_summary(*count) for _, count in sorted(panel_counts.items())]
    scored = [row for row in rows if row["mf1"] is not None]

    def macro(column: str) -> float | None:
        values = [float(row[column]) for row in scored if row[column] is not None]
        return sum(values) / len(values) if values else None

    positive_panels = sum(1 for _, _, fn in panel_counts.values() if fn > 0)
    # A panel with TP but no FN is also a positive reference panel.
    positive_panels = sum(1 for tp, _, fn in panel_counts.values() if tp + fn > 0)
    return {
        "tile_micro": counts_summary(total_tp, total_fp, total_fn),
        "expert_panel_macro": {
            "f1": macro("mf1"),
            "precision": macro("mprecision"),
            "recall": macro("mrecall"),
            "iou": macro("miou"),
            "dice": macro("mf1"),
            "panel_count": len(rows),
            "positive_panel_count": positive_panels,
            "scored_panel_count": len(scored),
        },
    }
