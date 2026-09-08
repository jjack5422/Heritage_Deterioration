from __future__ import annotations

import json
from pathlib import Path

import pytest

from dual_adapter_sam3.evaluate_decoder_calibration import (
    build_cross_validation_summary,
    build_parser,
)
from dual_adapter_sam3.train_decoder_calibration import (
    DEFAULT_EXPERIMENT,
    DEFAULT_SOURCE_EXPERIMENT,
)


def _metrics(value: float) -> dict[str, object]:
    return {
        "macro": {metric: value for metric in ("precision", "recall", "f1", "iou", "accuracy")},
        "per_class": {
            concept: {
                metric: value
                for metric in ("precision", "recall", "f1", "iou", "accuracy")
            }
            for concept in ("crack_craquelure", "loss")
        },
        "predicted_to_gt_pixel_ratio": {
            "crack_craquelure": value,
            "loss": value,
            "macro": value,
        },
    }


def test_evaluation_cli_uses_calibration_experiment_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.experiment_id == DEFAULT_EXPERIMENT
    assert args.source_experiment_id == DEFAULT_SOURCE_EXPERIMENT


def test_summary_reports_paired_source_delta_and_expert_links(tmp_path: Path) -> None:
    results = [
        {
            "fold": fold,
            "source_stage2": _metrics(0.4),
            "decoder_calibration": _metrics(0.5),
        }
        for fold in range(5)
    ]

    summary = build_cross_validation_summary(tmp_path, results)

    assert summary["decoder_calibration_minus_source_stage2"]["macro"]["f1"]["mean"] == pytest.approx(0.1)
    payload = json.loads(
        (tmp_path / "metrics" / "cross_validation_summary.json").read_text()
    )
    assert payload["expert"] == "da_sam3_decoder_calibration"
    html = (tmp_path / "reports" / "cross_validation.html").read_text()
    assert "../5fold/da_sam3_decoder_calibration/fold0/reports/index.html" in html
