from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC))

from crackseg_common.reporting.threshold_curves import (  # noqa: E402
    curve_series,
    render_pr_roc_curve,
)
from crackseg_common.thresholding import ThresholdSweep  # noqa: E402


def calibrated_policy():
    probability = np.array([[0.9, 0.8, 0.6, 0.4, 0.2, 0.1]], dtype=np.float32)
    target = np.array([[1, 0, 1, 0, 1, 0]], dtype=np.uint8)
    sweep = ThresholdSweep(np.array([0.3, 0.5, 0.7]))
    sweep.update(probability, target)
    return sweep.build_policy("crack", foreground_id=1)


def test_curve_series_contains_pr_and_roc_coordinates_for_every_threshold() -> None:
    series = curve_series(calibrated_policy())

    assert [point[2] for point in series["pr"]] == [0.3, 0.5, 0.7]
    assert series["pr"][1][:2] == (2 / 3, 2 / 3)
    assert series["roc"][1][:2] == (1 / 3, 2 / 3)


def test_rendered_pr_roc_curve_is_a_directly_viewable_local_png(tmp_path: Path) -> None:
    destination = render_pr_roc_curve(calibrated_policy(), tmp_path / "pr_roc_curve.png")

    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(destination) as image:
        assert image.size == (1600, 900)
        assert image.mode == "RGB"
        assert image.getbbox() == (0, 0, 1600, 900)
