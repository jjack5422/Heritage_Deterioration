from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crackseg_common.thresholding import (  # noqa: E402
    ThresholdPolicy,
    ThresholdSweep,
    apply_threshold,
    calibrate_validation_directory,
    load_foreground_probability,
)


def test_foreground_probability_and_threshold_boundary_are_explicit() -> None:
    two_channel = np.array(
        [[[0.9, 0.5], [0.2, 0.0]], [[0.1, 0.5], [0.8, 1.0]]],
        dtype=np.float32,
    )

    probability = load_foreground_probability(two_channel)
    prediction = apply_threshold(probability, 0.5)

    np.testing.assert_array_equal(probability, two_channel[1])
    np.testing.assert_array_equal(prediction, [[0, 0], [1, 1]])
    assert prediction.dtype == np.uint8


def test_threshold_sweep_builds_balanced_clean_and_sensitive_presets() -> None:
    probability = np.array([[0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]])
    target = np.array([[1, 1, 0, 1, 0, 0, 0, 255]], dtype=np.uint8)
    sweep = ThresholdSweep(np.array([0.2, 0.5, 0.7, 0.85]))

    sweep.update(probability, target, foreground_id=1, ignore_value=255)
    policy = sweep.build_policy(
        expert="crack",
        foreground_id=1,
        precision_floor=0.9,
        recall_floor=0.95,
        selected_preset="balanced",
        validation={"split": "validation", "image_count": 1},
    )

    assert policy.threshold == pytest.approx(0.5)
    assert policy.presets["balanced"].threshold == pytest.approx(0.5)
    assert policy.presets["clean"].threshold == pytest.approx(0.7)
    assert policy.presets["clean"].precision == pytest.approx(1.0)
    assert policy.presets["clean"].constraint_met is True
    assert policy.presets["sensitive"].threshold == pytest.approx(0.5)
    assert policy.presets["sensitive"].recall == pytest.approx(1.0)
    assert policy.presets["sensitive"].constraint_met is True
    assert policy.validation["negative_pixels"] == 4
    assert policy.presets["balanced"].tn == 3
    assert policy.presets["balanced"].fpr == pytest.approx(0.25)


def test_policy_json_round_trip_and_manual_threshold(tmp_path: Path) -> None:
    sweep = ThresholdSweep(np.array([0.25, 0.5, 0.75]))
    sweep.update(np.array([[0.9, 0.6, 0.4, 0.1]]), np.array([[1, 1, 0, 0]]))
    policy = sweep.build_policy(expert="craquelure", foreground_id=4)
    path = tmp_path / "threshold_policy.json"

    policy.save(path)
    restored = ThresholdPolicy.load(path)
    manual = restored.with_manual_threshold(0.63)

    assert restored == policy
    assert manual.threshold == pytest.approx(0.63)
    assert manual.source == "manual"
    assert manual.selected_preset == "manual"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_directory_calibration_reads_only_validation_tiles(tmp_path: Path) -> None:
    probability_dir = tmp_path / "prob"
    mask_dir = tmp_path / "masks"
    probability_dir.mkdir()
    mask_dir.mkdir()
    validation_name = "panelA_R1_C01__y00000_x00000.png"
    test_name = "panelB_R1_C01__y00000_x00000.png"
    np.save(probability_dir / f"{Path(validation_name).stem}.npy", np.array([[0.8, 0.2]]))
    np.save(probability_dir / f"{Path(test_name).stem}.npy", np.array([[0.1, 0.9]]))
    Image.fromarray(np.array([[1, 0]], dtype=np.uint8)).save(mask_dir / validation_name)
    Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(mask_dir / test_name)
    split_plan = tmp_path / "split_plan.json"
    split_plan.write_text(
        json.dumps(
            {
                "expert": {"name": "crack", "source_class_id": 1},
                "ignore_value": 255,
                "val": [validation_name],
                "test": [test_name],
            }
        ),
        encoding="utf-8",
    )

    policy = calibrate_validation_directory(
        probability_dir=probability_dir,
        mask_dir=mask_dir,
        split_plan_path=split_plan,
        thresholds=np.array([0.3, 0.5, 0.9]),
    )

    assert policy.expert == "crack"
    assert policy.foreground_id == 1
    assert policy.validation["image_count"] == 1
    assert len(policy.validation["threshold_metrics"]) == 3
    assert policy.presets["balanced"].threshold == pytest.approx(0.5)
