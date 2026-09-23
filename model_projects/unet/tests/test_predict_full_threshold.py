from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from predict_full import (  # noqa: E402
    discover_threshold_policy,
    files_from_split_plan,
    resolve_inference_sources,
    resolve_threshold_policy,
)
from crackseg_common.thresholding import ThresholdPolicy  # noqa: E402


def test_policy_threshold_is_used_and_expert_mismatch_is_rejected(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    ThresholdPolicy.default("crack", 1, threshold=0.67).save(policy_path)

    policy = resolve_threshold_policy("crack", policy_path=policy_path)

    assert policy.threshold == pytest.approx(0.67)
    with pytest.raises(ValueError, match="expert"):
        resolve_threshold_policy("craquelure", policy_path=policy_path)


def test_manual_cli_threshold_overrides_policy_but_is_recorded_as_manual(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    ThresholdPolicy.default("crack", 1, threshold=0.67).save(policy_path)

    policy = resolve_threshold_policy("crack", threshold=0.72, policy_path=policy_path)

    assert policy.threshold == pytest.approx(0.72)
    assert policy.source == "manual"


def test_split_plan_selects_only_requested_validation_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("train.png", "val.png", "test.png"):
        (image_dir / name).touch()
    plan = tmp_path / "split_plan.json"
    plan.write_text(
        json.dumps({"train": ["train.png"], "val": ["val.png"], "test": ["test.png"]}),
        encoding="utf-8",
    )

    selected = files_from_split_plan(image_dir, plan, "val")

    assert selected == [str(image_dir / "val.png")]


def test_preview_mode_opens_desktop_pickers_when_paths_are_omitted() -> None:
    calls = []

    def choose(checkpoint, image):
        calls.append((checkpoint, image))
        return "/runs/crack/fold0/best.pt", "/images/panel.png"

    sources = resolve_inference_sources(
        checkpoint=None,
        image=None,
        image_dir=None,
        preview_threshold=True,
        chooser=choose,
    )

    assert sources == ("/runs/crack/fold0/best.pt", "/images/panel.png", None)
    assert calls == [(None, None)]


def test_batch_mode_still_requires_explicit_checkpoint_and_image() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_inference_sources(None, None, None, preview_threshold=False)


def test_checkpoint_automatically_discovers_its_fold_threshold_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fold0" / "artifacts" / "checkpoints" / "best.pt"
    policy = tmp_path / "fold0" / "config" / "threshold_policy.json"
    checkpoint.parent.mkdir(parents=True)
    policy.parent.mkdir(parents=True)
    checkpoint.touch()
    policy.write_text("{}", encoding="utf-8")

    assert discover_threshold_policy(checkpoint) == policy
