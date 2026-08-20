from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import threshold_preview  # noqa: E402
from threshold_preview import (  # noqa: E402
    GalleryState,
    build_preview_panels,
    discover_gallery_entries,
    export_threshold_outputs,
    navigation_delta,
)
from crackseg_common.thresholding import ThresholdPolicy  # noqa: E402


def test_preview_has_original_probability_mask_and_overlay_panels() -> None:
    image = np.full((2, 3, 3), 100, dtype=np.uint8)
    probability = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]], dtype=np.float32)

    panels = build_preview_panels(image, probability, threshold=0.6)

    assert tuple(panels) == ("original", "probability", "mask", "overlay")
    assert all(panel.shape == image.shape for panel in panels.values())
    assert np.count_nonzero(panels["mask"][..., 0]) == 2


def test_export_uses_exact_manual_threshold_and_writes_metadata(tmp_path: Path) -> None:
    image = np.full((2, 2, 3), 120, dtype=np.uint8)
    probability = np.array([[0.62, 0.63], [0.9, 0.1]], dtype=np.float32)
    policy = ThresholdPolicy.default("crack", foreground_id=1)

    paths = export_threshold_outputs(
        output_dir=tmp_path,
        stem="sample",
        image=image,
        probability=probability,
        threshold=0.63,
        policy=policy,
    )

    mask = np.asarray(Image.open(paths["mask"]))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    saved_policy = ThresholdPolicy.load(paths["policy"])
    np.testing.assert_array_equal(mask, [[0, 0], [1, 0]])
    assert metadata["threshold"] == 0.63
    assert metadata["inference_rule"] == "foreground_probability > threshold"
    assert saved_policy.threshold == 0.63
    assert saved_policy.source == "manual"


def _make_gallery_fold(
    root: Path,
    *,
    experiment: str,
    expert: str,
    fold: int,
    image_names: tuple[str, ...],
) -> Path:
    fold_dir = root / "runs" / experiment / "5fold" / expert / f"fold{fold}"
    checkpoint = fold_dir / "artifacts" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    for name in image_names:
        image = fold_dir / "artifacts" / "qualitative" / name / "input.png"
        image.parent.mkdir(parents=True)
        image.touch()
    return checkpoint


def test_discover_gallery_entries_finds_experts_folds_and_validation_images(
    tmp_path: Path,
) -> None:
    crack_checkpoint = _make_gallery_fold(
        tmp_path,
        experiment="experiment_b",
        expert="crack",
        fold=1,
        image_names=("tile_b", "tile_a"),
    )
    _make_gallery_fold(
        tmp_path,
        experiment="experiment_a",
        expert="craquelure",
        fold=0,
        image_names=("tile_c",),
    )
    _make_gallery_fold(
        tmp_path,
        experiment="ignored",
        expert="loss",
        fold=0,
        image_names=("tile_x",),
    )

    entries = discover_gallery_entries(tmp_path)

    assert [(entry.experiment, entry.expert, entry.fold) for entry in entries] == [
        ("experiment_b", "crack", 1),
        ("experiment_a", "craquelure", 0),
    ]
    assert entries[0].checkpoint == crack_checkpoint
    assert [path.parent.name for path in entries[0].images] == ["tile_a", "tile_b"]
    assert entries[0].label == "experiment_b / crack / fold1"


def test_gallery_state_wraps_and_caches_each_image_once(tmp_path: Path) -> None:
    images = tuple(tmp_path / name for name in ("a.png", "b.png", "c.png"))
    state = GalleryState(images)
    loaded: list[Path] = []

    def load(path: Path) -> np.ndarray:
        loaded.append(path)
        return np.full((2, 2), len(loaded) / 10, dtype=np.float32)

    first = state.probability(load)
    assert state.move(navigation_delta("D")) == images[-1]
    state.probability(load)
    assert state.move(navigation_delta("f")) == images[0]
    assert state.probability(load) is first
    assert loaded == [images[0], images[-1]]


def test_navigation_only_maps_d_and_f() -> None:
    assert navigation_delta("D") == -1
    assert navigation_delta("f") == 1
    assert navigation_delta("x") is None


def test_empty_gallery_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少需要一張"):
        GalleryState(())


def test_main_without_arguments_launches_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        threshold_preview,
        "launch_gallery",
        lambda project_root: calls.append(project_root) or 17,
    )

    assert threshold_preview.main([]) == 17
    assert calls == [Path(threshold_preview.__file__).resolve().parents[1]]
