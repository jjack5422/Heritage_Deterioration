"""Behavioral checks for deterioration statistics and validation selection."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.data.analyze_deterioration_dataset import (
    MERGED_TARGETS,
    Tile,
    _read_mask,
    cross_dataset_leakage_audit,
    dataset114_training_only,
    merged_statistics,
    raw_statistics,
    selected_validation,
    validation_candidates,
)


def tile(group: str, name: str, counts: dict[int, int], *, total: int = 100, dataset: str = "dataset", image: Path | None = None) -> Tile:
    return Tile(dataset, group, name, image or Path(name + ".jpg"), Path(name + ".png"), total, counts)


def classes() -> dict[int, tuple[str, str]]:
    return {raw_id: ("BG" if raw_id == 0 else f"D-{raw_id:02d}", f"class-{raw_id}") for raw_id in range(38)}


def test_pixel_count_and_tile_occurrence_are_distinct():
    rows = raw_statistics([
        tile("a", "a1", {0: 90, 1: 10}),
        tile("a", "a2", {0: 95, 1: 5}),
        tile("b", "b1", {0: 93, 1: 7}),
    ], classes())
    crack = rows[0]
    assert crack["pixel_count"] == 22
    assert crack["tile_count"] == 3
    assert crack["source_group_count"] == 2
    assert crack["pixel_ratio"] == pytest.approx(22 / 300)
    assert crack["tile_ratio"] == 1.0


def test_merged_tile_occurrence_is_not_double_counted():
    rows = {row["target_name"]: row for row in merged_statistics([
        tile("a", "both", {0: 80, 3: 8, 4: 12}),
        tile("b", "one", {0: 95, 4: 5}),
    ])}
    merged = rows["shrinkage_craquelure"]
    assert merged["pixel_count"] == 25
    assert merged["tile_count"] == 2
    assert merged["source_group_count"] == 2


def test_unknown_raw_id_fails_immediately(tmp_path):
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (2, 1)).save(image)
    Image.fromarray(np.array([[0, 99]], dtype=np.uint8)).save(mask)
    with pytest.raises(ValueError, match=r"Unknown raw ID.*99"):
        _read_mask(mask, image, set(range(38)))


def test_selection_covers_the_most_merged_targets():
    tiles = [
        tile("a", "a", {3: 10}),
        tile("b", "b", {11: 10}),
        tile("c", "c", {16: 10}),
        tile("d", "d", {3: 1}),
    ]
    candidates = validation_candidates(tiles, 2)
    assert candidates[0]["missing_target_count"] == 1
    assert set(str(candidates[0]["source_groups"]).split(";")) != {"a", "d"}


def test_coverage_balance_precedes_tile_ratio_deviation():
    tiles = [
        tile("a", "a1", {3: 25, 11: 25, 16: 25}),
        tile("a", "a2", {3: 25, 11: 25, 16: 25}),
        tile("b", "b1", {3: 10, 11: 20, 16: 20}),
        tile("d", "d1", {3: 20, 11: 10, 16: 10}),
    ]
    best = validation_candidates(tiles, 1)[0]
    assert best["source_groups"] == "a"
    assert best["pixel_coverage_range"] == pytest.approx(0.0)
    assert best["validation_tile_ratio_deviation"] > 0


def test_source_group_never_crosses_training_and_validation():
    tiles = [
        tile("a", "a1", {3: 10, 11: 10, 16: 10}),
        tile("a", "a2", {3: 5}),
        tile("b", "b1", {4: 5, 1: 5, 2: 5}),
        tile("c", "c1", {3: 5, 11: 5, 36: 5}),
    ]
    candidates = validation_candidates(tiles, 1)
    selected = selected_validation(tiles, candidates, "class-hash", "dataset-hash")
    assert selected["leakage_audit"] == {"source_group_overlap": [], "tile_overlap": [], "passed": True}
    a_validation_tiles = {"a1", "a2"} & set(selected["validation_tiles"])
    assert len(a_validation_tiles) in {0, 2}


def test_identical_inputs_produce_identical_selection():
    tiles = [
        tile("a", "a", {3: 10, 11: 10, 16: 10}),
        tile("b", "b", {4: 10, 1: 10, 2: 10}),
        tile("c", "c", {3: 10, 11: 10, 36: 10}),
    ]
    first = validation_candidates(tiles, 1)
    second = validation_candidates(list(reversed(tiles)), 1)
    assert first == second
    assert selected_validation(tiles, first, "c", "d")["split_sha256"] == selected_validation(list(reversed(tiles)), second, "c", "d")["split_sha256"]
    assert tuple(MERGED_TARGETS) == ("shrinkage_craquelure", "scratch_crack", "abrasion_loss_rodent_bite")


def test_dataset114_exact_overlap_is_quarantined_and_never_validated(tmp_path):
    primary_image = tmp_path / "primary.png"
    secondary_image = tmp_path / "secondary.png"
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    Image.fromarray(pixels).save(primary_image)
    Image.fromarray(pixels).save(secondary_image)
    primary = [tile("current-group", "current", {3: 10}, image=primary_image)]
    secondary = [
        tile("legacy-group", "legacy-match", {3: 10}, dataset="dataset114", image=secondary_image),
        tile("safe-group", "safe", {3: 10}, dataset="dataset114", image=tmp_path / "safe.png"),
    ]
    Image.new("RGB", (8, 8), color=(12, 80, 160)).save(secondary[1].image)

    audit = cross_dataset_leakage_audit(primary, secondary)
    split = dataset114_training_only(secondary, audit)

    assert audit["quarantined_dataset114_source_groups"] == ["legacy-group"]
    assert split["training_source_groups"] == ["safe-group"]
    assert split["validation_tiles"] == []
    assert split["leakage_audit"]["passed"] is True
