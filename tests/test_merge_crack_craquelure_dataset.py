from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.merge_crack_craquelure_dataset import merge_dataset, remap_mask


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_remap_mask_merges_crack_and_craquelure_into_contiguous_class_ids() -> None:
    source = np.array([[0, 1, 2, 3, 4, 5, 6, 255]], dtype=np.uint8)

    actual = remap_mask(source)

    assert actual.tolist() == [[0, 1, 2, 3, 1, 4, 5, 255]]


def test_merge_dataset_preserves_source_and_split_membership(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "merged"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir()
    (source / "splits").mkdir()
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    mask = np.array([[0, 1, 4, 2], [3, 5, 6, 255]], dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(source / "images" / "panel_R1_C01.png")
    Image.fromarray(mask, mode="L").save(source / "masks" / "panel_R1_C01.png")
    (source / "tile_index.json").write_text(
        json.dumps({"summary": {"task": "multiclass_deterioration"}, "items": [{"tile": "panel_R1_C01.png"}]}),
        encoding="utf-8",
    )
    split = {
        "outer_fold": 0,
        "task": "multiclass_deterioration",
        "holdout_tiles": ["panel_R1_C01.png"],
        "folds": [{"train": []}],
        "data_contract": {},
        "evaluation_contract": {"ground_truth_rule": "raw_mask == 1"},
    }
    (source / "splits" / "fold0.json").write_text(json.dumps(split), encoding="utf-8")
    manifest = {
        "schema_version": "multiclass-pair-authority-v1",
        "pair_count": 1,
        "training_eligible": True,
        "label_contract": {
            "class_ids": {
                "background": 0,
                "crack": 1,
                "loss": 2,
                "shrinkage": 3,
                "craquelure": 4,
                "flaking": 5,
                "stain": 6,
            },
            "ignore_value": 255,
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source_mask_hash = _sha256(source / "masks" / "panel_R1_C01.png")

    summary = merge_dataset(source, destination)

    assert _sha256(source / "masks" / "panel_R1_C01.png") == source_mask_hash
    merged = np.asarray(Image.open(destination / "masks" / "panel_R1_C01.png"))
    assert merged.tolist() == [[0, 1, 1, 2], [3, 4, 5, 255]]
    assert summary["pair_count"] == 1
    assert summary["source_pixel_counts"]["crack"] == 1
    assert summary["source_pixel_counts"]["craquelure"] == 1
    assert summary["merged_pixel_counts"]["craquelure"] == 2
    merged_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert merged_manifest["label_contract"]["class_ids"] == {
        "background": 0,
        "craquelure": 1,
        "loss": 2,
        "shrinkage": 3,
        "flaking": 4,
        "stain": 5,
    }
    merged_split = json.loads((destination / "splits" / "fold0.json").read_text(encoding="utf-8"))
    assert merged_split["holdout_tiles"] == split["holdout_tiles"]
    assert merged_split["folds"] == split["folds"]
    manifest_class_names = [
        name
        for name, _ in sorted(
            merged_manifest["label_contract"]["class_ids"].items(),
            key=lambda item: item[1],
        )
    ]
    assert merged_split["data_contract"]["class_names"] == manifest_class_names
    assert merged_split["evaluation_contract"]["ground_truth_rule"] == "raw_mask == 1 (merged craquelure)"


def test_merge_dataset_refuses_to_overwrite_an_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "merged"
    source.mkdir()
    destination.mkdir()

    try:
        merge_dataset(source, destination)
    except FileExistsError as error:
        assert str(destination) in str(error)
    else:
        raise AssertionError("merge_dataset must not overwrite an existing destination")
