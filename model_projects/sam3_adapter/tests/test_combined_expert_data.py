"""Offline target materialization and prepared Dataset loading contracts."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from model_projects.sam3_adapter.training_data import (
    ExpertTileDataset,
    _canonical_hash,
    _content_identity,
    prepare_expert_data_plan,
)
from model_projects.sam3_adapter.scripts.data.prepare_combined_expert_splits import (
    Sample,
    _actual_path,
    _dataset_release_groups,
    expert_raw_ids,
)
from model_projects.sam3_adapter.scripts.data.prepare_transfer_expert_splits import _binary_target


def _save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path)


def test_materializer_maps_class_index_to_binary_target(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    _save(image, np.zeros((512, 512, 3), dtype=np.uint8))
    values = np.zeros((512, 512), dtype=np.uint8)
    values[0, :4] = [1, 3, 4, 255]
    _save(mask, values)
    sample = Sample(
        dataset="dataset_jacky",
        source_group="source",
        source_image_key="source",
        tile=image.name,
        image=image,
        mask=mask,
        masks={},
        pixels={},
    )

    target = _binary_target(sample, "shrinkage_craquelure")

    assert target[0, :4].tolist() == [0, 1, 1, 255]


def test_materializer_unions_binary_source_masks(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    first, second = tmp_path / "D-01.png", tmp_path / "D-11.png"
    _save(image, np.zeros((512, 512, 3), dtype=np.uint8))
    first_values = np.zeros((512, 512), dtype=np.uint8)
    second_values = np.zeros((512, 512), dtype=np.uint8)
    first_values[0, 0] = 255
    second_values[0, 1] = 255
    _save(first, first_values)
    _save(second, second_values)
    sample = Sample(
        dataset="dataset20260101",
        source_group="source",
        source_image_key="source",
        tile=image.name,
        image=image,
        mask=None,
        masks={1: first, 11: second},
        pixels={},
    )

    target = _binary_target(sample, "scratch_crack")

    assert target[0, :3].tolist() == [1, 1, 0]


def test_loader_reads_prepared_target_without_id_conversion(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "binary_target.png"
    _save(image, np.zeros((512, 512, 3), dtype=np.uint8))
    values = np.zeros((512, 512), dtype=np.uint8)
    values[0, :3] = [0, 1, 255]
    _save(mask, values)
    dataset = ExpertTileDataset(
        [{
            "dataset": "dataset_jacky",
            "source_group": "source",
            "tile": image.name,
            "image": str(image),
            "mask": str(mask),
            "mask_format": "binary_target",
        }],
        train_augmentation=False,
    )

    assert dataset[0]["target"][0, :3].tolist() == [0, 1, 255]


def test_metadata_apostrophe_path_resolves_exported_underscore(tmp_path: Path) -> None:
    actual = tmp_path / "KJWTomh-SC-M-A7_-1" / "tile.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"image")
    assert _actual_path(tmp_path, "KJWTomh-SC-M-A7'-1/tile.jpg") == actual.resolve()


def test_dynamic_release_uses_directory_name_and_standard_expert_ids(tmp_path: Path) -> None:
    root = tmp_path / "dataset20260101"
    image = root / "TEMPLE" / "image" / "source" / "tile.png"
    mask = root / "TEMPLE" / "mask" / "source" / "tile" / "D-01.png"
    image_values = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_values = np.zeros((512, 512), dtype=np.uint8)
    mask_values[0, 0] = 255
    _save(image, image_values)
    _save(mask, mask_values)
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "release.json").write_text(json.dumps({
        "release_id": root.name,
        "source_image_size": [512, 512],
        "annotation_scope": "all_expert_classes_reviewed",
    }), encoding="utf-8")
    with (metadata / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "image_group",
            "source_image_key",
            "image",
            "mask_class_id",
            "mask",
            "mask_foreground_pixels",
        ))
        writer.writeheader()
        writer.writerow({
            "image_group": "source",
            "source_image_key": "source",
            "image": image.relative_to(root).as_posix(),
            "mask_class_id": 1,
            "mask": mask.relative_to(root).as_posix(),
            "mask_foreground_pixels": 1,
        })

    groups = _dataset_release_groups(root)

    assert groups[0].dataset == "dataset20260101"
    assert expert_raw_ids("scratch_crack", groups[0].dataset) == (1, 11)
    assert groups[0].samples[0].image == image.resolve()


def test_training_plan_accepts_dynamic_release_identity(tmp_path: Path) -> None:
    images = []
    for index in range(3):
        path = tmp_path / f"image_{index}.png"
        _save(path, np.full((512, 512, 3), index, dtype=np.uint8))
        images.append(path)
    mask = tmp_path / "target.png"
    target = np.zeros((512, 512), dtype=np.uint8)
    target[0, 0] = 1
    _save(mask, target)

    mask_sha = hashlib.sha256(mask.read_bytes()).hexdigest()

    def row(dataset: str, group: str, tile: str, image: Path) -> dict[str, object]:
        return {
            "dataset": dataset,
            "source_group": group,
            "source_image_key": group,
            "tile": tile,
            "image": str(image),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "mask_format": "binary_target",
            "mask": str(mask),
            "mask_sha256": mask_sha,
            "mask_foreground_pixels": 1,
        }

    release_id = "dataset20260101"
    training = [
        row("dataset_jacky", "jacky-source", f"jacky-{index:03}.png", images[0])
        for index in range(743)
    ]
    training.append(row(release_id, "new-training-source", "new.png", images[0]))
    validation_groups = ("val-a", "val-b", "val-c")
    test_groups = ("test-a", "test-b")
    validation = [
        row(release_id, group, f"{group}.png", images[1])
        for group in validation_groups
    ]
    test = [
        row(release_id, group, f"{group}.png", images[2])
        for group in test_groups
    ]
    partitions = {"training": training, "validation": validation, "test": test}
    membership = {
        name: [{key: item[key] for key in ("dataset", "source_group", "tile")} for item in rows]
        for name, rows in partitions.items()
    }
    adopted_content = [
        _content_identity(item)
        for name in ("training", "validation", "test")
        for item in partitions[name]
    ]
    manifest = {
        "schema_version": 8,
        "expert": "scratch_crack",
        "dataset_release_id": release_id,
        "dataset_release": {"id": release_id},
        "expert_raw_ids": [1],
        "dataset_expert_raw_ids": {"dataset_jacky": [1], release_id: [1, 11]},
        "dataset_sha256": "dataset-hash",
        "adopted_dataset_sha256": _canonical_hash(adopted_content),
        "class_sha256": "class-hash",
        "split_sha256": _canonical_hash(membership),
        "dataset_policy": {
            "dataset_jacky": "all_743_training_only",
            release_id: "fixed_benchmark_groups_new_groups_training",
        },
        "split_contract": {
            "validation_groups": list(validation_groups),
            "test_groups": list(test_groups),
            "unused_holdout_groups": [],
            "unlisted_release_groups": "training",
        },
        "target_materialization": {
            "format": "uint8_binary_target",
            "values": {"background": 0, "foreground": 1, "ignore": 255},
        },
        "unused_holdout": [],
        "kyt_7px_audit": {},
        "partition_tiles": {name: len(rows) for name, rows in partitions.items()},
        "partition_positive_pixels": {name: len(rows) for name, rows in partitions.items()},
        "leakage_audit": {"passed": True},
        **partitions,
    }
    manifest_path = tmp_path / "scratch_crack.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = prepare_expert_data_plan(manifest_path, "scratch_crack")

    assert plan.dataset_release_id == release_id
    assert len(plan.train) == 744
