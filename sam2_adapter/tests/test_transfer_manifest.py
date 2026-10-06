"""Validation of the portable Dataset115 + Jacky source-group transfer manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from sam2_adapter.expert_training_data import prepare_locked_expert_data_plan
from sam3_adapter.expert_training_data import (
    ExpertDataContractError,
    prepare_expert_data_plan,
)

SPLIT_ID = "2026-09-25_expanded_dataset115_jacky_f1_ranked_source_group_v1"
DATASET_POLICY = {
    "dataset_jacky": "training_only_all_743_tiles",
    "dataset115_filtered": "source_group_locked_train_validation_test",
    "checkpoint_selection": "validation_only",
    "test": "evaluate_once_after_checkpoint_selection",
}
SPLIT_POLICY = {
    "strategy": "current_inference_f1_ranked_source_group",
    "threshold": 0.5,
    "source_group_constraint": "disjoint_across_training_validation_test",
    "jacky_policy": "all_tiles_training_only",
    "test_selection": "highest_clean_candidate_pooled_pixel_f1",
    "validation_selection": "next_clean_candidate_groups_by_pooled_pixel_f1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_pair(root: Path, dataset: str, name: str, color: int, mask_value: int) -> tuple[str, str, str, str]:
    image_path = root / dataset / f"{name}.png"
    mask_path = root / dataset / f"{name}_mask.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    pixels[:, :, 0] = color
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[0, 0] = mask_value
    Image.fromarray(pixels, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    return (
        image_path.relative_to(root).as_posix(),
        mask_path.relative_to(root).as_posix(),
        _sha256(image_path),
        _sha256(mask_path),
    )


def _build_manifest(root: Path) -> dict[str, Any]:
    train_pair = _write_pair(root, "dataset115_filtered", "train", 10, 255)
    validation_pair = _write_pair(root, "dataset115_filtered", "validation", 20, 255)
    test_pair = _write_pair(root, "dataset115_filtered", "test", 30, 255)
    jacky_pair = _write_pair(root, "dataset_jacky", "jacky", 40, 2)
    classes = root / "dataset_jacky/classes.txt"
    classes.write_text("0 background\n2 loss\n255 ignore\n", encoding="utf-8")

    def row(dataset: str, group: str, tile: str, pair: tuple[str, str, str, str], encoding: str) -> dict[str, str]:
        return {
            "dataset": dataset,
            "source_group": group,
            "tile": tile,
            "image": pair[0],
            "mask": pair[1],
            "mask_encoding": encoding,
            "image_sha256": pair[2],
            "mask_sha256": pair[3],
        }

    training = [row("dataset115_filtered", "train_group", "train.png", train_pair, "binary_uint8_0_255")]
    for index in range(743):
        group = "train_group" if index == 0 else f"jacky_group_{(index - 1) % 15:02d}"
        training.append(row("dataset_jacky", group, f"jacky_{index:04d}.png", jacky_pair, "class_index_uint8"))
    validation = [row("dataset115_filtered", "validation_group", "validation.png", validation_pair, "binary_uint8_0_255")]
    test = [row("dataset115_filtered", "test_group", "test.png", test_pair, "binary_uint8_0_255")]
    partitions = {"training": training, "validation": validation, "test": test}
    for name, rows in partitions.items():
        partitions[name] = sorted(
            rows,
            key=lambda item: (0 if item["dataset"] == "dataset_jacky" else 1, item["source_group"], item["tile"]),
        )

    inventory = {
        dataset: {
            "tiles": sum(item["dataset"] == dataset for rows in partitions.values() for item in rows),
            "source_groups": len(
                {item["source_group"] for rows in partitions.values() for item in rows if item["dataset"] == dataset}
            ),
        }
        for dataset in ("dataset115_filtered", "dataset_jacky")
    }
    partition_stats = {}
    locked_groups = {}
    for name, rows in partitions.items():
        by_dataset = {
            dataset: [item for item in rows if item["dataset"] == dataset]
            for dataset in ("dataset115_filtered", "dataset_jacky")
        }
        partition_stats[name] = {
            "tiles": len(rows),
            "dataset_tiles": {dataset: len(items) for dataset, items in by_dataset.items()},
            "dataset_source_groups": {
                dataset: len({item["source_group"] for item in items}) for dataset, items in by_dataset.items()
            },
        }
        locked_groups[name] = {
            dataset: sorted({item["source_group"] for item in items}) for dataset, items in by_dataset.items()
        }

    ranked = [
        {
            "source_group": "test_group",
            "partition": "test",
            "tile_count": 1,
            "current_inference_pooled_f1": 0.9,
            "previous_run_role": "not_in_original_split",
            "rank_all_97": 1,
            "rank_in_clean_test_candidate_pool": 1,
            "jacky_same_source_group_collision": False,
        },
        {
            "source_group": "validation_group",
            "partition": "validation",
            "tile_count": 1,
            "current_inference_pooled_f1": 0.8,
            "previous_run_role": "not_in_original_split",
            "rank_all_97": 2,
            "rank_in_clean_test_candidate_pool": 2,
            "jacky_same_source_group_collision": False,
        },
        {
            "source_group": "train_group",
            "partition": "training",
            "tile_count": 1,
            "current_inference_pooled_f1": 0.7,
            "previous_run_role": "not_in_original_split",
            "rank_all_97": 3,
            "rank_in_clean_test_candidate_pool": None,
            "jacky_same_source_group_collision": True,
        },
    ]
    collision_groups = ["train_group"]
    membership = {
        name: [{key: item[key] for key in ("dataset", "source_group", "tile")} for item in rows]
        for name, rows in partitions.items()
    }
    adopted = [
        {key: item[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in ("training", "validation", "test")
        for item in partitions[name]
    ]
    inventory_content = sorted(adopted, key=lambda item: (item["dataset"], item["source_group"], item["tile"]))
    test_membership = [
        {key: item[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for item in partitions["test"]
    ]
    source_hashes = {"dataset_jacky/classes.txt": _sha256(classes)}
    return {
        "schema_version": 8,
        "split_id": SPLIT_ID,
        "expert": "loss",
        "expert_raw_ids": [2],
        "target_contract": {},
        "dataset_policy": DATASET_POLICY,
        "dataset_roots": {
            "dataset115_filtered": "dataset115_filtered",
            "dataset_jacky": "dataset_jacky",
        },
        "split_policy": SPLIT_POLICY,
        "inventory": {**inventory, "logical_total_tiles": sum(item["tiles"] for item in inventory.values())},
        "partition_stats": partition_stats,
        "locked_groups": locked_groups,
        "jacky_source_group_collisions": collision_groups,
        "ranked_dataset115_groups": ranked,
        "source_hashes": source_hashes,
        "dataset_sha256": _canonical_hash(inventory_content),
        "adopted_dataset_sha256": _canonical_hash(adopted),
        "class_sha256": source_hashes["dataset_jacky/classes.txt"],
        "split_sha256": _canonical_hash(membership),
        "test_membership_sha256": _canonical_hash(test_membership),
        "exclusion_sha256": _canonical_hash([]),
        **partitions,
    }


def test_transfer_manifest_keeps_all_jacky_in_train_for_sam3_and_sam2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_build_manifest(tmp_path), ensure_ascii=False), encoding="utf-8")

    sam3_plan = prepare_expert_data_plan(manifest_path, "loss")
    assert sum(row["dataset"] == "dataset_jacky" for row in sam3_plan.train) == 743
    assert {row["source_group"] for row in sam3_plan.validation} == {"validation_group"}
    assert {row["source_group"] for row in sam3_plan.test} == {"test_group"}

    sam2_plan = prepare_locked_expert_data_plan(manifest_path, "loss")
    record = sam2_plan.record()
    assert record["partition_tiles"] == {"training": 744, "validation": 1, "test": 1, "excluded": 0}
    assert record["validation_groups"] == ["validation_group"]
    assert record["test_groups"] == ["test_group"]
    assert record["reference_run"] == SPLIT_ID


def test_transfer_manifest_rejects_a_source_group_split_across_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = _build_manifest(tmp_path)
    manifest["validation"][0]["source_group"] = "train_group"
    manifest["inventory"]["dataset115_filtered"]["source_groups"] = 2
    manifest["locked_groups"]["validation"]["dataset115_filtered"] = ["train_group"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ExpertDataContractError, match="source-group leakage"):
        prepare_expert_data_plan(manifest_path, "loss")
