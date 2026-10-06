"""Locked three-expert partitions for the approved SAM2-Adapter comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from sam3_adapter.expert_training_data import (
    EXPERT_RAW_IDS,
    ExpertTileDataset,
    denormalize_image,
    make_expert_target,
    prepare_expert_data_plan,
)

PARTITIONS = ("training", "validation", "test")
REFERENCE_RUNS = {
    "loss": "2026-09-16_three-experts_sam3-adapter-1008_jacky-dataset115_seed42",
    "scratch_crack": "2026-09-16_three-experts_sam3-adapter-1008_jacky-dataset115_seed42",
    "shrinkage_craquelure": "2026-09-17_craquelure-sam3-adapter-1008_no-kyt-2lb1_epochs60_seed42",
}
VALIDATION_GROUPS = {
    "loss": frozenset({"KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "KJWTomh-PH-M-1RB1-1"}),
    "scratch_crack": frozenset({"KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "MST-SC-M-A2-2-7"}),
    "shrinkage_craquelure": frozenset(
        {"KJWTomh-MH-M-A3E-1", "KJWTomh-PH-M-1RB1-1", "WFT-PH-M-1LB1-1-1"}
    ),
}
TEST_GROUPS = {
    "loss": frozenset(
        {"KJTHT-SC-R-A4-6", "KJWTomh-MH-M-A3E-3-2", "MST-SC-M-A2-2-6", "MST-SC-M-A2-2-7"}
    ),
    "scratch_crack": frozenset({"KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A6'-2"}),
    "shrinkage_craquelure": frozenset({"KYT-SC-1R-2LB1-1", "KYT-SC-1R-A9-4"}),
}
EXCLUDED_GROUPS = {
    "loss": frozenset(),
    "scratch_crack": frozenset(),
    "shrinkage_craquelure": frozenset({"KJWTomh-MH-M-A3E-3-2"}),
}
EXPECTED_COUNTS = {
    "loss": (1243, 108, 107, 0),
    "scratch_crack": (1241, 105, 112, 0),
    "shrinkage_craquelure": (1137, 123, 142, 56),
}
EXPECTED_POSITIVE_PIXELS = {
    "loss": (4_546_071, 90_175, 149_752, 0),
    "scratch_crack": (1_624_331, 132_702, 265_207, 0),
    "shrinkage_craquelure": (7_633_764, 198_072, 2_126_351, 326_650),
}
EXPECTED_SPLIT_SHA256 = {
    "loss": "c3cca35f5e9cc601574aa5536479c5eeb2264668ab3bdfda3030f585c73c9730",
    "scratch_crack": "ebd6abcb6cd5d1f652874f39b27fcb6d9ab2ea5292f61c865d89b62f46e6e756",
    "shrinkage_craquelure": "5f5a2e4230fbce2d146f09ec7b2224e60e2507b88ac5579154271faf8d9186c4",
}
EXPECTED_ADOPTED_SHA256 = {
    "loss": "1ec2042c7448535bef718b3ab0e15e9c72d2838b206d7a4b565dc707f9af9972",
    "scratch_crack": "69746aa69c5279a607d2c618334260d20161a5a7cc6746f699761010379ece44",
    "shrinkage_craquelure": "8d9fc4a4389360ffdd5797f621b32e5c01f419f83654fdae8618cbad699b5224",
}
EXPECTED_EXCLUSION_SHA256 = {
    "loss": hashlib.sha256(b"[]").hexdigest(),
    "scratch_crack": hashlib.sha256(b"[]").hexdigest(),
    "shrinkage_craquelure": "e324fc7591b73d1a024d373f6157d8497fb3a05b07f901a34218fba437f4b437",
}


class LockedExpertDataError(ValueError):
    """Raised when the approved comparison inventory or partition has drifted."""


@dataclass(frozen=True)
class LockedExpertDataPlan:
    manifest_path: Path
    expert: str
    raw_ids: tuple[int, ...]
    inventory_dataset_sha256: str
    inventory_adopted_dataset_sha256: str
    split_sha256: str
    adopted_dataset_sha256: str
    exclusion_sha256: str
    reference_run: str
    train: tuple[Mapping[str, str], ...]
    validation: tuple[Mapping[str, str], ...]
    test: tuple[Mapping[str, str], ...]
    excluded: tuple[Mapping[str, str], ...]
    partition_pixels: Mapping[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "manifest_inventory": str(self.manifest_path),
            "expert": self.expert,
            "foreground_raw_ids": list(self.raw_ids),
            "target_rule": "Jacky raw IDs or Dataset115 expert binary masks become foreground",
            "reference_run": self.reference_run,
            "inventory_dataset_sha256": self.inventory_dataset_sha256,
            "inventory_adopted_dataset_sha256": self.inventory_adopted_dataset_sha256,
            "split_sha256": self.split_sha256,
            "adopted_dataset_sha256": self.adopted_dataset_sha256,
            "exclusion_sha256": self.exclusion_sha256,
            "partition_tiles": {
                "training": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
                "excluded": len(self.excluded),
            },
            "partition_positive_pixels": dict(self.partition_pixels),
            "validation_groups": sorted(
                {
                    row["source_group"]
                    for row in self.validation
                    if row["dataset"] == "dataset115_filtered"
                }
            ),
            "test_groups": sorted(
                {
                    row["source_group"]
                    for row in self.test
                    if row["dataset"] == "dataset115_filtered"
                }
            ),
            "excluded_groups": sorted(
                {
                    row["source_group"]
                    for row in self.excluded
                    if row["dataset"] == "dataset115_filtered"
                }
            ),
            "dataset_policy": {
                "dataset_jacky": "training_only_all_743_tiles",
                "dataset115_filtered": "source_group_locked_train_validation_test",
                "checkpoint_selection": "validation_only",
                "test": "evaluate_once_after_checkpoint_selection",
            },
            "outer_test": "evaluate_once_after_validation_checkpoint_selection",
        }


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sort_rows(rows: Sequence[Mapping[str, str]]) -> tuple[Mapping[str, str], ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                0 if row["dataset"] == "dataset_jacky" else 1,
                row["source_group"],
                row["tile"],
            ),
        )
    )


def _positive_pixels(rows: Sequence[Mapping[str, str]], raw_ids: Sequence[int]) -> int:
    total = 0
    for row in rows:
        with Image.open(row["mask"]) as mask_file:
            mask = torch.from_numpy(np.asarray(mask_file, dtype=np.uint8).copy())
        target = make_expert_target(mask, raw_ids, row["mask_encoding"])
        total += int((target == 1).sum())
    return total


def prepare_locked_expert_data_plan(manifest_path: str | Path, expert: str) -> LockedExpertDataPlan:
    """Validate the live inventory and reconstruct the approved immutable partition."""

    inventory = prepare_expert_data_plan(manifest_path, expert)
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == 8:
        return LockedExpertDataPlan(
            manifest_path=inventory.manifest_path,
            expert=expert,
            raw_ids=inventory.raw_ids,
            inventory_dataset_sha256=inventory.dataset_sha256,
            inventory_adopted_dataset_sha256=inventory.adopted_dataset_sha256,
            split_sha256=inventory.split_sha256,
            adopted_dataset_sha256=inventory.adopted_dataset_sha256,
            exclusion_sha256=str(payload["exclusion_sha256"]),
            reference_run=str(payload["split_id"]),
            train=inventory.train,
            validation=inventory.validation,
            test=inventory.test,
            excluded=(),
            partition_pixels={**inventory.partition_pixels, "excluded": 0},
        )
    unique: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in (*inventory.train, *inventory.validation, *inventory.test):
        identity = (row["dataset"], row["tile"])
        if identity in unique:
            raise LockedExpertDataError(f"duplicate inventory identity: {identity}")
        unique[identity] = row
    validation_groups = VALIDATION_GROUPS[expert]
    test_groups = TEST_GROUPS[expert]
    excluded_groups = EXCLUDED_GROUPS[expert]
    if validation_groups & test_groups or validation_groups & excluded_groups or test_groups & excluded_groups:
        raise LockedExpertDataError(f"overlapping locked groups for {expert}")

    partitions: dict[str, list[Mapping[str, str]]] = {name: [] for name in PARTITIONS}
    excluded: list[Mapping[str, str]] = []
    for row in unique.values():
        if row["dataset"] == "dataset_jacky":
            partitions["training"].append(row)
        elif row["source_group"] in validation_groups:
            partitions["validation"].append(row)
        elif row["source_group"] in test_groups:
            partitions["test"].append(row)
        elif row["source_group"] in excluded_groups:
            excluded.append(row)
        else:
            partitions["training"].append(row)

    ordered = {name: _sort_rows(rows) for name, rows in partitions.items()}
    excluded_rows = _sort_rows(excluded)
    counts = tuple(len(ordered[name]) for name in PARTITIONS) + (len(excluded_rows),)
    if counts != EXPECTED_COUNTS[expert]:
        raise LockedExpertDataError(f"partition counts drifted for {expert}: {counts} != {EXPECTED_COUNTS[expert]}")
    if sum(row["dataset"] == "dataset_jacky" for row in ordered["training"]) != 743:
        raise LockedExpertDataError("all 743 Jacky tiles must be training-only")
    if any(row["dataset"] == "dataset_jacky" for name in ("validation", "test") for row in ordered[name]):
        raise LockedExpertDataError("Jacky tile leaked outside training")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in ordered[name]]
        for name in PARTITIONS
    }
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in PARTITIONS
        for row in ordered[name]
    ]
    excluded_content = [
        {
            **{key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")},
            "positive_pixels": _positive_pixels((row,), inventory.raw_ids),
        }
        for row in excluded_rows
    ]
    split_sha256 = _canonical_hash(membership)
    adopted_sha256 = _canonical_hash(adopted_content)
    exclusion_sha256 = _canonical_hash(excluded_content)
    for label, actual, expected in (
        ("split", split_sha256, EXPECTED_SPLIT_SHA256[expert]),
        ("adopted dataset", adopted_sha256, EXPECTED_ADOPTED_SHA256[expert]),
        ("exclusion", exclusion_sha256, EXPECTED_EXCLUSION_SHA256[expert]),
    ):
        if actual != expected:
            raise LockedExpertDataError(f"{label} hash drifted for {expert}: {actual} != {expected}")

    pixel_counts = {
        name: _positive_pixels(ordered[name], inventory.raw_ids)
        for name in PARTITIONS
    }
    pixel_counts["excluded"] = sum(int(row["positive_pixels"]) for row in excluded_content)
    observed_pixels = tuple(pixel_counts[name] for name in (*PARTITIONS, "excluded"))
    if observed_pixels != EXPECTED_POSITIVE_PIXELS[expert]:
        raise LockedExpertDataError(
            f"foreground pixels drifted for {expert}: {observed_pixels} != {EXPECTED_POSITIVE_PIXELS[expert]}"
        )

    return LockedExpertDataPlan(
        manifest_path=inventory.manifest_path,
        expert=expert,
        raw_ids=inventory.raw_ids,
        inventory_dataset_sha256=inventory.dataset_sha256,
        inventory_adopted_dataset_sha256=inventory.adopted_dataset_sha256,
        split_sha256=split_sha256,
        adopted_dataset_sha256=adopted_sha256,
        exclusion_sha256=exclusion_sha256,
        reference_run=REFERENCE_RUNS[expert],
        train=ordered["training"],
        validation=ordered["validation"],
        test=ordered["test"],
        excluded=excluded_rows,
        partition_pixels=pixel_counts,
    )


__all__ = [
    "EXPERT_RAW_IDS",
    "ExpertTileDataset",
    "LockedExpertDataError",
    "LockedExpertDataPlan",
    "denormalize_image",
    "prepare_locked_expert_data_plan",
]
