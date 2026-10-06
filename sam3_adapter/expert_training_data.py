"""Validated Jacky-training and Dataset115 train/validation/test plans."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_SIZE = 512
IGNORE_VALUE = 255
KNOWN_RAW_IDS = frozenset({*range(38), IGNORE_VALUE})
EXPERT_RAW_IDS: dict[str, tuple[int, ...]] = {
    "shrinkage_craquelure": (3, 4),
    "scratch_crack": (1,),
    "loss": (2,),
}
EXPERT_PARTITION_COUNTS: dict[str, tuple[int, int, int]] = {
    "shrinkage_craquelure": (1137, 123, 198),
    "scratch_crack": (1241, 105, 112),
    "loss": (1243, 108, 107),
}
EXPERT_DATASET115_COUNTS: dict[str, tuple[int, int, int]] = {
    "shrinkage_craquelure": (394, 123, 198),
    "scratch_crack": (498, 105, 112),
    "loss": (500, 108, 107),
}
EXCLUDED_TRAINING_GROUPS: dict[str, tuple[str, ...]] = {
    "shrinkage_craquelure": (),
    "scratch_crack": (),
    "loss": (),
}
EXPECTED_EXCLUDED_TRAINING_COUNTS = {
    "shrinkage_craquelure": 0,
    "scratch_crack": 0,
    "loss": 0,
}
PARTITIONS = ("training", "validation", "test")
_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
_BRIGHTNESS_RANGE = (0.85, 1.15)
_CONTRAST_RANGE = (0.85, 1.15)
_GAMMA_RANGE = (0.85, 1.15)
_CHANNEL_GAIN_RANGE = (0.95, 1.05)
_REQUIRED_ROW_KEYS = {
    "dataset",
    "source_group",
    "tile",
    "image",
    "mask",
    "mask_encoding",
    "image_sha256",
    "mask_sha256",
}
_MASK_ENCODINGS = frozenset({"class_index_uint8", "binary_uint8_0_255"})


class ExpertDataContractError(ValueError):
    """Raised when the locked expert manifest is incomplete or unsafe."""


@dataclass(frozen=True)
class ExpertDataPlan:
    manifest_path: Path
    expert: str
    raw_ids: tuple[int, ...]
    dataset_sha256: str
    adopted_dataset_sha256: str
    class_sha256: str
    split_sha256: str
    dataset_policy: Mapping[str, str]
    train: tuple[Mapping[str, str], ...]
    validation: tuple[Mapping[str, str], ...]
    test: tuple[Mapping[str, str], ...]
    partition_pixels: Mapping[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "expert": self.expert,
            "foreground_raw_ids": list(self.raw_ids),
            "target_rule": "Jacky raw IDs or Dataset115 expert binary masks become foreground",
            "dataset_sha256": self.dataset_sha256,
            "adopted_dataset_sha256": self.adopted_dataset_sha256,
            "class_sha256": self.class_sha256,
            "split_sha256": self.split_sha256,
            "partition_tiles": {
                "training": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "partition_positive_pixels": dict(self.partition_pixels),
            "dataset_policy": dict(self.dataset_policy),
            "outer_test": "evaluate_once_after_validation_checkpoint_selection",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_row(row: object, partition: str) -> dict[str, str]:
    if not isinstance(row, dict) or not _REQUIRED_ROW_KEYS.issubset(row):
        raise ExpertDataContractError(f"{partition} row lacks required fields: {row!r}")
    normalized = {key: str(row[key]) for key in _REQUIRED_ROW_KEYS}
    image, mask = Path(normalized["image"]), Path(normalized["mask"])
    if not image.is_file() or not mask.is_file():
        raise ExpertDataContractError(f"missing image/mask pair: {image}, {mask}")
    if _sha256(image) != normalized["image_sha256"] or _sha256(mask) != normalized["mask_sha256"]:
        raise ExpertDataContractError(f"content hash mismatch for {normalized['dataset']}:{normalized['tile']}")
    with Image.open(image) as image_file:
        image_size = image_file.size
    with Image.open(mask) as mask_file:
        mask_array = np.asarray(mask_file)
    if image_size != (IMAGE_SIZE, IMAGE_SIZE) or mask_array.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ExpertDataContractError(f"expected a 512x512 RGB/mask pair: {image}, {mask}")
    encoding = normalized["mask_encoding"]
    if encoding not in _MASK_ENCODINGS:
        raise ExpertDataContractError(f"unsupported mask encoding {encoding!r}")
    values = {int(value) for value in np.unique(mask_array)}
    if encoding == "class_index_uint8":
        unknown = sorted(values - KNOWN_RAW_IDS)
        if normalized["dataset"] != "dataset_jacky" or unknown:
            raise ExpertDataContractError(f"invalid Jacky class-index mask {mask}: unknown={unknown}")
    elif normalized["dataset"] != "dataset115_filtered" or not values.issubset({0, 255}):
        raise ExpertDataContractError(f"invalid Dataset115 binary mask {mask}: values={sorted(values)}")
    return normalized


def _transfer_sort_rows(rows: Sequence[Mapping[str, str]]) -> tuple[Mapping[str, str], ...]:
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

def _prepare_group_transfer_data_plan(
    manifest_path: Path,
    expert: str,
    payload: Mapping[str, Any],
) -> ExpertDataPlan:
    """Validate a portable, pre-split, source-group-locked transfer manifest."""

    if payload.get("expert") != expert:
        raise ExpertDataContractError("manifest expert does not match the requested expert")
    if tuple(payload.get("expert_raw_ids", ())) != EXPERT_RAW_IDS[expert]:
        raise ExpertDataContractError("manifest foreground IDs do not match the approved expert contract")
    expected_dataset_policy = {
        "dataset_jacky": "training_only_all_743_tiles",
        "dataset115_filtered": "source_group_locked_train_validation_test",
        "checkpoint_selection": "validation_only",
        "test": "evaluate_once_after_checkpoint_selection",
    }
    if payload.get("dataset_policy") != expected_dataset_policy:
        raise ExpertDataContractError("transfer manifest dataset policy is invalid")
    expected_split_policy = {
        "strategy": "current_inference_f1_ranked_source_group",
        "threshold": 0.5,
        "source_group_constraint": "disjoint_across_training_validation_test",
        "jacky_policy": "all_tiles_training_only",
        "test_selection": "highest_clean_candidate_pooled_pixel_f1",
        "validation_selection": "next_clean_candidate_groups_by_pooled_pixel_f1",
    }
    if payload.get("split_policy") != expected_split_policy:
        raise ExpertDataContractError("transfer manifest split policy is invalid")
    if payload.get("split_id") != "2026-09-25_expanded_dataset115_jacky_f1_ranked_source_group_v1":
        raise ExpertDataContractError("transfer manifest split_id is not the supported expanded-data split")
    dataset_roots = payload.get("dataset_roots")
    if not isinstance(dataset_roots, dict) or set(dataset_roots) != {"dataset115_filtered", "dataset_jacky"}:
        raise ExpertDataContractError("transfer manifest must map both dataset roots")
    relative_roots: dict[str, Path] = {}
    for dataset, root_value in dataset_roots.items():
        root = Path(str(root_value))
        if root.is_absolute() or not root.parts or ".." in root.parts:
            raise ExpertDataContractError(f"dataset root must be a safe project-relative path: {root_value}")
        relative_roots[dataset] = root

    partitions: dict[str, tuple[dict[str, str], ...]] = {}
    for name in PARTITIONS:
        raw_rows = payload.get(name)
        if not isinstance(raw_rows, list):
            raise ExpertDataContractError(f"{name} partition must be a list")
        validated: list[dict[str, str]] = []
        for row in raw_rows:
            normalized = _validate_row(row, name)
            for field in ("image", "mask"):
                path = Path(normalized[field])
                root = relative_roots[normalized["dataset"]]
                if path.is_absolute() or path.parts[: len(root.parts)] != root.parts or len(path.parts) <= len(root.parts):
                    raise ExpertDataContractError(f"transfer path is outside its dataset root: {path}")
            validated.append(normalized)
        partitions[name] = _transfer_sort_rows(validated)
        if not partitions[name]:
            raise ExpertDataContractError(f"{name} partition must not be empty")

    rows_by_partition = partitions
    all_rows = tuple(row for name in PARTITIONS for row in rows_by_partition[name])
    identities = {
        (row["dataset"], row["tile"])
        for row in all_rows
    }
    if len(identities) != len(all_rows):
        raise ExpertDataContractError("transfer manifest contains duplicate dataset-qualified tile identities")

    actual_inventory = {
        dataset: {
            "tiles": sum(row["dataset"] == dataset for row in all_rows),
            "source_groups": len({row["source_group"] for row in all_rows if row["dataset"] == dataset}),
        }
        for dataset in ("dataset115_filtered", "dataset_jacky")
    }
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict) or any(inventory.get(key) != value for key, value in actual_inventory.items()):
        raise ExpertDataContractError("transfer manifest inventory counts do not match its rows")
    logical_tiles = sum(value["tiles"] for value in actual_inventory.values())
    if inventory.get("logical_total_tiles") != logical_tiles:
        raise ExpertDataContractError("transfer manifest logical inventory count does not match its rows")
    if actual_inventory["dataset_jacky"]["tiles"] != 743:
        raise ExpertDataContractError("all 743 dataset_jacky tiles must be included")
    if any(row["dataset"] == "dataset_jacky" for name in ("validation", "test") for row in rows_by_partition[name]):
        raise ExpertDataContractError("dataset_jacky may appear only in training")
    if any(row["dataset"] != "dataset115_filtered" for name in ("validation", "test") for row in rows_by_partition[name]):
        raise ExpertDataContractError("validation and test must contain only dataset115_filtered")

    actual_stats: dict[str, dict[str, Any]] = {}
    actual_groups: dict[str, dict[str, list[str]]] = {}
    partition_group_keys: dict[str, set[tuple[str, str]]] = {}
    partition_image_hashes: dict[str, set[str]] = {}
    for name, rows in rows_by_partition.items():
        dataset_tiles = {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in ("dataset115_filtered", "dataset_jacky")
        }
        dataset_groups = {
            dataset: sorted({row["source_group"] for row in rows if row["dataset"] == dataset})
            for dataset in ("dataset115_filtered", "dataset_jacky")
        }
        actual_stats[name] = {
            "tiles": len(rows),
            "dataset_tiles": dataset_tiles,
            "dataset_source_groups": {dataset: len(groups) for dataset, groups in dataset_groups.items()},
        }
        actual_groups[name] = dataset_groups
        partition_group_keys[name] = {
            (row["dataset"], row["source_group"])
            for row in rows
        }
        partition_image_hashes[name] = {row["image_sha256"] for row in rows}
    if payload.get("partition_stats") != actual_stats:
        raise ExpertDataContractError("transfer manifest partition statistics do not match its rows")
    if payload.get("locked_groups") != actual_groups:
        raise ExpertDataContractError("transfer manifest group membership does not match its rows")

    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            if partition_group_keys[left] & partition_group_keys[right]:
                raise ExpertDataContractError(f"{left}/{right} source-group leakage")
            if partition_image_hashes[left] & partition_image_hashes[right]:
                raise ExpertDataContractError(f"{left}/{right} image-hash leakage")
    d115_groups = {
        row["source_group"]
        for row in all_rows
        if row["dataset"] == "dataset115_filtered"
    }
    jacky_groups = {
        row["source_group"]
        for row in all_rows
        if row["dataset"] == "dataset_jacky"
    }
    collisions = sorted(d115_groups & jacky_groups)
    if payload.get("jacky_source_group_collisions") != collisions:
        raise ExpertDataContractError("cross-dataset source-group collision list does not match the inventory")
    for group in collisions:
        if group not in actual_groups["training"]["dataset115_filtered"] or group not in actual_groups["training"]["dataset_jacky"]:
            raise ExpertDataContractError(f"shared source group {group!r} must stay in training")

    ranked_groups = payload.get("ranked_dataset115_groups")
    if not isinstance(ranked_groups, list):
        raise ExpertDataContractError("ranked Dataset115 source-group records are required")
    ranked_by_group = {str(row.get("source_group")): row for row in ranked_groups if isinstance(row, dict)}
    if len(ranked_by_group) != len(ranked_groups) or set(ranked_by_group) != d115_groups:
        raise ExpertDataContractError("ranked Dataset115 groups do not match the data inventory")
    d115_tile_counts = {
        group: sum(row["source_group"] == group and row["dataset"] == "dataset115_filtered" for row in all_rows)
        for group in d115_groups
    }
    for group, record in ranked_by_group.items():
        partition = next(
            name for name in PARTITIONS
            if group in actual_groups[name]["dataset115_filtered"]
        )
        if record.get("partition") != partition or record.get("tile_count") != d115_tile_counts[group]:
            raise ExpertDataContractError(f"ranked group record drifted for {group}")
        if partition in {"validation", "test"} and record.get("previous_run_role") in {"training", "validation"}:
            raise ExpertDataContractError(f"historical training/validation group {group} reused as held-out data")
    test_ranks = sorted(
        record.get("rank_in_clean_test_candidate_pool")
        for record in ranked_by_group.values()
        if record.get("partition") == "test"
    )
    validation_ranks = sorted(
        record.get("rank_in_clean_test_candidate_pool")
        for record in ranked_by_group.values()
        if record.get("partition") == "validation"
    )
    if test_ranks != list(range(1, len(test_ranks) + 1)):
        raise ExpertDataContractError("test candidate ranks must be contiguous from the top")
    if validation_ranks != list(range(len(test_ranks) + 1, len(test_ranks) + len(validation_ranks) + 1)):
        raise ExpertDataContractError("validation candidate ranks must follow the selected test ranks")
    for partition in ("test", "validation"):
        ranked_scores = [
            float(record["current_inference_pooled_f1"])
            for record in sorted(
                (r for r in ranked_by_group.values() if r.get("partition") == partition),
                key=lambda r: r["rank_in_clean_test_candidate_pool"],
            )
        ]
        if any(left < right for left, right in zip(ranked_scores, ranked_scores[1:])):
            raise ExpertDataContractError(f"{partition} groups are not ordered by descending inference F1")

    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ExpertDataContractError("transfer manifest source hashes are required")
    for relative_path, expected_hash in source_hashes.items():
        source_path = Path(relative_path)
        if source_path.is_absolute() or not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ExpertDataContractError(f"transfer source hash mismatch: {relative_path}")
    classes_path = (relative_roots["dataset_jacky"] / "classes.txt").as_posix()
    if payload.get("class_sha256") != source_hashes.get(classes_path):
        raise ExpertDataContractError("Jacky class hash does not match the transfer source record")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in rows]
        for name, rows in rows_by_partition.items()
    }
    if _canonical_hash(membership) != payload.get("split_sha256"):
        raise ExpertDataContractError("transfer split membership hash mismatch")
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in PARTITIONS
        for row in rows_by_partition[name]
    ]
    if _canonical_hash(adopted_content) != payload.get("adopted_dataset_sha256"):
        raise ExpertDataContractError("transfer adopted dataset hash mismatch")
    inventory_content = sorted(
        adopted_content,
        key=lambda row: (row["dataset"], row["source_group"], row["tile"]),
    )
    if _canonical_hash(inventory_content) != payload.get("dataset_sha256"):
        raise ExpertDataContractError("transfer dataset inventory hash mismatch")
    test_membership = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for row in rows_by_partition["test"]
    ]
    if _canonical_hash(test_membership) != payload.get("test_membership_sha256"):
        raise ExpertDataContractError("transfer test membership hash mismatch")
    if _canonical_hash([]) != payload.get("exclusion_sha256"):
        raise ExpertDataContractError("transfer manifests may not declare excluded rows")

    positive_pixels: dict[str, int] = {}
    raw_ids = EXPERT_RAW_IDS[expert]
    for name, rows in rows_by_partition.items():
        total = 0
        for row in rows:
            with Image.open(row["mask"]) as mask_file:
                mask = torch.from_numpy(np.asarray(mask_file, dtype=np.uint8).copy())
            target = make_expert_target(mask, raw_ids, row["mask_encoding"])
            total += int((target == 1).sum())
        if total == 0:
            raise ExpertDataContractError(f"{expert} has no positive pixels in {name}")
        positive_pixels[name] = total
    return ExpertDataPlan(
        manifest_path=manifest_path,
        expert=expert,
        raw_ids=raw_ids,
        dataset_sha256=str(payload["dataset_sha256"]),
        adopted_dataset_sha256=str(payload["adopted_dataset_sha256"]),
        class_sha256=str(payload["class_sha256"]),
        split_sha256=str(payload["split_sha256"]),
        dataset_policy={str(key): str(value) for key, value in expected_dataset_policy.items()},
        train=rows_by_partition["training"],
        validation=rows_by_partition["validation"],
        test=rows_by_partition["test"],
        partition_pixels=positive_pixels,
    )


def prepare_expert_data_plan(manifest_path: str | Path, expert: str) -> ExpertDataPlan:
    """Load and exhaustively validate the authoritative one-fold manifest."""

    if expert not in EXPERT_RAW_IDS:
        raise ExpertDataContractError(f"unsupported expert {expert!r}; expected one of {tuple(EXPERT_RAW_IDS)}")
    manifest_path = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertDataContractError(f"cannot read manifest {manifest_path}: {error}") from error
    if payload.get("schema_version") == 8:
        return _prepare_group_transfer_data_plan(manifest_path, expert, payload)
    if payload.get("schema_version") != 6:
        raise ExpertDataContractError("expert training requires schema_version 6 manifest")
    if payload.get("expert") != expert:
        raise ExpertDataContractError("manifest expert does not match the requested expert")
    if tuple(payload.get("expert_raw_ids", ())) != EXPERT_RAW_IDS[expert]:
        raise ExpertDataContractError("manifest foreground IDs do not match the approved expert contract")
    dataset_policy = payload.get("dataset_policy")
    expected_policy = {
        "dataset_jacky": "training_only_all_743_tiles",
        "dataset115_filtered": "source_group_locked_train_validation_test",
        "checkpoint_selection": "validation_only",
        "test": "evaluate_once_after_checkpoint_selection",
    }
    if dataset_policy != expected_policy:
        raise ExpertDataContractError("manifest dataset policy does not match the approved combined-data contract")
    if payload.get("exclusion_policy") != "declared source groups are omitted before training partition assembly":
        raise ExpertDataContractError("manifest exclusion policy does not match the approved contract")
    if payload.get("leakage_audit", {}).get("passed") is not True:
        raise ExpertDataContractError("manifest leakage audit did not pass")
    if payload.get("duplicate_audit", {}).get("passed") is not True:
        raise ExpertDataContractError("manifest duplicate audit did not pass")

    partitions = {
        name: tuple(_validate_row(row, name) for row in payload.get(name, ()))
        for name in PARTITIONS
    }
    train, validation, test = (partitions[name] for name in PARTITIONS)
    if not train or not validation or not test:
        raise ExpertDataContractError("training, validation, and test partitions must all be non-empty")
    if sum(row["dataset"] == "dataset_jacky" for row in train) != 743:
        raise ExpertDataContractError("training must contain all 743 dataset_jacky tiles")
    if any(row["dataset"] == "dataset_jacky" for row in (*validation, *test)):
        raise ExpertDataContractError("dataset_jacky may only appear in training")
    if any(row["dataset"] != "dataset115_filtered" for row in (*validation, *test)):
        raise ExpertDataContractError("validation and test must contain only dataset115_filtered")

    expected_counts = EXPERT_PARTITION_COUNTS[expert]
    actual_counts = tuple(len(partitions[name]) for name in PARTITIONS)
    if actual_counts != expected_counts:
        raise ExpertDataContractError(f"partition tile counts drifted: {actual_counts} != {expected_counts}")
    actual_dataset115 = tuple(
        sum(row["dataset"] == "dataset115_filtered" for row in partitions[name])
        for name in PARTITIONS
    )
    if actual_dataset115 != EXPERT_DATASET115_COUNTS[expert]:
        raise ExpertDataContractError("Dataset115 partition counts do not match the approved split")

    identities = {
        name: {(row["dataset"], row["tile"]) for row in rows}
        for name, rows in partitions.items()
    }
    groups = {
        name: {(row["dataset"], row["source_group"]) for row in rows}
        for name, rows in partitions.items()
    }
    image_hashes = {
        name: {row["image_sha256"] for row in rows}
        for name, rows in partitions.items()
    }
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            if identities[left] & identities[right] or groups[left] & groups[right] or image_hashes[left] & image_hashes[right]:
                raise ExpertDataContractError(f"{left}/{right} identity, source-group, or image leakage")
    if any(len(identities[name]) != len(partitions[name]) for name in PARTITIONS):
        raise ExpertDataContractError("duplicate dataset-qualified tile identity")

    locked_groups = payload.get("locked_groups", {})
    for name in ("validation", "test"):
        expected_groups = {("dataset115_filtered", str(group)) for group in locked_groups.get(name, ())}
        if groups[name] != expected_groups:
            raise ExpertDataContractError(f"{name} groups do not match the locked split")
    expected_excluded_groups = set(EXCLUDED_TRAINING_GROUPS[expert])
    if set(str(group) for group in locked_groups.get("excluded_training", ())) != expected_excluded_groups:
        raise ExpertDataContractError("excluded training groups do not match the approved contract")
    excluded_training = payload.get("excluded_training", ())
    if not isinstance(excluded_training, list):
        raise ExpertDataContractError("excluded training membership must be a list")
    if len(excluded_training) != EXPECTED_EXCLUDED_TRAINING_COUNTS[expert]:
        raise ExpertDataContractError("excluded training tile count does not match the approved contract")
    if any(
        row.get("dataset") != "dataset115_filtered"
        or row.get("source_group") not in expected_excluded_groups
        for row in excluded_training
    ):
        raise ExpertDataContractError("excluded training membership contains an unapproved record")
    excluded_identities = {
        (str(row.get("dataset")), str(row.get("tile")))
        for row in excluded_training
    }
    if len(excluded_identities) != len(excluded_training):
        raise ExpertDataContractError("excluded training membership contains duplicate tiles")
    if excluded_identities & set().union(*identities.values()):
        raise ExpertDataContractError("excluded training tile was adopted into a partition")
    if _canonical_hash(excluded_training) != payload.get("exclusion_sha256"):
        raise ExpertDataContractError("excluded training membership hash mismatch")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in rows]
        for name, rows in partitions.items()
    }
    if _canonical_hash(membership) != payload.get("split_sha256"):
        raise ExpertDataContractError("split membership hash mismatch")
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in PARTITIONS
        for row in partitions[name]
    ]
    if _canonical_hash(adopted_content) != payload.get("adopted_dataset_sha256"):
        raise ExpertDataContractError("adopted dataset hash mismatch")

    raw_ids = EXPERT_RAW_IDS[expert]
    positive_pixels: dict[str, int] = {}
    for name, rows in partitions.items():
        total = 0
        for row in rows:
            with Image.open(row["mask"]) as mask_file:
                mask = torch.from_numpy(np.asarray(mask_file, dtype=np.uint8).copy())
            target = make_expert_target(mask, raw_ids, row["mask_encoding"])
            total += int((target == 1).sum())
        if total == 0:
            raise ExpertDataContractError(f"{expert} has no positive pixels in {name}")
        positive_pixels[name] = total
    return ExpertDataPlan(
        manifest_path=manifest_path,
        expert=expert,
        raw_ids=raw_ids,
        dataset_sha256=str(payload["dataset_sha256"]),
        adopted_dataset_sha256=str(payload["adopted_dataset_sha256"]),
        class_sha256=str(payload["class_sha256"]),
        split_sha256=str(payload["split_sha256"]),
        dataset_policy={str(key): str(value) for key, value in dataset_policy.items()},
        train=train,
        validation=validation,
        test=test,
        partition_pixels=positive_pixels,
    )


def make_expert_target(
    mask: Tensor,
    raw_ids: Sequence[int],
    mask_encoding: str = "class_index_uint8",
) -> Tensor:
    """Map either supported mask encoding to foreground 1/background 0/ignore 255."""

    if mask_encoding == "binary_uint8_0_255":
        values = set(int(value) for value in torch.unique(mask).tolist())
        if not values.issubset({0, 255}):
            raise ExpertDataContractError(f"binary mask contains invalid values: {sorted(values)}")
        return (mask == 255).to(torch.long)
    if mask_encoding != "class_index_uint8":
        raise ExpertDataContractError(f"unsupported mask encoding {mask_encoding!r}")
    target = torch.zeros_like(mask, dtype=torch.long)
    for raw_id in raw_ids:
        target[mask == raw_id] = 1
    target[mask == IGNORE_VALUE] = IGNORE_VALUE
    return target


def _photometric_augment(image: np.ndarray) -> np.ndarray:
    """Apply conservative RGB-only intensity and color variation."""

    values = image.astype(np.float32) / 255.0
    mean = values.mean(axis=(0, 1), keepdims=True)
    values = (values - mean) * random.uniform(*_CONTRAST_RANGE) + mean
    values *= random.uniform(*_BRIGHTNESS_RANGE)
    values = np.clip(values, 0.0, 1.0) ** random.uniform(*_GAMMA_RANGE)
    channel_gains = np.asarray(
        [random.uniform(*_CHANNEL_GAIN_RANGE) for _ in range(3)],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    return np.rint(np.clip(values * channel_gains, 0.0, 1.0) * 255.0).astype(np.uint8)


def _augment_training_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact shared geometry and RGB-only photometric augmentation."""

    quarter_turns = random.randrange(4)
    if quarter_turns:
        image = np.rot90(image, quarter_turns, axes=(0, 1)).copy()
        mask = np.rot90(mask, quarter_turns, axes=(0, 1)).copy()
    if random.random() < 0.5:
        image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
    if random.random() < 0.5:
        image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
    return _photometric_augment(image), mask


class ExpertTileDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        raw_ids: Sequence[int],
        train_augmentation: bool,
    ) -> None:
        self.rows = tuple(rows)
        self.raw_ids = tuple(raw_ids)
        self.train_augmentation = train_augmentation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        with Image.open(row["image"]) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(row["mask"]) as mask_file:
            mask = np.asarray(mask_file, dtype=np.uint8).copy()
        if self.train_augmentation:
            image, mask = _augment_training_pair(image, mask)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        mask_tensor = torch.from_numpy(mask.astype(np.int64, copy=False))
        return {
            "image": (image_tensor - _MEAN) / _STD,
            "target": make_expert_target(mask_tensor, self.raw_ids, row["mask_encoding"]),
            "name": f"{row['dataset']}__{row['tile']}",
            "dataset": row["dataset"],
            "source_group": row["source_group"],
        }


def denormalize_image(image: Tensor) -> Tensor:
    return (image.detach().cpu() * _STD + _MEAN).clamp(0.0, 1.0)
