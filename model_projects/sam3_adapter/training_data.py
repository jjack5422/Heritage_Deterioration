"""Validated prepared binary-target data for three SAM3-Adapter experts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from model_projects.sam3_adapter.data_augmentation import augment_training_pair

IMAGE_SIZE = 512
IGNORE_VALUE = 255
EXPERT_RAW_IDS: dict[str, tuple[int, ...]] = {
    "shrinkage_craquelure": (3, 4),
    "scratch_crack": (1,),
    "loss": (2,),
}
DATASET_RELEASE_RAW_IDS: dict[str, tuple[int, ...]] = {
    "scratch_crack": (1, 11),
    "shrinkage_craquelure": (3, 4),
    "loss": (2,),
}
_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
_REQUIRED_ROW_KEYS = {
    "dataset",
    "source_group",
    "tile",
    "image",
    "image_sha256",
    "mask_format",
    "mask",
    "mask_sha256",
    "mask_foreground_pixels",
}


class ExpertDataContractError(ValueError):
    """Raised when the locked expert manifest is incomplete or unsafe."""


@dataclass(frozen=True)
class ExpertDataPlan:
    manifest_path: Path
    expert: str
    dataset_release_id: str
    raw_ids: tuple[int, ...]
    dataset_raw_ids: Mapping[str, tuple[int, ...]]
    dataset_sha256: str
    adopted_dataset_sha256: str
    class_sha256: str
    split_sha256: str
    dataset_policy: Mapping[str, str]
    train: tuple[Mapping[str, Any], ...]
    validation: tuple[Mapping[str, Any], ...]
    test: tuple[Mapping[str, Any], ...]
    partition_pixels: Mapping[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "expert": self.expert,
            "dataset_release_id": self.dataset_release_id,
            "foreground_raw_ids": list(self.raw_ids),
            "dataset_foreground_raw_ids": {
                dataset: list(raw_ids) for dataset, raw_ids in self.dataset_raw_ids.items()
            },
            "target_rule": "prepared 0/1/255 binary target loaded without runtime class-ID conversion",
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
            "outer_test": f"{self.dataset_release_id} only; excluded from checkpoint selection",
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


def _content_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("dataset", "source_group", "tile", "image_sha256", "mask_format", "mask_sha256")
    }


def _validate_row(row: object, partition: str) -> dict[str, Any]:
    if not isinstance(row, dict) or not _REQUIRED_ROW_KEYS.issubset(row):
        raise ExpertDataContractError(f"{partition} row lacks required fields: {row!r}")
    normalized: dict[str, Any] = {
        key: str(row[key])
        for key in _REQUIRED_ROW_KEYS
        if key != "mask_foreground_pixels"
    }
    normalized["mask_foreground_pixels"] = int(row["mask_foreground_pixels"])
    image = Path(normalized["image"])
    if not image.is_file() or _sha256(image) != normalized["image_sha256"]:
        raise ExpertDataContractError(f"missing image or content hash mismatch: {image}")
    with Image.open(image) as image_file:
        if image_file.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ExpertDataContractError(f"expected a 512x512 image: {image}")

    if normalized["mask_format"] != "binary_target":
        raise ExpertDataContractError(
            f"training requires a prepared binary_target mask, got {normalized['mask_format']!r}"
        )
    mask = Path(normalized["mask"])
    if not mask.is_file() or _sha256(mask) != normalized["mask_sha256"]:
        raise ExpertDataContractError(f"missing mask or content hash mismatch: {mask}")
    with Image.open(mask) as mask_file:
        mask_array = np.asarray(mask_file, dtype=np.uint8)
    if mask_array.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ExpertDataContractError(f"expected a 512x512 binary target: {mask}")
    values = set(int(value) for value in np.unique(mask_array))
    if not values.issubset({0, 1, IGNORE_VALUE}):
        raise ExpertDataContractError(f"binary target must contain only 0/1/255: {mask}")
    foreground_pixels = int((mask_array == 1).sum())
    if foreground_pixels != normalized["mask_foreground_pixels"]:
        raise ExpertDataContractError(f"binary target foreground count mismatch: {mask}")
    return normalized


def prepare_expert_data_plan(manifest_path: str | Path, expert: str) -> ExpertDataPlan:
    """Load and exhaustively validate the authoritative one-fold manifest."""

    if expert not in EXPERT_RAW_IDS:
        raise ExpertDataContractError(f"unsupported expert {expert!r}; expected one of {tuple(EXPERT_RAW_IDS)}")
    manifest_path = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertDataContractError(f"cannot read manifest {manifest_path}: {error}") from error
    schema = payload.get("schema_version")
    if schema != 8:
        raise ExpertDataContractError("expert training requires prepared schema_version 8 manifest")
    if payload.get("expert") != expert:
        raise ExpertDataContractError("manifest expert does not match the requested expert")
    if tuple(payload.get("expert_raw_ids", ())) != EXPERT_RAW_IDS[expert]:
        raise ExpertDataContractError("manifest foreground IDs do not match the approved expert contract")
    dataset_release = payload.get("dataset_release")
    dataset_release_id = payload.get("dataset_release_id")
    if (
        not isinstance(dataset_release, dict)
        or not isinstance(dataset_release_id, str)
        or not dataset_release_id
        or dataset_release_id == "dataset_jacky"
        or dataset_release.get("id") != dataset_release_id
    ):
        raise ExpertDataContractError("manifest lacks a valid Dataset115 release identity")
    dataset_raw_ids_payload = payload.get("dataset_expert_raw_ids")
    if not isinstance(dataset_raw_ids_payload, dict):
        raise ExpertDataContractError("manifest lacks dataset-specific foreground IDs")
    dataset_raw_ids = {
        str(dataset): tuple(int(raw_id) for raw_id in raw_ids)
        for dataset, raw_ids in dataset_raw_ids_payload.items()
    }
    expected_dataset_raw_ids = {
        "dataset_jacky": EXPERT_RAW_IDS[expert],
        dataset_release_id: DATASET_RELEASE_RAW_IDS[expert],
    }
    if dataset_raw_ids != expected_dataset_raw_ids:
        raise ExpertDataContractError("dataset-specific foreground IDs do not match the approved expert contract")
    target_materialization = payload.get("target_materialization")
    if not isinstance(target_materialization, dict) or (
        target_materialization.get("format") != "uint8_binary_target"
        or target_materialization.get("values") != {"background": 0, "foreground": 1, "ignore": 255}
    ):
        raise ExpertDataContractError("manifest lacks the prepared binary-target contract")
    dataset_policy = payload.get("dataset_policy")
    expected_policy = {
        "dataset_jacky": "all_743_training_only",
        dataset_release_id: "fixed_benchmark_groups_new_groups_training",
    }
    if dataset_policy != expected_policy:
        raise ExpertDataContractError("manifest dataset policy does not match schema")
    split_contract = payload.get("split_contract")
    if not isinstance(split_contract, dict) or split_contract.get("unlisted_release_groups") != "training":
        raise ExpertDataContractError("manifest lacks the dynamic release split contract")
    expected_validation = {str(group) for group in split_contract.get("validation_groups", ())}
    expected_test = {str(group) for group in split_contract.get("test_groups", ())}
    expected_holdout = {str(group) for group in split_contract.get("unused_holdout_groups", ())}
    if not expected_validation or not expected_test or expected_validation & expected_test:
        raise ExpertDataContractError("manifest split contract has invalid validation/test groups")
    audit = payload.get("leakage_audit", {})
    if audit.get("passed") is not True:
        raise ExpertDataContractError("manifest leakage audit did not pass")
    partitions = {
        name: tuple(_validate_row(row, name) for row in payload.get(name, ()))
        for name in ("training", "validation", "test")
    }
    train, validation, test = partitions["training"], partitions["validation"], partitions["test"]
    if not train or not validation or not test:
        raise ExpertDataContractError("training, validation, and test partitions must all be non-empty")
    jacky_train = [row for row in train if row["dataset"] == "dataset_jacky"]
    if len(jacky_train) != 743 or any(
        row["dataset"] == "dataset_jacky" for row in (*validation, *test)
    ):
        raise ExpertDataContractError("all 743 Jacky tiles must remain in training only")
    allowed_datasets = {"dataset_jacky", dataset_release_id}
    if any(row["dataset"] not in allowed_datasets for rows in partitions.values() for row in rows):
        raise ExpertDataContractError("manifest contains an unsupported dataset release")
    if any(row["dataset"] != dataset_release_id for row in (*validation, *test)):
        raise ExpertDataContractError("validation and test must come from the selected Dataset115 release")
    actual_validation = {row["source_group"] for row in validation}
    actual_test = {row["source_group"] for row in test}
    if actual_validation != expected_validation or actual_test != expected_test:
        raise ExpertDataContractError("manifest rows differ from the declared split contract")
    holdout = {str(row["source_group"]) for row in payload.get("unused_holdout", ())}
    if holdout != expected_holdout:
        raise ExpertDataContractError("manifest unused holdout differs from the split contract")
    if expert == "shrinkage_craquelure":
        kyt_groups = expected_test & {"KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1"}
        kyt_audit = payload.get("kyt_7px_audit", {})
        test_group_counts = {
            group: sum(row["source_group"] == group for row in test)
            for group in kyt_groups
        }
        if set(kyt_audit) != kyt_groups or any(
            int(kyt_audit[group]["tiles"]) != test_group_counts[group]
            for group in kyt_groups
        ):
            raise ExpertDataContractError("KYT-specific 7px audit is incomplete")
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
    partition_names = tuple(partitions)
    for index, left in enumerate(partition_names):
        if len(identities[left]) != len(partitions[left]):
            raise ExpertDataContractError(f"duplicate dataset-qualified tile identity in {left}")
        for right in partition_names[index + 1:]:
            if identities[left] & identities[right] or groups[left] & groups[right]:
                raise ExpertDataContractError(f"identity or source-image leakage between {left} and {right}")
            if image_hashes[left] & image_hashes[right]:
                raise ExpertDataContractError(f"image content leakage between {left} and {right}")
    if expert == "loss":
        required = "KJWTomh-SC-M-A7'-1"
        membership_names = [
            name for name, rows in partitions.items()
            if any(row["source_group"] == required for row in rows)
        ]
        if membership_names != ["training"]:
            raise ExpertDataContractError(f"{required} must occur in the loss training partition")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in rows]
        for name, rows in partitions.items()
    }
    if _canonical_hash(membership) != payload.get("split_sha256"):
        raise ExpertDataContractError("split membership hash mismatch")
    adopted_content = [
        _content_identity(row)
        for name in ("training", "validation", "test")
        for row in partitions[name]
    ]
    if _canonical_hash(adopted_content) != payload.get("adopted_dataset_sha256"):
        raise ExpertDataContractError("adopted dataset hash mismatch")

    declared_tiles = payload.get("partition_tiles")
    actual_tiles = {name: len(rows) for name, rows in partitions.items()}
    if not isinstance(declared_tiles, dict) or actual_tiles != {
        str(name): int(value) for name, value in declared_tiles.items()
    }:
        raise ExpertDataContractError("partition tile counts do not match the manifest")

    raw_ids = EXPERT_RAW_IDS[expert]
    declared_pixels = payload.get("partition_positive_pixels")
    if not isinstance(declared_pixels, dict):
        raise ExpertDataContractError("manifest lacks partition positive-pixel counts")
    positive_pixels = {
        name: sum(int(row["mask_foreground_pixels"]) for row in rows)
        for name, rows in partitions.items()
    }
    if any(total == 0 for total in positive_pixels.values()):
        raise ExpertDataContractError(f"{expert} has a partition with no positive pixels")
    if positive_pixels != {
        str(name): int(value)
        for name, value in declared_pixels.items()
    }:
        raise ExpertDataContractError(f"{expert} positive-pixel counts do not match the manifest")
    return ExpertDataPlan(
        manifest_path=manifest_path,
        expert=expert,
        dataset_release_id=dataset_release_id,
        raw_ids=raw_ids,
        dataset_raw_ids=dataset_raw_ids,
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


class ExpertTileDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], *, train_augmentation: bool) -> None:
        self.rows = tuple(rows)
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
            image, mask = augment_training_pair(image, mask)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return {
            "image": (image_tensor - _MEAN) / _STD,
            "target": torch.from_numpy(mask.astype(np.int64, copy=False)),
            "name": f"{row['dataset']}__{row['tile']}",
            "dataset": row["dataset"],
            "source_group": row["source_group"],
        }


def denormalize_image(image: Tensor) -> Tensor:
    return (image.detach().cpu() * _STD + _MEAN).clamp(0.0, 1.0)
