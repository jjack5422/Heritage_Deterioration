"""Validated two-dataset train/validation/test plan for three binary experts."""

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
EXPERT_DATASET_RAW_IDS: dict[str, dict[str, tuple[int, ...]]] = {
    "scratch_crack": {"dataset_jacky": (1,), "dataset115_filtered": (1, 11)},
    "shrinkage_craquelure": {"dataset_jacky": (3, 4), "dataset115_filtered": (3, 4)},
    "loss": {"dataset_jacky": (2,), "dataset115_filtered": (2,)},
}
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
    "image_sha256",
    "mask_format",
}


class ExpertDataContractError(ValueError):
    """Raised when the locked expert manifest is incomplete or unsafe."""


@dataclass(frozen=True)
class ExpertDataPlan:
    manifest_path: Path
    expert: str
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
            "foreground_raw_ids": list(self.raw_ids),
            "dataset_foreground_raw_ids": {
                dataset: list(raw_ids) for dataset, raw_ids in self.dataset_raw_ids.items()
            },
            "target_rule": "constituent raw IDs are foreground; all other known IDs are negative",
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
            "outer_test": "dataset115_filtered only; excluded from checkpoint selection",
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
    identity: dict[str, Any] = {
        key: row[key]
        for key in ("dataset", "source_group", "tile", "image_sha256", "mask_format")
    }
    if row["mask_format"] == "class_index":
        identity["mask_sha256"] = row["mask_sha256"]
    else:
        identity["positive_mask_sha256"] = list(row["positive_mask_sha256"])
    return identity


def _validate_row(row: object, partition: str) -> dict[str, Any]:
    if not isinstance(row, dict) or not _REQUIRED_ROW_KEYS.issubset(row):
        raise ExpertDataContractError(f"{partition} row lacks required fields: {row!r}")
    normalized: dict[str, Any] = {key: str(row[key]) for key in _REQUIRED_ROW_KEYS}
    image = Path(normalized["image"])
    if not image.is_file() or _sha256(image) != normalized["image_sha256"]:
        raise ExpertDataContractError(f"missing image or content hash mismatch: {image}")
    with Image.open(image) as image_file:
        if image_file.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ExpertDataContractError(f"expected a 512x512 image: {image}")

    if normalized["mask_format"] == "class_index":
        if not {"mask", "mask_sha256"}.issubset(row):
            raise ExpertDataContractError(f"class-index row lacks mask metadata: {row!r}")
        normalized.update(mask=str(row["mask"]), mask_sha256=str(row["mask_sha256"]))
        mask = Path(normalized["mask"])
        if not mask.is_file() or _sha256(mask) != normalized["mask_sha256"]:
            raise ExpertDataContractError(f"missing mask or content hash mismatch: {mask}")
        with Image.open(mask) as mask_file:
            mask_array = np.asarray(mask_file)
        if mask_array.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ExpertDataContractError(f"expected a 512x512 mask: {mask}")
        unknown = sorted(set(int(value) for value in np.unique(mask_array)) - KNOWN_RAW_IDS)
        if unknown:
            raise ExpertDataContractError(f"unknown raw IDs {unknown} in {mask}")
    elif normalized["mask_format"] == "binary_multilabel":
        paths, hashes = row.get("positive_masks"), row.get("positive_mask_sha256")
        if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes):
            raise ExpertDataContractError("binary multilabel row has invalid positive mask metadata")
        normalized["positive_masks"] = [str(path) for path in paths]
        normalized["positive_mask_sha256"] = [str(value) for value in hashes]
        for path_text, expected_hash in zip(normalized["positive_masks"], normalized["positive_mask_sha256"], strict=True):
            mask = Path(path_text)
            if not mask.is_file() or _sha256(mask) != expected_hash:
                raise ExpertDataContractError(f"missing mask or content hash mismatch: {mask}")
            with Image.open(mask) as mask_file:
                mask_array = np.asarray(mask_file)
            if mask_array.shape != (IMAGE_SIZE, IMAGE_SIZE) or not set(np.unique(mask_array)).issubset({0, 255}):
                raise ExpertDataContractError(f"invalid 512x512 binary mask: {mask}")
    else:
        raise ExpertDataContractError(f"unknown mask format {normalized['mask_format']!r}")
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
    if schema not in (5, 6):
        raise ExpertDataContractError("expert training requires schema_version 5 or 6 manifest")
    if payload.get("expert") != expert:
        raise ExpertDataContractError("manifest expert does not match the requested expert")
    if tuple(payload.get("expert_raw_ids", ())) != EXPERT_RAW_IDS[expert]:
        raise ExpertDataContractError("manifest foreground IDs do not match the approved expert contract")
    dataset_raw_ids_payload = payload.get("dataset_expert_raw_ids")
    if not isinstance(dataset_raw_ids_payload, dict):
        raise ExpertDataContractError("manifest lacks dataset-specific foreground IDs")
    dataset_raw_ids = {
        str(dataset): tuple(int(raw_id) for raw_id in raw_ids)
        for dataset, raw_ids in dataset_raw_ids_payload.items()
    }
    if dataset_raw_ids != EXPERT_DATASET_RAW_IDS[expert]:
        raise ExpertDataContractError("dataset-specific foreground IDs do not match the approved expert contract")
    dataset_policy = payload.get("dataset_policy")
    if not isinstance(dataset_policy, dict):
        raise ExpertDataContractError("manifest lacks dataset policy")
    expected_policy = (
        {"dataset_jacky": "all_743_training_only", "dataset115_filtered": "fixed_source_group_train_validation_test"}
        if schema == 6 else
        {"dataset_jacky": "training_or_validation_only", "dataset115_filtered": "training_validation_or_test"}
    )
    if dataset_policy != expected_policy:
        raise ExpertDataContractError("manifest dataset policy does not match schema")
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
    if any(row["dataset"] == "dataset_jacky" for row in test):
        raise ExpertDataContractError("dataset_jacky cannot appear in test")
    if schema == 6:
        jacky_train = [row for row in train if row["dataset"] == "dataset_jacky"]
        if len(jacky_train) != 743 or any(row["dataset"] == "dataset_jacky" for row in validation):
            raise ExpertDataContractError("locked transfer requires all 743 Jacky tiles in training only")
        fixed_groups = {
            "scratch_crack": ({"KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "MST-SC-M-A2-2-7"}, {"KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A6'-2"}, (1241, 105, 112)),
            "loss": ({"KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "KJWTomh-PH-M-1RB1-1"}, {"MST-SC-M-A2-2-7", "KJTHT-SC-R-A4-6", "KJWTomh-MH-M-A3E-3-2", "MST-SC-M-A2-2-6"}, (1243, 108, 107)),
            "shrinkage_craquelure": ({"KJWTomh-MH-M-A3E-1", "WFT-PH-M-1LB1-1-1", "KJWTomh-PH-M-1RB1-1"}, {"KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1"}, (1081, 123, 142)),
        }
        expected_validation, expected_test, expected_counts = fixed_groups[expert]
        actual_validation = {row["source_group"] for row in validation}
        actual_test = {row["source_group"] for row in test}
        if actual_validation != expected_validation or actual_test != expected_test:
            raise ExpertDataContractError("locked transfer source-group membership differs")
        if (len(train), len(validation), len(test)) != expected_counts:
            raise ExpertDataContractError("locked transfer tile counts differ")
        if expert == "shrinkage_craquelure":
            holdout = {row["source_group"] for row in payload.get("unused_holdout", ())}
            if holdout != {"KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A3E-3-2"}:
                raise ExpertDataContractError("craquelure unused holdout groups differ")
            audit = payload.get("kyt_7px_audit", {})
            if set(audit) != expected_test or any(int(audit[group]["tiles"]) != count for group, count in (("KYT-SC-1R-A9-4", 80), ("KYT-SC-1R-2LB1-1", 62))):
                raise ExpertDataContractError("KYT 7px audit is incomplete")
    allowed_datasets = {"dataset_jacky", "dataset115_filtered"}
    if any(row["dataset"] not in allowed_datasets for rows in partitions.values() for row in rows):
        raise ExpertDataContractError("manifest contains an unsupported dataset")
    identities = {
        name: {(row["dataset"], row["tile"]) for row in rows}
        for name, rows in partitions.items()
    }
    groups = {
        name: {(row["dataset"], row["source_group"]) for row in rows}
        for name, rows in partitions.items()
    }
    partition_names = tuple(partitions)
    for index, left in enumerate(partition_names):
        if len(identities[left]) != len(partitions[left]):
            raise ExpertDataContractError(f"duplicate dataset-qualified tile identity in {left}")
        for right in partition_names[index + 1:]:
            if identities[left] & identities[right] or groups[left] & groups[right]:
                raise ExpertDataContractError(f"identity or source-image leakage between {left} and {right}")
    total_tiles = sum(len(rows) for rows in partitions.values())
    ratios = {name: len(rows) / total_tiles for name, rows in partitions.items()}
    targets = {"training": 0.70, "validation": 0.15, "test": 0.15}
    if schema == 5 and any(abs(ratios[name] - targets[name]) > 0.025 for name in targets):
        raise ExpertDataContractError(f"partition ratios are outside the 70/15/15 tolerance: {ratios}")
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

    raw_ids = EXPERT_RAW_IDS[expert]
    declared_pixels = payload.get("partition_positive_pixels")
    if not isinstance(declared_pixels, dict):
        raise ExpertDataContractError("manifest lacks partition positive-pixel counts")
    positive_pixels: dict[str, int] = {}
    for name, rows in partitions.items():
        total = 0
        for row in rows:
            if row["mask_format"] == "class_index":
                with Image.open(row["mask"]) as mask_file:
                    mask = np.asarray(mask_file)
                total += int(np.isin(mask, raw_ids).sum())
            else:
                union = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
                for path in row["positive_masks"]:
                    with Image.open(path) as mask_file:
                        union |= np.asarray(mask_file) > 0
                total += int(union.sum())
        if total == 0:
            raise ExpertDataContractError(f"{expert} has no positive pixels in {name}")
        if total != int(declared_pixels.get(name, -1)):
            raise ExpertDataContractError(f"{expert} positive-pixel count mismatch in {name}")
        positive_pixels[name] = total
    return ExpertDataPlan(
        manifest_path=manifest_path,
        expert=expert,
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


def make_expert_target(mask: Tensor, raw_ids: Sequence[int]) -> Tensor:
    """Map a multiclass mask to one binary target while preserving ignore pixels."""
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
    def __init__(self, rows: Sequence[Mapping[str, Any]], *, raw_ids: Sequence[int], train_augmentation: bool) -> None:
        self.rows = tuple(rows)
        self.raw_ids = tuple(raw_ids)
        self.train_augmentation = train_augmentation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        with Image.open(row["image"]) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8).copy()
        if row["mask_format"] == "class_index":
            with Image.open(row["mask"]) as mask_file:
                raw_mask = np.asarray(mask_file, dtype=np.uint8).copy()
            mask = np.zeros_like(raw_mask, dtype=np.uint8)
            for raw_id in self.raw_ids:
                mask[raw_mask == raw_id] = 1
            mask[raw_mask == IGNORE_VALUE] = IGNORE_VALUE
        else:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            for path in row["positive_masks"]:
                with Image.open(path) as mask_file:
                    mask[np.asarray(mask_file) > 0] = 1
        if self.train_augmentation:
            image, mask = _augment_training_pair(image, mask)
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
