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


def prepare_expert_data_plan(manifest_path: str | Path, expert: str) -> ExpertDataPlan:
    """Load and exhaustively validate the authoritative one-fold manifest."""

    if expert not in EXPERT_RAW_IDS:
        raise ExpertDataContractError(f"unsupported expert {expert!r}; expected one of {tuple(EXPERT_RAW_IDS)}")
    manifest_path = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertDataContractError(f"cannot read manifest {manifest_path}: {error}") from error
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
