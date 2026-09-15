"""Validated Jacky-training/reference-validation plan for three binary experts."""

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
EXPERT_PARTITION_COUNTS: dict[str, tuple[int, int]] = {
    "shrinkage_craquelure": (605, 138),
    "scratch_crack": (601, 142),
    "loss": (588, 155),
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
    "mask",
    "image_sha256",
    "mask_sha256",
}


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
    partition_pixels: Mapping[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "expert": self.expert,
            "foreground_raw_ids": list(self.raw_ids),
            "target_rule": "constituent raw IDs are foreground; all other known IDs are negative",
            "dataset_sha256": self.dataset_sha256,
            "adopted_dataset_sha256": self.adopted_dataset_sha256,
            "class_sha256": self.class_sha256,
            "split_sha256": self.split_sha256,
            "partition_tiles": {"training": len(self.train), "validation": len(self.validation)},
            "partition_positive_pixels": dict(self.partition_pixels),
            "dataset_policy": dict(self.dataset_policy),
            "outer_test": "skipped",
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
    unknown = sorted(set(int(value) for value in np.unique(mask_array)) - KNOWN_RAW_IDS)
    if unknown:
        raise ExpertDataContractError(f"unknown raw IDs {unknown} in {mask}")
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
    if payload.get("schema_version") != 4:
        raise ExpertDataContractError("expert training requires schema_version 4 manifest")
    if payload.get("expert") != expert:
        raise ExpertDataContractError("manifest expert does not match the requested expert")
    if tuple(payload.get("expert_raw_ids", ())) != EXPERT_RAW_IDS[expert]:
        raise ExpertDataContractError("manifest foreground IDs do not match the approved expert contract")
    dataset_policy = payload.get("dataset_policy")
    if (
        not isinstance(dataset_policy, dict)
        or dataset_policy.get("dataset_jacky")
        != "fourteen_source_groups_training_two_source_groups_validation"
    ):
        raise ExpertDataContractError("manifest must use the approved dataset_jacky source-group split")
    if dataset_policy.get("dataset") != "excluded" or dataset_policy.get("dataset114") != "excluded":
        raise ExpertDataContractError("only dataset_jacky may be present")
    audit = payload.get("leakage_audit", {})
    if audit.get("passed") is not True:
        raise ExpertDataContractError("manifest leakage audit did not pass")
    duplicate_audit = payload.get("duplicate_audit", {})
    if duplicate_audit.get("passed") is not True or duplicate_audit.get("suspected_matches"):
        raise ExpertDataContractError("manifest duplicate audit did not pass")

    partitions = {
        name: tuple(_validate_row(row, name) for row in payload.get(name, ()))
        for name in ("training", "validation")
    }
    train, validation = partitions["training"], partitions["validation"]
    if not train or not validation:
        raise ExpertDataContractError("training and validation partitions must both be non-empty")
    if any(row["dataset"] != "dataset_jacky" for row in (*train, *validation)):
        raise ExpertDataContractError("training and validation must contain only dataset_jacky")
    if not any(row["dataset"] == "dataset_jacky" for row in train):
        raise ExpertDataContractError("training must include dataset_jacky")
    identities = {
        name: {(row["dataset"], row["tile"]) for row in rows}
        for name, rows in partitions.items()
    }
    groups = {
        name: {(row["dataset"], row["source_group"]) for row in rows}
        for name, rows in partitions.items()
    }
    if identities["training"] & identities["validation"] or groups["training"] & groups["validation"]:
        raise ExpertDataContractError("training/validation identity or source-group leakage")
    if len(identities["training"]) != len(train) or len(identities["validation"]) != len(validation):
        raise ExpertDataContractError("duplicate dataset-qualified tile identity")
    expected_train, expected_validation = EXPERT_PARTITION_COUNTS[expert]
    if (len(train), len(validation)) != (expected_train, expected_validation):
        raise ExpertDataContractError("partition tile counts do not match the approved expert split")
    if (len(groups["training"]), len(groups["validation"])) != (14, 2):
        raise ExpertDataContractError("approved split requires fourteen training groups and two validation groups")
    selected_groups = payload.get("selected_validation", {}).get("source_groups")
    if not isinstance(selected_groups, list) or set(selected_groups) != {
        row["source_group"] for row in validation
    }:
        raise ExpertDataContractError("selected validation groups do not match validation membership")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in rows]
        for name, rows in partitions.items()
    }
    if _canonical_hash(membership) != payload.get("split_sha256"):
        raise ExpertDataContractError("split membership hash mismatch")
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in ("training", "validation")
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
                mask = np.asarray(mask_file)
            total += int(np.isin(mask, raw_ids).sum())
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
    def __init__(self, rows: Sequence[Mapping[str, str]], *, train_augmentation: bool) -> None:
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
            image, mask = _augment_training_pair(image, mask)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return {
            "image": (image_tensor - _MEAN) / _STD,
            "mask": torch.from_numpy(mask.astype(np.int64, copy=False)),
            "name": f"{row['dataset']}__{row['tile']}",
            "dataset": row["dataset"],
            "source_group": row["source_group"],
        }


def denormalize_image(image: Tensor) -> Tensor:
    return (image.detach().cpu() * _STD + _MEAN).clamp(0.0, 1.0)
