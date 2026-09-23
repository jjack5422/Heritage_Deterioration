"""Dataset validation, nested group splits, and 512px SAM2 input loading."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_SIZE = 512
IGNORE_VALUE = 255
RAW_CLASS_NAMES = (
    "background",
    "craquelure",
    "loss",
    "shrinkage",
    "flaking",
    "stain",
)
SCORED_PIXEL_NAMES = ("background", "foreground", "excluded")
CLASS_IDS = {name: index for index, name in enumerate(RAW_CLASS_NAMES)}
_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class DataContractError(ValueError):
    """Raised when a manifest, split, or image/mask pair violates H0's contract."""


@dataclass(frozen=True)
class H0DataPlan:
    """Immutable, validated nested train/validation/outer-test split."""

    root: Path
    manifest_hash: str
    image_manifest_hash: str
    mask_manifest_hash: str
    ignore_value: int
    outer_fold: int
    inner_fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    index: dict[str, dict[str, Any]]
    train_counts: tuple[int, int, int]
    val_counts: tuple[int, int, int]
    test_counts: tuple[int, int, int]

    def groups(self, names: Sequence[str]) -> set[str]:
        return {str(self.index[name]["source_group"]) for name in names}

    def record(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.root),
            "manifest_hash": self.manifest_hash,
            "image_manifest_hash": self.image_manifest_hash,
            "mask_manifest_hash": self.mask_manifest_hash,
            "raw_class_names": list(RAW_CLASS_NAMES),
            "target_class_names": ["background", "foreground"],
            "target_rule": "raw label 1 is foreground; raw label 0 is background; all other labels are excluded",
            "ignore_value": self.ignore_value,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "train": list(self.train),
            "validation": list(self.val),
            "outer_test": list(self.test),
            "pixel_counts": {
                "train": dict(zip(SCORED_PIXEL_NAMES, self.train_counts, strict=True)),
                "validation": dict(zip(SCORED_PIXEL_NAMES, self.val_counts, strict=True)),
                "outer_test": dict(zip(SCORED_PIXEL_NAMES, self.test_counts, strict=True)),
            },
            "group_counts": {
                "train": len(self.groups(self.train)),
                "validation": len(self.groups(self.val)),
                "outer_test": len(self.groups(self.test)),
            },
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DataContractError(f"missing required dataset file: {path}") from error
    except json.JSONDecodeError as error:
        raise DataContractError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError(f"expected a JSON object in {path}")
    return value


def _load_contract(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("training_eligible") is not True:
        raise DataContractError("manifest.json does not mark this dataset training_eligible")
    try:
        ids = manifest["label_contract"]["class_ids"]
        ignore_value = int(manifest["label_contract"].get("ignore_value", IGNORE_VALUE))
        pair_count = int(manifest["pair_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise DataContractError("manifest.json label contract is incomplete") from error
    if ids != CLASS_IDS:
        raise DataContractError(f"merged-crack training requires class IDs {CLASS_IDS!r}, got {ids!r}")
    if ignore_value != IGNORE_VALUE:
        raise DataContractError(f"H0 expects ignore value {IGNORE_VALUE}, got {ignore_value}")
    raw_items = _read_json(root / "tile_index.json").get("items")
    if not isinstance(raw_items, list):
        raise DataContractError("tile_index.json lacks an items array")
    index: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict) or not isinstance(item.get("tile"), str):
            raise DataContractError("tile_index.json item lacks a tile name")
        if item["tile"] in index:
            raise DataContractError(f"duplicate tile in tile_index: {item['tile']}")
        source_group = item.get("source_group")
        if not isinstance(source_group, str) or not source_group:
            match = re.match(r"(.+?)_R\d+_C\d+", Path(item["tile"]).stem)
            if match is None:
                raise DataContractError(
                    f"cannot derive source_group from tile name: {item['tile']!r}"
                )
            source_group = match.group(1)
        index[item["tile"]] = {**item, "source_group": source_group}
    if len(index) != pair_count:
        raise DataContractError(f"manifest pair_count={pair_count}, tile_index count={len(index)}")
    return manifest, index


def _load_fold(
    root: Path,
    *,
    fold: int,
    eligible: set[str],
    manifest: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    data = _read_json(root / "splits" / f"fold{fold}.json")
    if data.get("outer_fold") != fold:
        raise DataContractError(f"fold{fold}.json has an inconsistent outer_fold")
    try:
        train = tuple(data["folds"][0]["train"])
        holdout = tuple(data["holdout_tiles"])
        contract = data["data_contract"]
    except (KeyError, IndexError, TypeError) as error:
        raise DataContractError(f"fold{fold}.json has an incomplete split contract") from error
    if not all(isinstance(name, str) for name in (*train, *holdout)):
        raise DataContractError(f"fold{fold}.json contains a non-string tile ID")
    if len(train) != len(set(train)) or len(holdout) != len(set(holdout)):
        raise DataContractError(f"fold{fold}.json contains duplicate tiles")
    if set(train) & set(holdout) or set(train) | set(holdout) != eligible:
        raise DataContractError(f"fold{fold}.json does not partition tile_index exactly")
    expected = {
        "class_names": list(RAW_CLASS_NAMES),
        "ignore_value": IGNORE_VALUE,
        "task_image_manifest_sha256": manifest.get("image_manifest_sha256"),
        "task_mask_manifest_sha256": manifest.get("mask_manifest_sha256"),
    }
    actual = {key: contract.get(key) for key in expected}
    if actual != expected:
        raise DataContractError(f"fold{fold}.json data_contract does not match manifest")
    return train, holdout


def _count_pixels(root: Path, names: Sequence[str], ignore_value: int) -> tuple[int, int, int]:
    totals = np.zeros(3, dtype=np.int64)
    for name in names:
        image_path = root / "images" / name
        mask_path = root / "masks" / name
        if not image_path.is_file() or not mask_path.is_file():
            raise DataContractError(f"missing image/mask pair for {name}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise DataContractError(f"image/mask dimensions differ for {name}")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise DataContractError(
                    f"H0's locked 512px tile contract is violated by {name}: {image.size}"
                )
            values = np.asarray(mask)
        if values.ndim != 2:
            raise DataContractError(f"mask must be a single-channel label-ID image: {name}")
        unknown = set(np.unique(values).tolist()) - {*CLASS_IDS.values(), ignore_value}
        if unknown:
            raise DataContractError(f"mask {name} has unknown labels: {sorted(unknown)}")
        totals[0] += int((values == 0).sum())
        totals[1] += int((values == 1).sum())
        totals[2] += int(((values != 0) & (values != 1)).sum())
    return tuple(int(value) for value in totals)


def prepare_data_plan(root: str | Path, *, outer_fold: int) -> H0DataPlan:
    """Validate the merged-label data contract and make one nested CV split."""

    root = Path(root).resolve()
    manifest, index = _load_contract(root)
    eligible = set(index)
    fold_paths = sorted(root.joinpath("splits").glob("fold*.json"))
    folds = sorted(
        int(path.stem[4:]) for path in fold_paths if path.stem[4:].isdigit()
    )
    if outer_fold not in folds or len(folds) < 2:
        raise DataContractError("at least two valid numbered folds are required")
    outer_train, test = _load_fold(root, fold=outer_fold, eligible=eligible, manifest=manifest)
    inner_fold = folds[(folds.index(outer_fold) + 1) % len(folds)]
    _, inner_holdout = _load_fold(root, fold=inner_fold, eligible=eligible, manifest=manifest)
    validation_set = set(inner_holdout)
    if not validation_set.issubset(outer_train):
        raise DataContractError(
            f"inner fold {inner_fold} holdout is not contained in outer fold {outer_fold} training data"
        )
    train = tuple(name for name in outer_train if name not in validation_set)
    val = tuple(name for name in outer_train if name in validation_set)
    if not train or not val or not test:
        raise DataContractError("nested split has an empty train, validation, or outer-test partition")
    partitions = (train, val, test)
    group_sets = [{str(index[name]["source_group"]) for name in names} for names in partitions]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise DataContractError("source_group leakage across train/validation/outer-test partitions")
    ignore_value = int(manifest["label_contract"]["ignore_value"])
    return H0DataPlan(
        root=root,
        manifest_hash=str(manifest.get("manifest_sha256", "")),
        image_manifest_hash=str(manifest.get("image_manifest_sha256", "")),
        mask_manifest_hash=str(manifest.get("mask_manifest_sha256", "")),
        ignore_value=ignore_value,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        train=train,
        val=val,
        test=test,
        index=index,
        train_counts=_count_pixels(root, train, ignore_value),
        val_counts=_count_pixels(root, val, ignore_value),
        test_counts=_count_pixels(root, test, ignore_value),
    )


class H0TileDataset(Dataset[dict[str, Tensor | str]]):
    """Load a 512px RGB tile and its original merged-dataset label IDs."""

    def __init__(
        self,
        plan: H0DataPlan,
        names: Sequence[str],
        *,
        train_augmentation: bool,
    ) -> None:
        self.plan = plan
        self.names = tuple(names)
        self.train_augmentation = train_augmentation

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        name = self.names[index]
        with Image.open(self.plan.root / "images" / name) as image:
            rgb = image.convert("RGB")
        with Image.open(self.plan.root / "masks" / name) as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8).copy()
        if rgb.size != (IMAGE_SIZE, IMAGE_SIZE) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise DataContractError(f"unexpected tile size during load: {name}")
        image_array = np.asarray(rgb, dtype=np.uint8).copy()
        if self.train_augmentation:
            # Geometric-only augmentation preserves photo intensity and label semantics.
            if random.random() < 0.5:
                image_array = np.flip(image_array, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if random.random() < 0.5:
                image_array = np.flip(image_array, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().div_(255.0)
        image_tensor = (image_tensor - _MEAN) / _STD
        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask.astype(np.int64, copy=False)),
            "name": name,
            "source_group": str(self.plan.index[name]["source_group"]),
        }


def denormalize_image(image: Tensor) -> Tensor:
    """Convert a normalized CHW tensor back to RGB in [0, 1] for reports."""

    return (image.detach().cpu() * _STD + _MEAN).clamp(0.0, 1.0)
