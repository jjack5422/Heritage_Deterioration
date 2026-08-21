"""Validate dataset contracts and build leak-free expert splits."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class DataError(ValueError):
    """Dataset files and metadata disagree."""


class ExpertDataset(torch.utils.data.Dataset):
    """Map a multiclass source mask to one expert's binary target."""

    def __init__(self, source: torch.utils.data.Dataset, expert_id: int, ignore_value: int) -> None:
        self.source = source
        self.expert_id = expert_id
        self.ignore_value = ignore_value

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.source[index]
        source_mask = item["mask"]
        binary_mask = (source_mask == self.expert_id).long()
        binary_mask[source_mask == self.ignore_value] = self.ignore_value
        return {**item, "mask": binary_mask}


class MergedForegroundDataset(torch.utils.data.Dataset):
    """Keep merged crack as foreground and exclude every unrelated label."""

    def __init__(
        self,
        source: torch.utils.data.Dataset,
        foreground_id: int,
        ignore_value: int,
    ) -> None:
        self.source = source
        self.foreground_id = foreground_id
        self.ignore_value = ignore_value

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.source[index]
        source_mask = item["mask"]
        target = torch.full_like(source_mask, self.ignore_value)
        target[source_mask == 0] = 0
        target[source_mask == self.foreground_id] = 1
        return {**item, "mask": target.long()}


class JointDataset(torch.utils.data.Dataset):
    """Remap source labels to background/crack/craquelure and ignore all others."""

    def __init__(
        self,
        source: torch.utils.data.Dataset,
        crack_id: int,
        craquelure_id: int,
        ignore_value: int,
    ) -> None:
        self.source = source
        self.crack_id = crack_id
        self.craquelure_id = craquelure_id
        self.ignore_value = ignore_value

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.source[index]
        source_mask = item["mask"]
        target = torch.full_like(source_mask, self.ignore_value)
        target[source_mask == 0] = 0
        target[source_mask == self.crack_id] = 1
        target[source_mask == self.craquelure_id] = 2
        return {**item, "mask": target.long()}


@dataclass
class DataPlan:
    """Validated files and nested split for one segmentation task."""

    root: Path
    source_class_names: tuple[str, ...]
    class_names: tuple[str, ...]
    expert_name: str
    expert_id: int
    source_class_ids: tuple[int, ...]
    ignore_value: int
    manifest_hash: str
    outer_fold: int
    inner_fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    train_counts: np.ndarray
    val_counts: np.ndarray
    test_counts: np.ndarray
    index: dict[str, dict[str, Any]]
    tile_counts: dict[str, np.ndarray]

    def items(self, names: Sequence[str]) -> list[dict[str, Any]]:
        return [self.index[name] for name in names]

    def record(self) -> dict[str, Any]:
        """Return the reproducible split metadata written beside a run."""
        record = {
            "source_class_names": self.source_class_names,
            "class_names": self.class_names,
            "task": {
                "name": self.expert_name,
                "source_class_ids": self.source_class_ids,
            },
            "ignore_value": self.ignore_value,
            "manifest_hash": self.manifest_hash,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "train_counts": self.train_counts.tolist(),
            "val_counts": self.val_counts.tolist(),
            "test_counts": self.test_counts.tolist(),
        }
        if len(self.source_class_ids) == 1:
            record["expert"] = {
                "name": self.expert_name,
                "source_class_id": self.expert_id,
            }
        return record


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataError(f"找不到必要檔案: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"JSON 格式錯誤: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DataError(f"JSON 根節點必須是 object: {path}")
    return data


def panel_name(tile: str) -> str:
    match = re.match(r"(.+?)_R\d+_C\d+", Path(tile).stem)
    if not match:
        raise DataError(f"無法由檔名解析 panel: {tile}")
    return match.group(1)


def load_contract(root: Path) -> tuple[tuple[str, ...], int, int, str, dict]:
    manifest = read_json(root / "manifest.json")
    if manifest.get("training_eligible") is not True:
        raise DataError("manifest.json 未標示 training_eligible=true")
    try:
        class_ids = manifest["label_contract"]["class_ids"]
        ordered = sorted((int(index), str(name)) for name, index in class_ids.items())
        ignore = int(manifest["label_contract"].get("ignore_value", 255))
        pair_count = int(manifest["pair_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError("manifest.json 的 label_contract 無效") from exc
    if [index for index, _ in ordered] != list(range(len(ordered))):
        raise DataError("class IDs 必須由 0 連續排列")
    names = tuple(name for _, name in ordered)
    if not names or names[0] != "background":
        raise DataError("class ID 0 必須是 background")
    return names, ignore, pair_count, str(manifest.get("manifest_sha256", "")), manifest


def load_index(root: Path, pair_count: int) -> dict[str, dict[str, Any]]:
    items = read_json(root / "tile_index.json").get("items")
    if not isinstance(items, list):
        raise DataError("tile_index.json 缺少 items array")
    index: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("tile"), str):
            raise DataError("tile_index item 缺少 tile 字串")
        if item["tile"] in index:
            raise DataError(f"tile_index 有重複檔名: {item['tile']}")
        index[item["tile"]] = item
    if len(index) != pair_count:
        raise DataError(f"tile_index={len(index)}，manifest pair_count={pair_count}")
    return index


def inspect_tiles(
    root: Path,
    names: Sequence[str],
    num_classes: int,
    ignore: int,
) -> dict[str, np.ndarray]:
    image_dir, mask_dir = root / "images", root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise DataError("dataset 必須包含 images/ 與單通道 label-ID masks/")
    allowed = set(range(num_classes)) | {ignore}
    counts: dict[str, np.ndarray] = {}
    for name in names:
        image_path, mask_path = image_dir / name, mask_dir / name
        if not image_path.is_file() or not mask_path.is_file():
            raise DataError(f"缺少 image/mask pair: {name}")
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask:
            array = np.asarray(mask)
            mask_size = mask.size
        if array.ndim != 2:
            raise DataError(f"mask 必須是單通道 label IDs: {mask_path}")
        if image_size != mask_size:
            raise DataError(f"image/mask 尺寸不一致: {name}")
        unknown = set(np.unique(array).tolist()) - allowed
        if unknown:
            raise DataError(f"mask {name} 含未知 label IDs: {sorted(unknown)}")
        valid = array != ignore
        counts[name] = np.bincount(
            array[valid].astype(np.int64),
            minlength=num_classes,
        )[:num_classes]
    return counts


def load_fold(
    path: Path,
    fold: int,
    eligible: set[str],
    class_names: tuple[str, ...],
    ignore: int,
    manifest: dict,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    data = read_json(path)
    if data.get("outer_fold") != fold:
        raise DataError(f"{path.name} 的 outer_fold 不正確")
    try:
        train = tuple(data["folds"][0]["train"])
        test = tuple(data["holdout_tiles"])
        contract = data["data_contract"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DataError(f"split 格式錯誤: {path}") from exc
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise DataError(f"split 含重複檔名: {path}")
    if set(train) & set(test):
        raise DataError(f"split train/holdout 重疊: {path}")
    if set(train) | set(test) != eligible:
        missing = (set(train) | set(test)) - eligible
        raise DataError(f"split 與 tile_index 不一致: {sorted(missing)[:5]}")
    expected = (
        tuple(contract.get("class_names", ())),
        int(contract.get("ignore_value", 255)),
        contract.get("task_image_manifest_sha256"),
        contract.get("task_mask_manifest_sha256"),
    )
    actual = (
        class_names,
        ignore,
        manifest.get("image_manifest_sha256"),
        manifest.get("mask_manifest_sha256"),
    )
    if expected != actual:
        raise DataError(f"split data contract 與 manifest 不一致: {path}")
    return train, test


def sum_counts(counts: dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.sum([counts[name] for name in names], axis=0, dtype=np.int64)


def binary_counts(source_counts: np.ndarray, expert_id: int) -> np.ndarray:
    foreground = int(source_counts[expert_id])
    return np.array([int(source_counts.sum()) - foreground, foreground], dtype=np.int64)


def merged_foreground_counts(source_counts: np.ndarray, foreground_id: int) -> np.ndarray:
    """Count only background and merged crack; unrelated defects are excluded."""

    return np.array([source_counts[0], source_counts[foreground_id]], dtype=np.int64)


def joint_counts(source_counts: np.ndarray, crack_id: int, craquelure_id: int) -> np.ndarray:
    """Count only supervised joint-task pixels; other deterioration is ignored."""

    return np.array(
        [source_counts[0], source_counts[crack_id], source_counts[craquelure_id]],
        dtype=np.int64,
    )


def joint_class_weights(counts: np.ndarray) -> torch.Tensor:
    """Mild inverse-sqrt weights with equal-mean foreground and bounded extremes."""

    frequencies = counts / max(int(counts.sum()), 1)
    weights = 1 / np.sqrt(np.clip(frequencies, 1e-12, None))
    weights /= max(float(weights[1:].mean()), 1e-12)
    return torch.tensor(np.clip(weights, 0.25, 3.0), dtype=torch.float32)


def class_weights(counts: np.ndarray) -> torch.Tensor:
    frequencies = counts / max(int(counts.sum()), 1)
    present = frequencies > 0
    weights = np.zeros_like(frequencies, dtype=np.float64)
    weights[present] = np.median(frequencies[present]) / frequencies[present]
    return torch.tensor(weights, dtype=torch.float32)


def cost_weight_for_epoch(epoch: int, maximum: float) -> float:
    """Keep cost off for five epochs, then linearly ramp through epoch 15."""

    return maximum * min(1.0, max(0.0, (epoch - 5) / 10))


def joint_sampling_weights(
    plan: DataPlan,
    names: Sequence[str],
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> torch.Tensor:
    """Give crack/craquelure/other tile groups the requested epoch exposure."""

    crack_id, craquelure_id = plan.source_class_ids
    groups: list[int] = []
    for name in names:
        counts = plan.tile_counts[name]
        if counts[crack_id] > 0:
            groups.append(0)
        elif counts[craquelure_id] > 0:
            groups.append(1)
        else:
            groups.append(2)
    sizes = np.bincount(groups, minlength=3)
    weights = [fractions[group] / max(int(sizes[group]), 1) for group in groups]
    return torch.tensor(weights, dtype=torch.double)


def prepare_dataset(
    root: str | Path,
    expert: str,
    outer_fold: int,
    inner_fold: int | None = None,
) -> DataPlan:
    """Build a leak-free split with the next fold used for automatic validation."""
    root = Path(root).resolve()
    source_names, ignore, pair_count, manifest_hash, manifest = load_contract(root)
    joint = expert == "crack_craquelure"
    merged_foreground = expert == "foreground"
    expected_merged_names = (
        "background",
        "craquelure",
        "loss",
        "shrinkage",
        "flaking",
        "stain",
    )
    if merged_foreground and source_names != expected_merged_names:
        raise DataError(
            "foreground task requires the merged dataset label contract "
            f"{expected_merged_names!r}"
        )
    if joint and not {"crack", "craquelure"}.issubset(source_names):
        raise DataError("joint task 需要 crack 與 craquelure label IDs")
    if not joint and not merged_foreground and (
        expert == "background" or expert not in source_names
    ):
        available = ", ".join(source_names[1:])
        raise DataError(f"未知 expert {expert!r}；可用劣化類別: {available}")
    source_class_ids = (
        (source_names.index("crack"), source_names.index("craquelure"))
        if joint
        else ((source_names.index("craquelure"),) if merged_foreground else (source_names.index(expert),))
    )
    expert_id = -1 if joint else source_class_ids[0]
    index = load_index(root, pair_count)
    eligible = set(index)
    tile_counts = inspect_tiles(root, tuple(index), len(source_names), ignore)
    outer_train, test = load_fold(
        root / "splits" / f"fold{outer_fold}.json",
        outer_fold,
        eligible,
        source_names,
        ignore,
        manifest,
    )

    candidates: list[
        tuple[tuple[int, int, int], int, tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]
    ] = []
    available_folds = sorted(
        int(path.stem.removeprefix("fold"))
        for path in (root / "splits").glob("fold*.json")
        if path.stem.removeprefix("fold").isdigit()
    )
    automatic_inner_fold = inner_fold is None
    if automatic_inner_fold:
        if len(available_folds) < 2 or outer_fold not in available_folds:
            raise DataError("自動 validation 輪替至少需要兩個包含 outer fold 的 split")
        outer_position = available_folds.index(outer_fold)
        folds = [available_folds[(outer_position + 1) % len(available_folds)]]
    else:
        folds = [inner_fold]
    for candidate in folds:
        if candidate == outer_fold:
            continue
        _, candidate_holdout = load_fold(
            root / "splits" / f"fold{candidate}.json",
            candidate,
            eligible,
            source_names,
            ignore,
            manifest,
        )
        val_set = set(candidate_holdout)
        if not val_set.issubset(set(outer_train)):
            continue
        train = tuple(name for name in outer_train if name not in val_set)
        val = tuple(name for name in outer_train if name in val_set)
        if not train or not val:
            continue
        train_panels = {panel_name(name) for name in train}
        val_panels = {panel_name(name) for name in val}
        test_panels = {panel_name(name) for name in test}
        if train_panels & val_panels or train_panels & test_panels or val_panels & test_panels:
            raise DataError("panel 在 train/val/test 之間重疊")
        if joint:
            train_counts = joint_counts(sum_counts(tile_counts, train), *source_class_ids)
            val_counts = joint_counts(sum_counts(tile_counts, val), *source_class_ids)
        elif merged_foreground:
            train_counts = merged_foreground_counts(sum_counts(tile_counts, train), expert_id)
            val_counts = merged_foreground_counts(sum_counts(tile_counts, val), expert_id)
        else:
            train_counts = binary_counts(sum_counts(tile_counts, train), expert_id)
            val_counts = binary_counts(sum_counts(tile_counts, val), expert_id)
        if np.any(train_counts[1:] == 0) or np.any(val_counts[1:] == 0):
            continue
        score = (
            min(int(train_counts[1:].min()), int(val_counts[1:].min())),
            int(val_counts[1:].min()),
            -candidate,
        )
        candidates.append((score, candidate, train, val, train_counts, val_counts))
    if not candidates:
        if automatic_inner_fold:
            raise DataError(
                f"自動輪替的 validation fold {folds[0]} 無法建立有效 split；"
                "請修正該 fold 的正樣本，或用 --inner-fold 明確指定"
            )
        raise DataError(f"expert {expert} 無法建立 train/validation 都含正樣本的 split")
    _, chosen, train, val, train_counts, val_counts = max(candidates, key=lambda item: item[0])
    return DataPlan(
        root=root,
        source_class_names=source_names,
        class_names=("background", "crack", "craquelure") if joint else ("background", expert),
        expert_name=expert,
        expert_id=expert_id,
        source_class_ids=source_class_ids,
        ignore_value=ignore,
        manifest_hash=manifest_hash,
        outer_fold=outer_fold,
        inner_fold=chosen,
        train=train,
        val=val,
        test=test,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=(
            joint_counts(sum_counts(tile_counts, test), *source_class_ids)
            if joint
            else (
                merged_foreground_counts(sum_counts(tile_counts, test), expert_id)
                if merged_foreground
                else binary_counts(sum_counts(tile_counts, test), expert_id)
            )
        ),
        index=index,
        tile_counts=tile_counts,
    )
