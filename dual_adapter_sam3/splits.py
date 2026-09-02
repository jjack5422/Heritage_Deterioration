"""Audit and persist immutable source-group five-fold membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class SplitContractError(ValueError):
    """Raised when folds leak groups or lack either approved target class."""


@dataclass(frozen=True)
class FoldMembership:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    outer_test: tuple[str, ...]


@dataclass(frozen=True)
class SplitContract:
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.payload["sha256"])


def source_group_from_tile(name: str) -> str:
    match = re.match(r"(.+?)_R\d+_C\d+", Path(name).stem)
    if match is None:
        raise SplitContractError(f"cannot derive source_group from tile: {name}")
    return match.group(1)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SplitContractError(f"expected JSON object: {path}")
    return value


def _fold_statistics(root: Path, names: list[str]) -> dict[str, Any]:
    positive_tiles = [0, 0]
    positive_pixels = [0, 0]
    valid_pixels = [0, 0]
    for name in names:
        with Image.open(root / "masks" / name) as image:
            mask = np.asarray(image, dtype=np.uint8)
        for index, label in enumerate((1, 2)):
            count = int((mask == label).sum())
            positive_tiles[index] += int(count > 0)
            positive_pixels[index] += count
            valid_pixels[index] += int(((mask == 0) | (mask == label)).sum())
    return {
        "tile_count": len(names),
        "group_count": len({source_group_from_tile(name) for name in names}),
        "positive_tiles": {"crack_craquelure": positive_tiles[0], "loss": positive_tiles[1]},
        "positive_pixels": {"crack_craquelure": positive_pixels[0], "loss": positive_pixels[1]},
        "valid_pixels": {"crack_craquelure": valid_pixels[0], "loss": valid_pixels[1]},
    }


def audit_existing_splits(dataset_root: str | Path) -> SplitContract:
    root = Path(dataset_root).resolve()
    manifest = _read_object(root / "manifest.json")
    tile_index = _read_object(root / "tile_index.json")
    raw_items = tile_index.get("items")
    if not isinstance(raw_items, list):
        raise SplitContractError("tile_index items are missing")
    eligible = {str(item["tile"]) for item in raw_items}
    if len(eligible) != int(manifest.get("pair_count", -1)):
        raise SplitContractError("tile_index count does not match manifest")
    folds: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_groups: set[str] = set()
    for fold_index in range(5):
        source = _read_object(root / "splits" / f"fold{fold_index}.json")
        if source.get("outer_fold") != fold_index:
            raise SplitContractError(f"fold{fold_index} has inconsistent outer_fold")
        names = [str(name) for name in source.get("holdout_tiles", [])]
        if not names or len(names) != len(set(names)):
            raise SplitContractError(f"fold{fold_index} is empty or contains duplicate tiles")
        if not set(names).issubset(eligible) or seen.intersection(names):
            raise SplitContractError(f"fold{fold_index} has unknown or cross-fold tiles")
        groups = {source_group_from_tile(name) for name in names}
        if seen_groups.intersection(groups):
            raise SplitContractError(f"fold{fold_index} leaks source_group across folds")
        stats = _fold_statistics(root, names)
        if min(stats["positive_tiles"].values()) <= 0:
            raise SplitContractError(f"fold{fold_index} lacks one approved positive class")
        folds.append({"fold": fold_index, "tiles": sorted(names), "source_groups": sorted(groups), "statistics": stats})
        seen.update(names)
        seen_groups.update(groups)
    if seen != eligible:
        raise SplitContractError(f"folds do not partition all tiles: missing={len(eligible - seen)}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "strategy": "existing_immutable_source_group_folds",
        "dataset_root": str(root),
        "dataset_manifest_sha256": manifest.get("manifest_sha256"),
        "image_manifest_sha256": manifest.get("image_manifest_sha256"),
        "mask_manifest_sha256": manifest.get("mask_manifest_sha256"),
        "folds": folds,
        "nested_rule": "outer_test=k; validation=(k+1)%5; remaining folds=train",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SplitContract(payload)


def load_split_contract(path: str | Path, *, dataset_root: str | Path | None = None) -> SplitContract:
    payload = _read_object(Path(path))
    claimed = payload.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["sha256"] = claimed
    if claimed != actual:
        raise SplitContractError("split contract sha256 mismatch")
    if dataset_root is not None:
        manifest = _read_object(Path(dataset_root) / "manifest.json")
        for field, manifest_field in (
            ("dataset_manifest_sha256", "manifest_sha256"),
            ("image_manifest_sha256", "image_manifest_sha256"),
            ("mask_manifest_sha256", "mask_manifest_sha256"),
        ):
            if payload.get(field) != manifest.get(manifest_field):
                raise SplitContractError(f"dataset changed: {field} mismatch")
    return SplitContract(payload)


def fold_membership(contract: SplitContract, outer_fold: int) -> FoldMembership:
    if outer_fold not in range(5):
        raise SplitContractError("outer_fold must be 0..4")
    by_fold = {int(item["fold"]): tuple(item["tiles"]) for item in contract.payload["folds"]}
    validation_fold = (outer_fold + 1) % 5
    train = tuple(name for fold in range(5) if fold not in (outer_fold, validation_fold) for name in by_fold[fold])
    return FoldMembership(train, by_fold[validation_fold], by_fold[outer_fold])


def source_group_index(contract: SplitContract) -> dict[str, str]:
    return {
        name: group
        for fold in contract.payload["folds"]
        for group in fold["source_groups"]
        for name in fold["tiles"]
        if source_group_from_tile(name) == group
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contract = audit_existing_splits(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "sha256": contract.sha256,
        "folds": [item["statistics"] for item in contract.payload["folds"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
