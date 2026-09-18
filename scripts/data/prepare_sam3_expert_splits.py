"""Build audited SAM3 expert manifests from Jacky and Dataset115."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.data.analyze_deterioration_dataset import Tile, _canonical_hash, _dataset_hash, _sha256, scan_primary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JACKY = ROOT / "dataset_jacky"
DEFAULT_DATASET115 = ROOT / "dataset115_filtered"
DEFAULT_OUTPUT = ROOT / "outputs" / "deterioration_statistics" / "sam3_experts"
IGNORE_VALUE = 255
EXPERT_RAW_IDS: dict[str, tuple[int, ...]] = {
    "scratch_crack": (1,),
    "shrinkage_craquelure": (3, 4),
    "loss": (2,),
}
VALIDATION_GROUPS: dict[str, tuple[str, ...]] = {
    "shrinkage_craquelure": ("KJWTomh-MH-M-A3E-1", "WFT-PH-M-1LB1-1-1", "KJWTomh-PH-M-1RB1-1"),
    "scratch_crack": ("KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "MST-SC-M-A2-2-7"),
    "loss": ("KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "KJWTomh-PH-M-1RB1-1"),
}
TEST_GROUPS: dict[str, tuple[str, ...]] = {
    "shrinkage_craquelure": ("KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1", "KJWTomh-MH-M-A3E-3-2"),
    "scratch_crack": ("KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A6'-2"),
    "loss": ("MST-SC-M-A2-2-7", "KJTHT-SC-R-A4-6", "KJWTomh-MH-M-A3E-3-2", "MST-SC-M-A2-2-6"),
}
EXCLUDED_TRAINING_GROUPS: dict[str, tuple[str, ...]] = {
    "shrinkage_craquelure": (),
    "scratch_crack": (),
    "loss": (),
}
EXPECTED_EXCLUDED_COUNTS: dict[str, tuple[int, int]] = {
    "shrinkage_craquelure": (0, 0),
    "scratch_crack": (0, 0),
    "loss": (0, 0),
}
EXPECTED_DATASET115_COUNTS: dict[str, tuple[int, int, int]] = {
    "shrinkage_craquelure": (394, 123, 198),
    "scratch_crack": (498, 105, 112),
    "loss": (500, 108, 107),
}
EXPECTED_POSITIVE_TILES: dict[str, tuple[int, int, int]] = {
    "shrinkage_craquelure": (108, 56, 176),
    "scratch_crack": (236, 45, 99),
    "loss": (100, 23, 20),
}
PARTITIONS = ("training", "validation", "test")


def _jacky_record(tile: Tile, raw_ids: Sequence[int]) -> dict[str, str | int]:
    return {
        "dataset": "dataset_jacky",
        "source_group": tile.source_group,
        "tile": tile.tile,
        "image": str(tile.image),
        "mask": str(tile.mask),
        "mask_encoding": "class_index_uint8",
        "image_sha256": _sha256(tile.image),
        "mask_sha256": _sha256(tile.mask),
        "positive_pixels": tile.merged_pixels(raw_ids),
    }


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes dataset root: {relative}")
    return path


def _dataset115_records(root: Path, expert: str) -> list[dict[str, str | int]]:
    manifest_path = root / "expert_views" / expert / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 715:
        raise ValueError(f"{expert} Dataset115 manifest must contain 715 rows; found {len(rows)}")

    records: list[dict[str, str | int]] = []
    for row in rows:
        if row["expert"] != expert:
            raise ValueError(f"Dataset115 manifest expert mismatch: {row['expert']} != {expert}")
        image = _resolve_under(root, row["image"])
        mask = _resolve_under(root, row["mask"])
        if _sha256(image) != row["image_sha256"] or _sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"Dataset115 content checksum mismatch: {row['tile']}")
        foreground_pixels = int(row["foreground_pixels"])
        if int(row["is_positive"]) != int(foreground_pixels > 0):
            raise ValueError(f"Dataset115 positivity metadata mismatch: {row['tile']}")
        records.append(
            {
                "dataset": "dataset115_filtered",
                "source_group": row["source_group"],
                "tile": row["tile"],
                "image": str(image),
                "mask": str(mask),
                "mask_encoding": "binary_uint8_0_255",
                "image_sha256": row["image_sha256"],
                "mask_sha256": row["mask_sha256"],
                "positive_pixels": foreground_pixels,
            }
        )
    identities = {(str(row["dataset"]), str(row["tile"])) for row in records}
    if len(identities) != len(records):
        raise ValueError(f"{expert} Dataset115 manifest contains duplicate tile identities")
    return sorted(records, key=lambda row: (str(row["source_group"]), str(row["tile"])))


def _partition_dataset115(
    records: Sequence[Mapping[str, str | int]],
    expert: str,
) -> tuple[dict[str, list[Mapping[str, str | int]]], list[Mapping[str, str | int]]]:
    validation_groups = set(VALIDATION_GROUPS[expert])
    test_groups = set(TEST_GROUPS[expert])
    excluded_groups = set(EXCLUDED_TRAINING_GROUPS[expert])
    if validation_groups & test_groups or excluded_groups & (validation_groups | test_groups):
        raise ValueError(f"{expert} has an overlapping validation, test, or excluded source group")
    available = {str(row["source_group"]) for row in records}
    missing = (validation_groups | test_groups | excluded_groups) - available
    if missing:
        raise ValueError(f"{expert} required source groups are missing: {sorted(missing)}")

    partitions: dict[str, list[Mapping[str, str | int]]] = {name: [] for name in PARTITIONS}
    excluded: list[Mapping[str, str | int]] = []
    for row in records:
        group = str(row["source_group"])
        if group in excluded_groups:
            excluded.append(row)
            continue
        partition = "validation" if group in validation_groups else "test" if group in test_groups else "training"
        partitions[partition].append(row)
    actual_excluded = (len(excluded), sum(int(row["positive_pixels"]) > 0 for row in excluded))
    if actual_excluded != EXPECTED_EXCLUDED_COUNTS[expert]:
        raise ValueError(
            f"{expert} Dataset115 excluded counts drifted: "
            f"{actual_excluded} != {EXPECTED_EXCLUDED_COUNTS[expert]}"
        )
    actual = tuple(len(partitions[name]) for name in PARTITIONS)
    if actual != EXPECTED_DATASET115_COUNTS[expert]:
        raise ValueError(f"{expert} Dataset115 partition counts drifted: {actual} != {EXPECTED_DATASET115_COUNTS[expert]}")
    positive_tiles = tuple(sum(int(row["positive_pixels"]) > 0 for row in partitions[name]) for name in PARTITIONS)
    if positive_tiles != EXPECTED_POSITIVE_TILES[expert]:
        raise ValueError(f"{expert} Dataset115 positive-tile counts drifted: {positive_tiles} != {EXPECTED_POSITIVE_TILES[expert]}")
    return partitions, excluded


def _partition_stats(rows: Sequence[Mapping[str, str | int]]) -> dict[str, Any]:
    datasets = Counter(str(row["dataset"]) for row in rows)
    return {
        "tiles": len(rows),
        "positive_tiles": sum(int(row["positive_pixels"]) > 0 for row in rows),
        "positive_pixels": sum(int(row["positive_pixels"]) for row in rows),
        "datasets": dict(sorted(datasets.items())),
        "source_groups": sorted({f"{row['dataset']}:{row['source_group']}" for row in rows}),
    }


def _audit_partitions(partitions: Mapping[str, Sequence[Mapping[str, str | int]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    identities = {
        name: {(str(row["dataset"]), str(row["tile"])) for row in rows}
        for name, rows in partitions.items()
    }
    groups = {
        name: {(str(row["dataset"]), str(row["source_group"])) for row in rows}
        for name, rows in partitions.items()
    }
    image_hash_partitions: dict[str, set[str]] = {}
    for name, rows in partitions.items():
        for row in rows:
            image_hash_partitions.setdefault(str(row["image_sha256"]), set()).add(name)
    duplicate_crossings = {
        digest: sorted(names)
        for digest, names in image_hash_partitions.items()
        if len(names) > 1
    }
    pair_overlaps: dict[str, dict[str, list[Any]]] = {}
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            pair_overlaps[f"{left}_vs_{right}"] = {
                "tile_identities": sorted(identities[left] & identities[right]),
                "source_groups": sorted(groups[left] & groups[right]),
            }
    leakage_passed = not duplicate_crossings and all(
        not values["tile_identities"] and not values["source_groups"]
        for values in pair_overlaps.values()
    )
    duplicate_audit = {
        "criterion": "identical image SHA-256 must not cross partitions",
        "cross_partition_matches": duplicate_crossings,
        "passed": not duplicate_crossings,
    }
    leakage_audit = {"partition_overlaps": pair_overlaps, "passed": leakage_passed}
    if not leakage_passed:
        raise ValueError("partition leakage or cross-partition duplicate image detected")
    return duplicate_audit, leakage_audit


def _build_manifest(
    jacky_tiles: Sequence[Tile],
    dataset115_root: Path,
    *,
    expert: str,
    jacky_root: Path,
) -> dict[str, Any]:
    raw_ids = EXPERT_RAW_IDS[expert]
    jacky_records = [_jacky_record(tile, raw_ids) for tile in sorted(jacky_tiles, key=lambda item: (item.source_group, item.tile))]
    dataset115_records = _dataset115_records(dataset115_root, expert)
    dataset115_partitions, excluded_training = _partition_dataset115(dataset115_records, expert)
    partitions: dict[str, list[Mapping[str, str | int]]] = {
        "training": [*jacky_records, *dataset115_partitions["training"]],
        "validation": dataset115_partitions["validation"],
        "test": dataset115_partitions["test"],
    }
    if len(jacky_records) != 743 or any(row["dataset"] != "dataset_jacky" for row in jacky_records):
        raise ValueError("all 743 Jacky tiles must be available for training")
    if any(row["dataset"] == "dataset_jacky" for name in ("validation", "test") for row in partitions[name]):
        raise ValueError("dataset_jacky may only appear in training")

    duplicate_audit, leakage_audit = _audit_partitions(partitions)
    partition_stats = {name: _partition_stats(rows) for name, rows in partitions.items()}
    if any(partition_stats[name]["positive_pixels"] <= 0 for name in PARTITIONS):
        raise ValueError(f"{expert} has an empty-positive partition")

    membership = {
        name: [
            {key: row[key] for key in ("dataset", "source_group", "tile")}
            for row in rows
        ]
        for name, rows in partitions.items()
    }
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in PARTITIONS
        for row in partitions[name]
    ]
    excluded_content = [
        {
            key: row[key]
            for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256", "positive_pixels")
        }
        for row in excluded_training
    ]
    source_hashes = {
        "dataset_jacky_manifest_sha256": _sha256(jacky_root / "manifest.json"),
        "dataset_jacky_classes_sha256": _sha256(jacky_root / "classes.txt"),
        "dataset115_manifest_sha256": _sha256(dataset115_root / "metadata" / "manifest.csv"),
        "dataset115_expert_views_sha256": _sha256(dataset115_root / "metadata" / "expert_views.json"),
        "dataset115_expert_manifest_sha256": _sha256(dataset115_root / "expert_views" / expert / "manifest.csv"),
    }
    return {
        "schema_version": 6,
        "seed": 42,
        "expert": expert,
        "expert_raw_ids": list(raw_ids),
        "target_contract": {
            "dataset_jacky": "class-index raw IDs become foreground; 255 remains ignore",
            "dataset115_filtered": "expert-specific binary 0/255 mask; 255 becomes foreground",
        },
        "dataset_policy": {
            "dataset_jacky": "training_only_all_743_tiles",
            "dataset115_filtered": "source_group_locked_train_validation_test",
            "checkpoint_selection": "validation_only",
            "test": "evaluate_once_after_checkpoint_selection",
        },
        "exclusion_policy": "declared source groups are omitted before training partition assembly",
        "locked_groups": {
            "validation": list(VALIDATION_GROUPS[expert]),
            "test": list(TEST_GROUPS[expert]),
            "excluded_training": list(EXCLUDED_TRAINING_GROUPS[expert]),
        },
        "source_hashes": source_hashes,
        "dataset_sha256": _canonical_hash(source_hashes),
        "class_sha256": _canonical_hash({"expert": expert, "raw_ids": raw_ids, "source_hashes": source_hashes}),
        "partition_stats": partition_stats,
        "excluded_training_stats": _partition_stats(excluded_training),
        "training": partitions["training"],
        "validation": partitions["validation"],
        "test": partitions["test"],
        "excluded_training": excluded_content,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash(adopted_content),
        "exclusion_sha256": _canonical_hash(excluded_content),
        "duplicate_audit": duplicate_audit,
        "leakage_audit": leakage_audit,
    }


def _replace_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    had_destination = destination.exists()
    if had_destination:
        destination.replace(backup)
    try:
        staging.replace(destination)
    except BaseException:
        if had_destination and backup.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def prepare_splits(
    jacky_root: str | Path,
    dataset115_root: str | Path,
    output_root: str | Path,
) -> dict[str, dict[str, Any]]:
    jacky = Path(jacky_root).resolve()
    dataset115 = Path(dataset115_root).resolve()
    output = Path(output_root).resolve()
    jacky_tiles = scan_primary(jacky, {0, 1, 2, 3, 4, 5, IGNORE_VALUE})
    if len(jacky_tiles) != 743 or len({tile.source_group for tile in jacky_tiles}) != 16:
        raise ValueError(
            "dataset_jacky contract requires exactly 743 tiles from 16 source groups; "
            f"found {len(jacky_tiles)} tiles from {len({tile.source_group for tile in jacky_tiles})} groups"
        )
    manifests = {
        expert: _build_manifest(jacky_tiles, dataset115, expert=expert, jacky_root=jacky)
        for expert in EXPERT_RAW_IDS
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        for expert, manifest in manifests.items():
            (staging / f"{expert}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        _replace_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacky", type=Path, default=DEFAULT_JACKY)
    parser.add_argument("--dataset115", type=Path, default=DEFAULT_DATASET115)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifests = prepare_splits(args.jacky, args.dataset115, args.output)
    print(
        json.dumps(
            {
                expert: {
                    "manifest": str((args.output / f"{expert}.json").resolve()),
                    "partition_stats": manifest["partition_stats"],
                    "split_sha256": manifest["split_sha256"],
                }
                for expert, manifest in manifests.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
