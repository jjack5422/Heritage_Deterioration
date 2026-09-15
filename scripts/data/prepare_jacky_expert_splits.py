"""Build approved per-expert train/validation manifests from dataset_jacky."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.data.analyze_deterioration_dataset import (
    Tile,
    _canonical_hash,
    _dataset_hash,
    _sha256,
    cross_dataset_leakage_audit,
    scan_primary,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JACKY = ROOT / "dataset_jacky"
DEFAULT_OUTPUT = ROOT / "outputs/deterioration_statistics/jacky_experts"
IGNORE_VALUE = 255
EXPERT_RAW_IDS: dict[str, tuple[int, ...]] = {
    "scratch_crack": (1,),
    "shrinkage_craquelure": (3, 4),
    "loss": (2,),
}
APPROVED_VALIDATION_GROUPS: dict[str, tuple[str, str]] = {
    "scratch_crack": (
        "01_門神部分(必要標註)",
        "MGLST-RH-2R-A5_-2",
    ),
    "shrinkage_craquelure": (
        "KJTHT-SC-M-A4-8",
        "MGLST-DT-1R-A2-1",
    ),
    "loss": (
        "01_門神部分(必要標註)",
        "MGLST-SC-1L-A3-1",
    ),
}
EXPECTED_PARTITION_COUNTS = {
    "scratch_crack": (601, 142),
    "shrinkage_craquelure": (605, 138),
    "loss": (588, 155),
}


def _record(tile: Tile) -> dict[str, str]:
    return {
        "dataset": tile.dataset,
        "source_group": tile.source_group,
        "tile": tile.tile,
        "image": str(tile.image),
        "mask": str(tile.mask),
        "image_sha256": _sha256(tile.image),
        "mask_sha256": _sha256(tile.mask),
    }


def _duplicate_audit(training: Sequence[Tile], validation: Sequence[Tile]) -> dict[str, object]:
    raw = cross_dataset_leakage_audit(training, validation)
    matches = [
        {
            "training_source_group": row["primary_source_group"],
            "training_tile": row["primary_tile"],
            "validation_source_group": row["dataset114_source_group"],
            "validation_tile": row["dataset114_tile"],
            "reasons": row["reasons"],
        }
        for row in raw["suspected_cross_dataset_matches"]
    ]
    return {
        "schema_version": 1,
        "checked_training_tiles": len(training),
        "checked_validation_tiles": len(validation),
        "criteria": raw["criteria"],
        "suspected_matches": matches,
        "passed": not matches,
    }


def _build_manifest(
    tiles: Sequence[Tile],
    *,
    expert: str,
    jacky_root: Path,
) -> dict[str, object]:
    raw_ids = EXPERT_RAW_IDS[expert]
    validation_groups = set(APPROVED_VALIDATION_GROUPS[expert])
    available_groups = {tile.source_group for tile in tiles}
    missing_groups = validation_groups - available_groups
    if missing_groups:
        raise ValueError(f"{expert} validation groups are missing: {sorted(missing_groups)}")

    training_tiles = sorted(
        (tile for tile in tiles if tile.source_group not in validation_groups),
        key=lambda tile: (tile.source_group, tile.tile),
    )
    validation_tiles = sorted(
        (tile for tile in tiles if tile.source_group in validation_groups),
        key=lambda tile: (tile.source_group, tile.tile),
    )
    expected_training, expected_validation = EXPECTED_PARTITION_COUNTS[expert]
    if (len(training_tiles), len(validation_tiles)) != (expected_training, expected_validation):
        raise ValueError(
            f"{expert} partition counts drifted: "
            f"{len(training_tiles)}/{len(validation_tiles)} != {expected_training}/{expected_validation}"
        )

    partition_tiles = {"training": training_tiles, "validation": validation_tiles}
    partition_positive_pixels = {
        name: sum(tile.merged_pixels(raw_ids) for tile in members)
        for name, members in partition_tiles.items()
    }
    partition_positive_tiles = {
        name: sum(tile.merged_pixels(raw_ids) > 0 for tile in members)
        for name, members in partition_tiles.items()
    }
    if any(value == 0 for value in partition_positive_pixels.values()):
        raise ValueError(f"{expert} lacks positive pixels in one partition")

    training = [_record(tile) for tile in training_tiles]
    validation = [_record(tile) for tile in validation_tiles]
    partitions = {"training": training, "validation": validation}
    identities = {
        name: {(row["dataset"], row["tile"]) for row in rows}
        for name, rows in partitions.items()
    }
    groups = {
        name: {(row["dataset"], row["source_group"]) for row in rows}
        for name, rows in partitions.items()
    }
    membership = {
        name: [
            {key: row[key] for key in ("dataset", "source_group", "tile")}
            for row in rows
        ]
        for name, rows in partitions.items()
    }
    adopted_content = [
        {
            key: row[key]
            for key in (
                "dataset",
                "source_group",
                "tile",
                "image_sha256",
                "mask_sha256",
            )
        }
        for name in ("training", "validation")
        for row in partitions[name]
    ]
    duplicate_audit = _duplicate_audit(training_tiles, validation_tiles)
    leakage_passed = not (
        identities["training"] & identities["validation"]
        or groups["training"] & groups["validation"]
        or len(training) + len(validation) != len(tiles)
        or not duplicate_audit["passed"]
    )
    if not leakage_passed:
        raise ValueError(f"{expert} split failed leakage or duplicate audit")

    class_contract = {
        "classes_sha256": _sha256(jacky_root / "classes.txt"),
        "manifest_sha256": _sha256(jacky_root / "manifest.json"),
        "expert": expert,
        "expert_raw_ids": list(raw_ids),
        "ignore_value": IGNORE_VALUE,
        "scratch_supervision": "unavailable; scratch_crack is trained with crack ID 1 only"
        if expert == "scratch_crack"
        else "not applicable",
    }
    return {
        "schema_version": 4,
        "seed": 42,
        "expert": expert,
        "dataset_sha256": _dataset_hash(tiles),
        "class_sha256": _canonical_hash(class_contract),
        "expert_raw_ids": list(raw_ids),
        "dataset_policy": {
            "dataset_jacky": "fourteen_source_groups_training_two_source_groups_validation",
            "dataset": "excluded",
            "dataset114": "excluded",
        },
        "selected_validation": {
            "source_groups": list(APPROVED_VALIDATION_GROUPS[expert]),
            "tile_count": len(validation),
            "positive_tiles": partition_positive_tiles["validation"],
            "positive_tile_ratio": partition_positive_tiles["validation"] / len(validation),
            "positive_pixels": partition_positive_pixels["validation"],
        },
        "partition_positive_pixels": partition_positive_pixels,
        "partition_positive_tiles": partition_positive_tiles,
        "training": training,
        "validation": validation,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash(adopted_content),
        "duplicate_audit": duplicate_audit,
        "leakage_audit": {
            "source_group_overlap": sorted(groups["training"] & groups["validation"]),
            "tile_overlap": sorted(identities["training"] & identities["validation"]),
            "all_jacky_tiles_partitioned_once": len(training) + len(validation) == len(tiles),
            "passed": leakage_passed,
        },
    }


def prepare_splits(jacky_root: str | Path, output_root: str | Path) -> dict[str, dict[str, object]]:
    jacky = Path(jacky_root).resolve()
    output = Path(output_root).resolve()
    tiles = scan_primary(jacky, {0, 1, 2, 3, 4, 5, IGNORE_VALUE})
    if len(tiles) != 743 or len({tile.source_group for tile in tiles}) != 16:
        raise ValueError(
            "dataset_jacky contract requires exactly 743 tiles from 16 source groups; "
            f"found {len(tiles)} tiles from {len({tile.source_group for tile in tiles})} groups"
        )

    output.mkdir(parents=True, exist_ok=True)
    manifests = {
        expert: _build_manifest(tiles, expert=expert, jacky_root=jacky)
        for expert in EXPERT_RAW_IDS
    }
    for expert, manifest in manifests.items():
        (output / f"{expert}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacky", type=Path, default=DEFAULT_JACKY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifests = prepare_splits(args.jacky, args.output)
    print(
        json.dumps(
            {
                expert: {
                    "manifest": str((args.output / f"{expert}.json").resolve()),
                    "training_tiles": len(manifest["training"]),
                    "validation_tiles": len(manifest["validation"]),
                    "validation_groups": manifest["selected_validation"]["source_groups"],
                    "validation_positive_tile_ratio": manifest["selected_validation"]["positive_tile_ratio"],
                }
                for expert, manifest in manifests.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
