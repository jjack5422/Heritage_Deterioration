"""Build portable trainer-native manifests for the expanded Dataset115 + Jacky split."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25"
TRANSFER_ROOT_RELATIVE = Path("outputs/remote_training_transfer_2026-09-25")
DATA_ROOT_RELATIVE = TRANSFER_ROOT_RELATIVE / "data"
OUTPUT_DIR = PROJECT_ROOT / TRANSFER_ROOT_RELATIVE / "manifests"
EXPERT_RAW_IDS = {
    "scratch_crack": [1],
    "loss": [2],
    "shrinkage_craquelure": [3, 4],
}
PARTITIONS = ("training", "validation", "test")
DATASET_POLICY = {
    "dataset_jacky": "training_only_all_743_tiles",
    "dataset115_filtered": "source_group_locked_train_validation_test",
    "checkpoint_selection": "validation_only",
    "test": "evaluate_once_after_checkpoint_selection",
}
SPLIT_POLICY = {
    "strategy": "current_inference_f1_ranked_source_group",
    "threshold": 0.5,
    "source_group_constraint": "disjoint_across_training_validation_test",
    "jacky_policy": "all_tiles_training_only",
    "test_selection": "highest_clean_candidate_pooled_pixel_f1",
    "validation_selection": "next_clean_candidate_groups_by_pooled_pixel_f1",
}
SPLIT_ID = "2026-09-25_expanded_dataset115_jacky_f1_ranked_source_group_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _transfer_path(dataset: str, source_path: str) -> str:
    path = Path(source_path)
    if path.is_absolute():
        raise ValueError(f"expected a project-relative {dataset} path, got {source_path!r}")
    transfer_root = DATA_ROOT_RELATIVE / dataset
    if path.parts[: len(transfer_root.parts)] == transfer_root.parts:
        return path.as_posix()
    if not path.parts or path.parts[0] != dataset:
        raise ValueError(f"expected a project-relative {dataset} path, got {source_path!r}")
    return (transfer_root / Path(*path.parts[1:])).as_posix()

def _row_for_trainer(row: dict[str, str]) -> dict[str, str]:
    normalized = {
        key: row[key]
        for key in ("dataset", "source_group", "tile", "image", "mask", "mask_encoding", "image_sha256", "mask_sha256")
    }
    for field in ("image", "mask"):
        normalized[field] = _transfer_path(row["dataset"], row[field])
    return normalized


def _portable_csv_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    for field in ("image", "mask"):
        normalized[field] = _transfer_path(row["dataset"], row[field])
    return normalized


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if row["dataset"] == "dataset_jacky" else 1,
            row["source_group"],
            row["tile"],
        ),
    )
def _source_hashes(expert: str) -> dict[str, str]:
    paths = (
        OUTPUT_DIR / f"merged_training_manifest_{expert}.csv",
        OUTPUT_DIR / "source/independent_expert_split_proposal.csv",
        OUTPUT_DIR / "source/independent_expert_split_summary.csv",
        PROJECT_ROOT / DATA_ROOT_RELATIVE / "dataset115_filtered/metadata/manifest.csv",
        PROJECT_ROOT / DATA_ROOT_RELATIVE / "dataset115_filtered/metadata/expert_views.json",
        PROJECT_ROOT / DATA_ROOT_RELATIVE / "dataset_jacky/manifest.json",
        PROJECT_ROOT / DATA_ROOT_RELATIVE / "dataset_jacky/classes.txt",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths}


def _ranked_groups(expert: str, proposal: list[dict[str, str]]) -> list[dict[str, Any]]:
    records = [row for row in proposal if row["expert"] == expert]
    records.sort(key=lambda row: int(row["rank_all_97_by_current_f1"]))
    ranked: list[dict[str, Any]] = []
    for row in records:
        current_f1 = row["current_inference_f1"].strip()
        clean_rank = row["rank_in_clean_test_candidate_pool"].strip()
        ranked.append(
            {
                "source_group": row["source_group"],
                "partition": {"train": "training"}.get(row["proposed_partition"], row["proposed_partition"]),
                "tile_count": int(row["tile_count"]),
                "current_inference_pooled_f1": float(current_f1) if current_f1 else None,
                "previous_run_role": row["previous_run_role"],
                "rank_all_97": int(row["rank_all_97_by_current_f1"]),
                "rank_in_clean_test_candidate_pool": int(clean_rank) if clean_rank else None,
                "jacky_same_source_group_collision": row["jacky_same_source_group_collision"].lower() == "true",
            }
        )
    return ranked


def _build_manifest(expert: str, rows: list[dict[str, str]], ranked_groups: list[dict[str, Any]]) -> dict[str, Any]:
    partitions: dict[str, list[dict[str, str]]] = {name: [] for name in PARTITIONS}
    for row in rows:
        partition = row["partition"]
        if partition not in partitions:
            raise ValueError(f"unexpected partition {partition!r} in {expert} union manifest")
        partitions[partition].append(_row_for_trainer(row))
    for name in PARTITIONS:
        partitions[name] = _sort_rows(partitions[name])

    actual_inventory = {
        dataset: {
            "tiles": sum(row["dataset"] == dataset for rows_in_partition in partitions.values() for row in rows_in_partition),
            "source_groups": len(
                {
                    row["source_group"]
                    for rows_in_partition in partitions.values()
                    for row in rows_in_partition
                    if row["dataset"] == dataset
                }
            ),
        }
        for dataset in ("dataset115_filtered", "dataset_jacky")
    }
    if actual_inventory != {
        "dataset115_filtered": {"tiles": 6319, "source_groups": 97},
        "dataset_jacky": {"tiles": 743, "source_groups": 16},
    }:
        raise ValueError(f"unexpected source inventory for {expert}: {actual_inventory}")

    partition_stats: dict[str, dict[str, Any]] = {}
    locked_groups: dict[str, dict[str, list[str]]] = {}
    for name, rows_in_partition in partitions.items():
        dataset_tiles = {
            dataset: sum(row["dataset"] == dataset for row in rows_in_partition)
            for dataset in ("dataset115_filtered", "dataset_jacky")
        }
        dataset_groups = {
            dataset: sorted({row["source_group"] for row in rows_in_partition if row["dataset"] == dataset})
            for dataset in ("dataset115_filtered", "dataset_jacky")
        }
        partition_stats[name] = {
            "tiles": len(rows_in_partition),
            "dataset_tiles": dataset_tiles,
            "dataset_source_groups": {dataset: len(groups) for dataset, groups in dataset_groups.items()},
        }
        locked_groups[name] = dataset_groups

    expected_groups = {"training": (68, 16), "validation": (14, 0), "test": (15, 0)}
    for name, expected in expected_groups.items():
        counts = (
            partition_stats[name]["dataset_source_groups"]["dataset115_filtered"],
            partition_stats[name]["dataset_source_groups"]["dataset_jacky"],
        )
        if counts != expected:
            raise ValueError(f"unexpected group split for {expert}/{name}: {counts} != {expected}")
    if partition_stats["training"]["dataset_tiles"]["dataset_jacky"] != 743:
        raise ValueError(f"not all Jacky tiles are training for {expert}")

    collision_groups = sorted(
        set(locked_groups["training"]["dataset115_filtered"])
        & set(locked_groups["training"]["dataset_jacky"])
    )
    expected_collisions = ["KJTHT-SC-R-A4-3"]
    if collision_groups != expected_collisions:
        raise ValueError(f"unexpected cross-dataset source collisions: {collision_groups}")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in partitions[name]]
        for name in PARTITIONS
    }
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for name in PARTITIONS
        for row in partitions[name]
    ]
    inventory_content = sorted(adopted_content, key=lambda row: (row["dataset"], row["source_group"], row["tile"]))
    test_membership = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for row in partitions["test"]
    ]
    source_hashes = _source_hashes(expert)

    manifest: dict[str, Any] = {
        "schema_version": 8,
        "split_id": SPLIT_ID,
        "expert": expert,
        "expert_raw_ids": EXPERT_RAW_IDS[expert],
        "target_contract": {
            "dataset_jacky": "class-index uint8; selected raw IDs become foreground; 255 remains ignore",
            "dataset115_filtered": "expert-specific binary uint8 mask; 255 becomes foreground",
        },
        "dataset_policy": DATASET_POLICY,
        "dataset_roots": {
            dataset: (DATA_ROOT_RELATIVE / dataset).as_posix()
            for dataset in ("dataset115_filtered", "dataset_jacky")
        },
        "split_policy": SPLIT_POLICY,
        "inventory": {**actual_inventory, "logical_total_tiles": 7062},
        "partition_stats": partition_stats,
        "locked_groups": locked_groups,
        "jacky_source_group_collisions": collision_groups,
        "ranked_dataset115_groups": ranked_groups,
        "source_hashes": source_hashes,
        "dataset_sha256": _canonical_hash(inventory_content),
        "adopted_dataset_sha256": _canonical_hash(adopted_content),
        "class_sha256": source_hashes[(DATA_ROOT_RELATIVE / "dataset_jacky/classes.txt").as_posix()],
        "split_sha256": _canonical_hash(membership),
        "test_membership_sha256": _canonical_hash(test_membership),
        "exclusion_sha256": _canonical_hash([]),
        **partitions,
    }
    return manifest


def main() -> None:
    source_dir = OUTPUT_DIR / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in ("independent_expert_split_proposal.csv", "independent_expert_split_summary.csv"):
        (source_dir / name).write_bytes((SOURCE_DIR / name).read_bytes())
    proposal = _read_csv(SOURCE_DIR / "independent_expert_split_proposal.csv")
    for expert in EXPERT_RAW_IDS:
        source_rows = _read_csv(SOURCE_DIR / f"merged_training_manifest_{expert}.csv")
        transfer_rows = [_portable_csv_row(row) for row in source_rows]
        transfer_csv = OUTPUT_DIR / f"merged_training_manifest_{expert}.csv"
        transfer_csv.parent.mkdir(parents=True, exist_ok=True)
        with transfer_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(transfer_rows[0]))
            writer.writeheader()
            writer.writerows(transfer_rows)
        manifest = _build_manifest(expert, transfer_rows, _ranked_groups(expert, proposal))
        destination = OUTPUT_DIR / f"{expert}.json"
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{destination.relative_to(PROJECT_ROOT)}: {len(manifest['training'])}/{len(manifest['validation'])}/{len(manifest['test'])} rows")


if __name__ == "__main__":
    main()
