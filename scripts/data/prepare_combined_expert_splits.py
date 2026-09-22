"""Build deterministic 70/15/15 manifests for the three SAM3 experts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.data.analyze_deterioration_dataset import scan_primary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JACKY = ROOT / "dataset_jacky"
DEFAULT_DATASET115 = ROOT / "dataset115_filtered"
DEFAULT_OUTPUT = ROOT / "outputs" / "deterioration_statistics" / "combined_experts"
IMAGE_PIXELS = 512 * 512
SEED = 42
TARGET_RATIOS = {"training": 0.70, "validation": 0.15, "test": 0.15}
EXPERT_IDS: dict[str, dict[str, tuple[int, ...]]] = {
    "scratch_crack": {"dataset_jacky": (1,), "dataset115_filtered": (1, 11)},
    "shrinkage_craquelure": {"dataset_jacky": (3, 4), "dataset115_filtered": (3, 4)},
    "loss": {"dataset_jacky": (2,), "dataset115_filtered": (2,)},
}
REQUIRED_LOSS_TRAIN_GROUP = "KJWTomh-SC-M-A7'-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _actual_path(root: Path, metadata_path: str) -> Path:
    candidates = (root / metadata_path, root / metadata_path.replace("'", "_"))
    matches = [path.resolve() for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"metadata path does not resolve to a file: {metadata_path}")
    return matches[0]


@dataclass(frozen=True)
class Sample:
    dataset: str
    source_group: str
    source_image_key: str
    tile: str
    image: Path
    mask: Path | None
    masks: Mapping[int, Path]
    pixels: Mapping[int, int]

    def positive_pixels(self, raw_ids: Sequence[int]) -> int:
        if self.mask is not None:
            return sum(self.pixels.get(raw_id, 0) for raw_id in raw_ids)
        union = np.zeros((512, 512), dtype=bool)
        for raw_id in raw_ids:
            path = self.masks.get(raw_id)
            if path is not None:
                with Image.open(path) as mask_file:
                    union |= np.asarray(mask_file) > 0
        return int(union.sum())


@dataclass(frozen=True)
class SourceGroup:
    dataset: str
    name: str
    samples: tuple[Sample, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.name

    @property
    def tile_count(self) -> int:
        return len(self.samples)

    def positive_pixels(self, expert: str) -> int:
        raw_ids = EXPERT_IDS[expert][self.dataset]
        return sum(sample.positive_pixels(raw_ids) for sample in self.samples)


def _jacky_groups(root: Path) -> list[SourceGroup]:
    tiles = scan_primary(root, {0, 1, 2, 3, 4, 5, 255})
    grouped: dict[str, list[Sample]] = {}
    for tile in tiles:
        grouped.setdefault(tile.source_group, []).append(Sample(
            dataset="dataset_jacky",
            source_group=tile.source_group,
            source_image_key=tile.source_group,
            tile=tile.tile,
            image=tile.image.resolve(),
            mask=tile.mask.resolve(),
            masks={},
            pixels=tile.raw_pixels,
        ))
    if len(tiles) != 743 or len(grouped) != 16:
        raise ValueError(f"dataset_jacky must contain 743 tiles from 16 source images; got {len(tiles)}/{len(grouped)}")
    return [SourceGroup("dataset_jacky", name, tuple(sorted(samples, key=lambda item: item.tile))) for name, samples in sorted(grouped.items())]


def _dataset115_groups(root: Path) -> list[SourceGroup]:
    manifest = root / "metadata" / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_group", "source_image_key", "image", "mask_class_id", "mask"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"dataset115 manifest must contain {sorted(required)}")
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)
    grouped: dict[str, list[Sample]] = {}
    for image_metadata, tile_rows in sorted(by_image.items()):
        first = tile_rows[0]
        source_group = first["image_group"]
        if any(row["image_group"] != source_group or row["source_image_key"] != first["source_image_key"] for row in tile_rows):
            raise ValueError(f"inconsistent source-image metadata for {image_metadata}")
        image = _actual_path(root, image_metadata)
        masks: dict[int, Path] = {}
        pixels: dict[int, int] = {}
        for row in tile_rows:
            raw_id = int(row["mask_class_id"])
            if raw_id in masks:
                raise ValueError(f"duplicate D-code mask for {image_metadata}: {raw_id}")
            masks[raw_id] = _actual_path(root, row["mask"])
            pixels[raw_id] = int(row["mask_foreground_pixels"])
        grouped.setdefault(source_group, []).append(Sample(
            dataset="dataset115_filtered",
            source_group=source_group,
            source_image_key=first["source_image_key"],
            tile=image.name,
            image=image,
            mask=None,
            masks=masks,
            pixels=pixels,
        ))
    if len(by_image) != 715 or len(grouped) != 17:
        raise ValueError(f"dataset115_filtered must contain 715 tiles from 17 source images; got {len(by_image)}/{len(grouped)}")
    return [SourceGroup("dataset115_filtered", name, tuple(sorted(samples, key=lambda item: item.tile))) for name, samples in sorted(grouped.items())]


def _select_test(groups: Sequence[SourceGroup], expert: str, target_tiles: int) -> tuple[SourceGroup, ...]:
    eligible = [group for group in groups if group.dataset == "dataset115_filtered"]
    if expert == "loss":
        eligible = [group for group in eligible if group.name != REQUIRED_LOSS_TRAIN_GROUP]
    group_pixels = {group.key: group.positive_pixels(expert) for group in eligible}
    best: tuple[tuple[Any, ...], tuple[SourceGroup, ...]] | None = None
    for size in range(1, len(eligible) + 1):
        for chosen in itertools.combinations(eligible, size):
            tile_count = sum(group.tile_count for group in chosen)
            positive_pixels = sum(group_pixels[group.key] for group in chosen)
            if positive_pixels == 0:
                continue
            density = positive_pixels / (tile_count * IMAGE_PIXELS)
            score = (abs(tile_count - target_tiles), -density, tuple(group.key for group in chosen))
            if best is None or score < best[0]:
                best = score, chosen
    if best is None:
        raise ValueError(f"no positive test split is available for {expert}")
    return best[1]


def _select_validation(
    groups: Sequence[SourceGroup], expert: str, target_tiles: int, excluded: set[tuple[str, str]],
) -> tuple[SourceGroup, ...]:
    candidates = [group for group in groups if group.key not in excluded]
    if expert == "loss":
        candidates = [group for group in candidates if group.name != REQUIRED_LOSS_TRAIN_GROUP]
    cached_pixels = {group.key: group.positive_pixels(expert) for group in candidates}
    # tile count -> (positive pixels, group tuple); one deterministic best candidate per count
    states: dict[int, tuple[int, tuple[SourceGroup, ...]]] = {0: (0, ())}
    for group in candidates:
        additions: dict[int, tuple[int, tuple[SourceGroup, ...]]] = {}
        group_pixels = cached_pixels[group.key]
        for count, (pixels, chosen) in states.items():
            new_count = count + group.tile_count
            candidate = (pixels + group_pixels, (*chosen, group))
            current = additions.get(new_count, states.get(new_count))
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and tuple(item.key for item in candidate[1]) < tuple(item.key for item in current[1])
            ):
                additions[new_count] = candidate
        for count, candidate in additions.items():
            current = states.get(count)
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and tuple(item.key for item in candidate[1]) < tuple(item.key for item in current[1])
            ):
                states[count] = candidate
    viable = [
        (abs(count - target_tiles), -pixels, tuple(group.key for group in chosen), chosen)
        for count, (pixels, chosen) in states.items()
        if chosen and pixels > 0
    ]
    if not viable:
        raise ValueError(f"no positive validation split is available for {expert}")
    return min(viable)[-1]


def _record(sample: Sample, expert: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": sample.dataset,
        "source_group": sample.source_group,
        "source_image_key": sample.source_image_key,
        "tile": sample.tile,
        "image": str(sample.image),
        "image_sha256": _sha256(sample.image),
    }
    if sample.mask is not None:
        row.update(mask_format="class_index", mask=str(sample.mask), mask_sha256=_sha256(sample.mask))
    else:
        paths = [sample.masks[raw_id] for raw_id in EXPERT_IDS[expert][sample.dataset] if raw_id in sample.masks]
        row.update(
            mask_format="binary_multilabel",
            positive_masks=[str(path) for path in paths],
            positive_mask_sha256=[_sha256(path) for path in paths],
        )
    return row


def _content_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_format")}
    identity["mask_sha256" if row["mask_format"] == "class_index" else "positive_mask_sha256"] = (
        row["mask_sha256"] if row["mask_format"] == "class_index" else row["positive_mask_sha256"]
    )
    return identity


def _build_manifest(groups: Sequence[SourceGroup], expert: str, jacky: Path, dataset115: Path) -> dict[str, Any]:
    total_tiles = sum(group.tile_count for group in groups)
    target_tiles = round(total_tiles * 0.15)
    test_groups = _select_test(groups, expert, target_tiles)
    test_keys = {group.key for group in test_groups}
    validation_groups = _select_validation(groups, expert, target_tiles, test_keys)
    validation_keys = {group.key for group in validation_groups}
    training_groups = tuple(group for group in groups if group.key not in test_keys | validation_keys)
    partitions = {"training": training_groups, "validation": validation_groups, "test": test_groups}
    partition_tiles = {name: sum(group.tile_count for group in members) for name, members in partitions.items()}
    partition_pixels = {name: sum(group.positive_pixels(expert) for group in members) for name, members in partitions.items()}
    if any(value == 0 for value in partition_pixels.values()):
        raise ValueError(f"{expert} has a zero-positive partition")
    if expert == "loss" and not any(group.name == REQUIRED_LOSS_TRAIN_GROUP for group in training_groups):
        raise ValueError(f"{REQUIRED_LOSS_TRAIN_GROUP} is not in the loss training split")

    records = {
        name: [_record(sample, expert) for group in sorted(members, key=lambda item: item.key) for sample in group.samples]
        for name, members in partitions.items()
    }
    group_sets = {name: {group.key for group in members} for name, members in partitions.items()}
    image_hash_sets = {name: {row["image_sha256"] for row in rows} for name, rows in records.items()}
    overlaps: dict[str, list[Any]] = {}
    for left, right in (("training", "validation"), ("training", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}_groups"] = sorted(group_sets[left] & group_sets[right])
        overlaps[f"{left}_{right}_image_hashes"] = sorted(image_hash_sets[left] & image_hash_sets[right])
    leakage_passed = not any(overlaps.values()) and all(
        row["dataset"] == "dataset115_filtered" for row in records["test"]
    )
    if not leakage_passed:
        raise ValueError(f"{expert} split failed leakage audit: {overlaps}")

    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in rows]
        for name, rows in records.items()
    }
    content = [_content_identity(row) for name in ("training", "validation", "test") for row in records[name]]
    test_pixels = partition_pixels["test"]
    return {
        "schema_version": 5,
        "seed": SEED,
        "expert": expert,
        "expert_raw_ids": list(EXPERT_IDS[expert]["dataset_jacky"]),
        "dataset_expert_raw_ids": {
            dataset: list(raw_ids) for dataset, raw_ids in EXPERT_IDS[expert].items()
        },
        "dataset_sha256": _canonical_hash({
            "dataset_jacky_manifest": _sha256(jacky / "manifest.json"),
            "dataset115_manifest": _sha256(dataset115 / "metadata" / "manifest.csv"),
        }),
        "class_sha256": _canonical_hash({
            "dataset_jacky": _sha256(jacky / "classes.txt"),
            "dataset115_filtered": _sha256(dataset115 / "metadata" / "classes.json"),
            "expert_ids": EXPERT_IDS[expert],
        }),
        "dataset_policy": {
            "dataset_jacky": "training_or_validation_only",
            "dataset115_filtered": "training_validation_or_test",
        },
        "target_ratios": TARGET_RATIOS,
        "partition_tiles": partition_tiles,
        "partition_ratios": {name: count / total_tiles for name, count in partition_tiles.items()},
        "partition_positive_pixels": partition_pixels,
        "selected_groups": {name: [{"dataset": group.dataset, "source_group": group.name} for group in members] for name, members in partitions.items()},
        "test_selection": {
            "rule": "minimum absolute tile-count deviation from 15%, then maximum expert foreground density",
            "target_tiles": target_tiles,
            "foreground_density": test_pixels / (partition_tiles["test"] * IMAGE_PIXELS),
        },
        **records,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash(content),
        "leakage_audit": {**overlaps, "dataset_jacky_test_tiles": 0, "passed": leakage_passed},
    }


def prepare_splits(jacky_root: Path, dataset115_root: Path, output_root: Path) -> dict[str, dict[str, Any]]:
    jacky, dataset115, output = jacky_root.resolve(), dataset115_root.resolve(), output_root.resolve()
    groups = [*_jacky_groups(jacky), *_dataset115_groups(dataset115)]
    output.mkdir(parents=True, exist_ok=True)
    manifests = {expert: _build_manifest(groups, expert, jacky, dataset115) for expert in EXPERT_IDS}
    for expert, manifest in manifests.items():
        (output / f"{expert}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacky", type=Path, default=DEFAULT_JACKY)
    parser.add_argument("--dataset115", type=Path, default=DEFAULT_DATASET115)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifests = prepare_splits(args.jacky, args.dataset115, args.output)
    print(json.dumps({
        expert: {
            "manifest": str((args.output / f"{expert}.json").resolve()),
            "partition_tiles": manifest["partition_tiles"],
            "partition_ratios": manifest["partition_ratios"],
            "test_groups": manifest["selected_groups"]["test"],
            "test_foreground_density": manifest["test_selection"]["foreground_density"],
        }
        for expert, manifest in manifests.items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
