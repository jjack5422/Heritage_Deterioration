"""Rebuild the fixed DATASET_SPLIT_TRANSFER source-group contract."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from scripts.data.prepare_combined_expert_splits import (
    DEFAULT_DATASET115,
    DEFAULT_JACKY,
    EXPERT_IDS,
    Sample,
    SourceGroup,
    _canonical_hash,
    _content_identity,
    _dataset115_groups,
    _jacky_groups,
    _record,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "deterioration_statistics" / "transfer_512"
SPLITS = {
    "scratch_crack": {
        "validation": ("KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "MST-SC-M-A2-2-7"),
        "test": ("KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A6'-2"),
        "unused_holdout": (),
        "counts": (1241, 105, 112),
    },
    "loss": {
        "validation": ("KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "KJWTomh-PH-M-1RB1-1"),
        "test": ("MST-SC-M-A2-2-7", "KJTHT-SC-R-A4-6", "KJWTomh-MH-M-A3E-3-2", "MST-SC-M-A2-2-6"),
        "unused_holdout": (),
        "counts": (1243, 108, 107),
    },
    "shrinkage_craquelure": {
        "validation": ("KJWTomh-MH-M-A3E-1", "WFT-PH-M-1LB1-1-1", "KJWTomh-PH-M-1RB1-1"),
        "test": ("KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1"),
        "unused_holdout": ("KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A3E-3-2"),
        "counts": (1081, 123, 142),
    },
}
KYT = frozenset(SPLITS["shrinkage_craquelure"]["test"])


def zhang_suen_skeleton(foreground: np.ndarray) -> np.ndarray:
    """Apply both Zhang-Suen sub-iterations until no further deletion."""
    skeleton = foreground.astype(bool, copy=True)
    while True:
        changed = False
        for subiteration in (0, 1):
            padded = np.pad(skeleton, 1, constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbor.astype(np.uint8) for neighbor in neighbors)
            transitions = sum((~neighbors[index] & neighbors[(index + 1) % 8]).astype(np.uint8) for index in range(8))
            removal = skeleton & (count >= 2) & (count <= 6) & (transitions == 1)
            if subiteration == 0:
                removal &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                removal &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if removal.any():
                skeleton[removal] = False
                changed = True
        if not changed:
            return skeleton


def thin_7px(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (512, 512) or not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("KYT D-04 source must be a 512x512 binary 0/255 mask")
    radius = 3
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    skeleton = zhang_suen_skeleton(mask == 255)
    return binary_dilation(skeleton, structure=disk).astype(np.uint8) * 255


def _verify_metadata(root: Path) -> dict[str, int]:
    with (root / "metadata" / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    from scripts.data.prepare_combined_expert_splits import _actual_path
    checked_images: set[Path] = set()
    for row in rows:
        image = _actual_path(root, row["image"])
        mask = _actual_path(root, row["mask"])
        if image not in checked_images:
            if _sha256(image) != row["image_sha256"]:
                raise ValueError(f"Dataset115 image SHA mismatch: {image}")
            checked_images.add(image)
        if _sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"Dataset115 mask SHA mismatch: {mask}")
        with Image.open(mask) as file:
            array = np.asarray(file)
        if array.shape != (512, 512) or not set(np.unique(array)).issubset({0, 255}):
            raise ValueError(f"invalid binary mask: {mask}")
        if int((array == 255).sum()) != int(row["mask_foreground_pixels"]):
            raise ValueError(f"Dataset115 mask foreground count mismatch: {mask}")
    return {"metadata_rows": len(rows), "unique_tiles": len(checked_images)}


def _kyt_views(groups: list[SourceGroup], output: Path) -> tuple[list[SourceGroup], dict[str, object]]:
    variant_root = output / "kyt_d04_7px"
    derived: list[SourceGroup] = []
    audit: dict[str, object] = {}
    for group in groups:
        if group.dataset != "dataset115_filtered" or group.name not in KYT:
            derived.append(group)
            continue
        updated: list[Sample] = []
        positive_tiles = pixels = source_masks = 0
        group_dir = variant_root / ("A9" if group.name.endswith("A9-4") else "2LB1")
        group_dir.mkdir(parents=True, exist_ok=True)
        for sample in group.samples:
            source = sample.masks.get(4)
            if source is None:
                updated.append(replace(sample, masks={}, pixels={}))
                continue
            source_masks += 1
            with Image.open(source) as file:
                original = np.asarray(file, dtype=np.uint8)
            transformed = thin_7px(original)
            target = group_dir / f"{_sha256(sample.image)[:20]}.png"
            if target.exists():
                with Image.open(target) as file:
                    if not np.array_equal(np.asarray(file), transformed):
                        raise ValueError(f"existing 7px variant differs: {target}")
            else:
                Image.fromarray(transformed).save(target)
            count = int((transformed == 255).sum())
            positive_tiles += count > 0
            pixels += count
            updated.append(replace(sample, masks={4: target.resolve()}, pixels={4: count}))
        derived.append(replace(group, samples=tuple(updated)))
        audit[group.name] = {"tiles": group.tile_count, "source_d04_masks": source_masks, "positive_tiles": positive_tiles, "foreground_pixels": pixels}
    return derived, audit


def _manifest(groups: list[SourceGroup], expert: str, jacky: Path, dataset115: Path, variant_audit: dict[str, object]) -> dict[str, object]:
    spec = SPLITS[expert]
    dataset115_names = {group.name for group in groups if group.dataset == "dataset115_filtered"}
    required = set(spec["validation"]) | set(spec["test"]) | set(spec["unused_holdout"])
    if not required.issubset(dataset115_names):
        raise ValueError(f"missing fixed source groups: {sorted(required - dataset115_names)}")
    partitions = {
        "training": [group for group in groups if group.dataset == "dataset_jacky" or group.name not in required],
        "validation": [group for group in groups if group.dataset == "dataset115_filtered" and group.name in spec["validation"]],
        "test": [group for group in groups if group.dataset == "dataset115_filtered" and group.name in spec["test"]],
    }
    unused = [group for group in groups if group.dataset == "dataset115_filtered" and group.name in spec["unused_holdout"]]
    counts = tuple(sum(group.tile_count for group in partitions[name]) for name in ("training", "validation", "test"))
    if counts != spec["counts"] or sum(group.tile_count for group in unused) != (112 if expert == "shrinkage_craquelure" else 0):
        raise ValueError(f"fixed partition count mismatch for {expert}: {counts}")
    rows = {name: [_record(sample, expert) for group in sorted(members, key=lambda item: item.key) for sample in group.samples] for name, members in partitions.items()}
    membership = {name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in members] for name, members in rows.items()}
    hashes = {name: {row["image_sha256"] for row in members} for name, members in rows.items()}
    for left, right in (("training", "validation"), ("training", "test"), ("validation", "test")):
        if hashes[left] & hashes[right]:
            raise ValueError(f"image SHA leakage: {left}/{right}")
    pixels = {name: sum(group.positive_pixels(expert) for group in members) for name, members in partitions.items()}
    if not all(pixels.values()):
        raise ValueError(f"zero foreground in {expert} partition")
    return {
        "schema_version": 6,
        "seed": 42,
        "expert": expert,
        "expert_raw_ids": list(EXPERT_IDS[expert]["dataset_jacky"]),
        "dataset_expert_raw_ids": {dataset: list(ids) for dataset, ids in EXPERT_IDS[expert].items()},
        "dataset_sha256": _canonical_hash({"jacky_manifest": _sha256(jacky / "manifest.json"), "dataset115_manifest": _sha256(dataset115 / "metadata" / "manifest.csv")}),
        "class_sha256": _canonical_hash({"jacky_classes": _sha256(jacky / "classes.txt"), "dataset115_classes": _sha256(dataset115 / "metadata" / "classes.json"), "expert_ids": EXPERT_IDS[expert]}),
        "dataset_policy": {"dataset_jacky": "all_743_training_only", "dataset115_filtered": "fixed_source_group_train_validation_test"},
        "selected_groups": {name: [{"dataset": group.dataset, "source_group": group.name} for group in members] for name, members in partitions.items()},
        "unused_holdout": [{"dataset": group.dataset, "source_group": group.name, "tile_count": group.tile_count} for group in unused],
        "kyt_7px_audit": variant_audit if expert == "shrinkage_craquelure" else {},
        "partition_tiles": dict(zip(("training", "validation", "test"), counts, strict=True)),
        "partition_positive_pixels": pixels,
        **rows,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash([_content_identity(row) for name in ("training", "validation", "test") for row in rows[name]]),
        "leakage_audit": {"passed": True, "source_group_overlap": [], "image_sha256_overlap": [], "jacky_validation_tiles": 0, "jacky_test_tiles": 0},
    }


def prepare(jacky: Path, dataset115: Path, output: Path) -> dict[str, dict[str, object]]:
    jacky, dataset115, output = jacky.resolve(), dataset115.resolve(), output.resolve()
    metadata = _verify_metadata(dataset115)
    if metadata["unique_tiles"] != 715:
        raise ValueError(f"Dataset115 tile count mismatch: {metadata}")
    groups = [*_jacky_groups(jacky), *_dataset115_groups(dataset115)]
    output.mkdir(parents=True, exist_ok=True)
    groups, variant_audit = _kyt_views(groups, output)
    manifests = {expert: _manifest(groups, expert, jacky, dataset115, variant_audit) for expert in SPLITS}
    for expert, manifest in manifests.items():
        target = output / f"{expert}.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError(f"refusing to overwrite a different manifest: {target}")
        else:
            target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacky", type=Path, default=DEFAULT_JACKY)
    parser.add_argument("--dataset115", type=Path, default=DEFAULT_DATASET115)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifests = prepare(args.jacky, args.dataset115, args.output)
    print(json.dumps({expert: {"tiles": data["partition_tiles"], "positive_pixels": data["partition_positive_pixels"], "split_sha256": data["split_sha256"]} for expert, data in manifests.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
