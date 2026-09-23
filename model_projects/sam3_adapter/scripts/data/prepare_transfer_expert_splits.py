"""Build prepared expert data from Jacky and one cumulative Dataset115 release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from model_projects.sam3_adapter.scripts.data.prepare_combined_expert_splits import (
    DEFAULT_DATASET,
    DEFAULT_JACKY,
    SOURCE_SIZE,
    Sample,
    SourceGroup,
    _canonical_hash,
    _dataset_release_groups,
    _jacky_groups,
    _sha256,
    expert_raw_ids,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "deterioration_statistics" / "transfer_512"
DEFAULT_RELEASE_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "deterioration_statistics" / "prepared"
SPLITS = {
    "scratch_crack": {
        "validation": ("KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "MST-SC-M-A2-2-7"),
        "test": ("KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A6'-2"),
        "unused_holdout": (),
    },
    "loss": {
        "validation": ("KJLYT-SC-M-A4-7", "KJTHT-PH-M-2RB1-3", "KJWTomh-PH-M-1RB1-1"),
        "test": ("MST-SC-M-A2-2-7", "KJTHT-SC-R-A4-6", "KJWTomh-MH-M-A3E-3-2", "MST-SC-M-A2-2-6"),
        "unused_holdout": (),
    },
    "shrinkage_craquelure": {
        "validation": ("KJWTomh-MH-M-A3E-1", "WFT-PH-M-1LB1-1-1", "KJWTomh-PH-M-1RB1-1"),
        "test": ("KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1"),
        "unused_holdout": ("KJWTomh-MH-M-A3E-2", "KJWTomh-MH-M-A3E-3-2"),
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
    padded = np.pad(skeleton, radius, constant_values=False)
    dilated = np.zeros_like(skeleton)
    height, width = skeleton.shape
    for y, x in np.argwhere(disk):
        dilated |= padded[y:y + height, x:x + width]
    return dilated.astype(np.uint8) * 255


def _verify_metadata(root: Path) -> dict[str, int]:
    with (root / "metadata" / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    from model_projects.sam3_adapter.scripts.data.prepare_combined_expert_splits import _actual_path
    checked_images: set[Path] = set()
    for row in rows:
        image = _actual_path(root, row["image"])
        mask = _actual_path(root, row["mask"])
        if image not in checked_images:
            if _sha256(image) != row["image_sha256"]:
                raise ValueError(f"{root.name} image SHA mismatch: {image}")
            with Image.open(image) as file:
                if file.size != (SOURCE_SIZE, SOURCE_SIZE):
                    raise ValueError(f"{root.name} image must be {SOURCE_SIZE}x{SOURCE_SIZE}: {image}")
            checked_images.add(image)
        if _sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"{root.name} mask SHA mismatch: {mask}")
        with Image.open(mask) as file:
            array = np.asarray(file)
        if array.shape != (SOURCE_SIZE, SOURCE_SIZE) or not set(np.unique(array)).issubset({0, 255}):
            raise ValueError(f"invalid binary mask: {mask}")
        if int((array == 255).sum()) != int(row["mask_foreground_pixels"]):
            raise ValueError(f"{root.name} mask foreground count mismatch: {mask}")
    if not checked_images:
        raise ValueError(f"{root.name} contains no images")
    return {"metadata_rows": len(rows), "unique_tiles": len(checked_images)}


def _kyt_views(groups: list[SourceGroup], output: Path) -> tuple[list[SourceGroup], dict[str, object]]:
    variant_root = output / "kyt_d04_7px"
    derived: list[SourceGroup] = []
    audit: dict[str, object] = {}
    for group in groups:
        if group.dataset == "dataset_jacky" or group.name not in KYT:
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


def _binary_target(sample: Sample, expert: str) -> np.ndarray:
    """Materialize one expert's final 0/1/255 target from the origin annotation."""

    raw_ids = expert_raw_ids(expert, sample.dataset)
    target = np.zeros((SOURCE_SIZE, SOURCE_SIZE), dtype=np.uint8)
    if sample.mask is not None:
        with Image.open(sample.mask) as file:
            raw_mask = np.asarray(file, dtype=np.uint8)
        if raw_mask.shape != target.shape:
            raise ValueError(f"expected a 512x512 class-index mask: {sample.mask}")
        target[np.isin(raw_mask, raw_ids)] = 1
        target[raw_mask == 255] = 255
        return target

    for raw_id in raw_ids:
        path = sample.masks.get(raw_id)
        if path is None:
            continue
        with Image.open(path) as file:
            binary_mask = np.asarray(file, dtype=np.uint8)
        if binary_mask.shape != target.shape or not set(np.unique(binary_mask)).issubset({0, 255}):
            raise ValueError(f"expected a 512x512 binary 0/255 mask: {path}")
        target[binary_mask > 0] = 1
    return target


def _materialize_training_row(sample: Sample, expert: str, output: Path) -> dict[str, object]:
    """Write the final binary target once and return its immutable manifest row."""

    target = _binary_target(sample, expert)
    image_hash = _sha256(sample.image)
    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
    target_path = output / "binary_targets" / expert / sample.dataset / f"{image_hash[:20]}_{target_hash[:20]}.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        with Image.open(target_path) as file:
            existing = np.asarray(file, dtype=np.uint8)
        if not np.array_equal(existing, target):
            raise ValueError(f"existing binary target differs: {target_path}")
    else:
        Image.fromarray(target, mode="L").save(target_path)
    return {
        "dataset": sample.dataset,
        "source_group": sample.source_group,
        "source_image_key": sample.source_image_key,
        "tile": sample.tile,
        "image": str(sample.image),
        "image_sha256": image_hash,
        "mask_format": "binary_target",
        "mask": str(target_path.resolve()),
        "mask_sha256": _sha256(target_path),
        "mask_foreground_pixels": int((target == 1).sum()),
    }


def _training_content_identity(row: dict[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in ("dataset", "source_group", "tile", "image_sha256", "mask_format", "mask_sha256")
    }


PREPARED_SCHEMA_VERSION = 8


def _manifest(
    groups: list[SourceGroup],
    expert: str,
    jacky: Path,
    dataset: Path,
    output: Path,
    variant_audit: dict[str, object],
) -> dict[str, object]:
    spec = SPLITS[expert]
    dataset_id = dataset.name
    release_names = {group.name for group in groups if group.dataset == dataset_id}
    required = set(spec["validation"]) | set(spec["test"]) | set(spec["unused_holdout"])
    if not required.issubset(release_names):
        raise ValueError(f"{dataset_id} is missing fixed source groups: {sorted(required - release_names)}")
    partitions = {
        "training": [
            group
            for group in groups
            if group.dataset == "dataset_jacky"
            or (group.dataset == dataset_id and group.name not in required)
        ],
        "validation": [
            group
            for group in groups
            if group.dataset == dataset_id and group.name in spec["validation"]
        ],
        "test": [
            group
            for group in groups
            if group.dataset == dataset_id and group.name in spec["test"]
        ],
    }
    unused = [
        group
        for group in groups
        if group.dataset == dataset_id and group.name in spec["unused_holdout"]
    ]
    counts = tuple(sum(group.tile_count for group in partitions[name]) for name in ("training", "validation", "test"))
    rows = {
        name: [
            _materialize_training_row(sample, expert, output)
            for group in sorted(members, key=lambda item: item.key)
            for sample in group.samples
        ]
        for name, members in partitions.items()
    }
    membership = {
        name: [{key: row[key] for key in ("dataset", "source_group", "tile")} for row in members]
        for name, members in rows.items()
    }
    hashes = {name: {row["image_sha256"] for row in members} for name, members in rows.items()}
    for left, right in (("training", "validation"), ("training", "test"), ("validation", "test")):
        if hashes[left] & hashes[right]:
            raise ValueError(f"image SHA leakage: {left}/{right}")
    pixels = {
        name: sum(int(row["mask_foreground_pixels"]) for row in members)
        for name, members in rows.items()
    }
    if not all(pixels.values()):
        raise ValueError(f"zero foreground in {expert} partition")
    dataset_raw_ids = {
        "dataset_jacky": list(expert_raw_ids(expert, "dataset_jacky")),
        dataset_id: list(expert_raw_ids(expert, dataset_id)),
    }
    with (dataset / "metadata" / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        metadata_rows = sum(1 for _ in csv.DictReader(stream))
    return {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "seed": 42,
        "expert": expert,
        "dataset_release_id": dataset_id,
        "dataset_release": {
            "id": dataset_id,
            "root": str(dataset),
            "metadata_rows": metadata_rows,
            "tiles": sum(group.tile_count for group in groups if group.dataset == dataset_id),
        },
        "expert_raw_ids": dataset_raw_ids["dataset_jacky"],
        "dataset_expert_raw_ids": dataset_raw_ids,
        "dataset_sha256": _canonical_hash({
            "dataset_jacky": _sha256(jacky / "manifest.json"),
            dataset_id: _sha256(dataset / "metadata" / "manifest.csv"),
        }),
        "class_sha256": _canonical_hash({
            "dataset_jacky": _sha256(jacky / "classes.txt"),
            dataset_id: _sha256(dataset / "metadata" / "classes.json"),
            "expert_ids": dataset_raw_ids,
        }),
        "dataset_policy": {
            "dataset_jacky": "all_743_training_only",
            dataset_id: "fixed_benchmark_groups_new_groups_training",
        },
        "split_contract": {
            "validation_groups": list(spec["validation"]),
            "test_groups": list(spec["test"]),
            "unused_holdout_groups": list(spec["unused_holdout"]),
            "unlisted_release_groups": "training",
        },
        "target_materialization": {
            "format": "uint8_binary_target",
            "values": {"background": 0, "foreground": 1, "ignore": 255},
            "directory": str((output / "binary_targets" / expert).resolve()),
        },
        "selected_groups": {
            name: [{"dataset": group.dataset, "source_group": group.name} for group in members]
            for name, members in partitions.items()
        },
        "unused_holdout": [
            {"dataset": group.dataset, "source_group": group.name, "tile_count": group.tile_count}
            for group in unused
        ],
        "kyt_7px_audit": variant_audit if expert == "shrinkage_craquelure" else {},
        "partition_tiles": dict(zip(("training", "validation", "test"), counts, strict=True)),
        "partition_positive_pixels": pixels,
        **rows,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash([
            _training_content_identity(row)
            for name in ("training", "validation", "test")
            for row in rows[name]
        ]),
        "leakage_audit": {
            "passed": True,
            "source_group_overlap": [],
            "image_sha256_overlap": [],
            "jacky_validation_tiles": 0,
            "jacky_test_tiles": 0,
        },
    }


def prepare(jacky: Path, dataset: Path, output: Path) -> dict[str, dict[str, object]]:
    jacky, dataset, output = jacky.resolve(), dataset.resolve(), output.resolve()
    _verify_metadata(dataset)
    groups = [*_jacky_groups(jacky), *_dataset_release_groups(dataset)]
    output.mkdir(parents=True, exist_ok=True)
    groups, variant_audit = _kyt_views(groups, output)
    manifests = {
        expert: _manifest(groups, expert, jacky, dataset, output, variant_audit)
        for expert in SPLITS
    }
    for expert, manifest in manifests.items():
        target = output / f"{expert}.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing == manifest:
                continue
            if int(existing.get("schema_version", 0)) >= PREPARED_SCHEMA_VERSION:
                raise ValueError(f"refusing to overwrite a different manifest: {target}")
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifests


def _default_output(dataset: Path) -> Path:
    if dataset.resolve() == DEFAULT_DATASET.resolve():
        return DEFAULT_OUTPUT
    return DEFAULT_RELEASE_OUTPUT_ROOT / dataset.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacky", type=Path, default=DEFAULT_JACKY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or _default_output(args.dataset)
    manifests = prepare(args.jacky, args.dataset, output)
    print(json.dumps({expert: {"manifest": str((output / f"{expert}.json").resolve()), "tiles": data["partition_tiles"], "positive_pixels": data["partition_positive_pixels"], "split_sha256": data["split_sha256"]} for expert, data in manifests.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
