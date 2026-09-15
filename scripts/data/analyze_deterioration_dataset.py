"""Analyze raw deterioration classes and choose leakage-safe validation groups.

The primary dataset is laid out as ``<temple>/<source_group>/{image,mask}_tiles``.
Dataset114 is read through ``metadata/manifest.csv`` so its source-image grouping
is explicit. ``dataset_v2_3class`` is read through its manifest and metadata.
Masks are single-channel class-ID images; unknown IDs are fatal. The generated
class-pixel CSV and Markdown report include background and ignored pixels. Each
dataset also receives its own named output subdirectory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

SEED = 42
BACKGROUND_ID = 0
RAW_IDS = tuple(range(1, 38))
MERGED_TARGETS: dict[str, tuple[int, ...]] = {
    "shrinkage_craquelure": (3, 4),
    "scratch_crack": (11, 1),
    "abrasion_loss_rodent_bite": (16, 2, 36),
}


@dataclass(frozen=True)
class Tile:
    dataset: str
    source_group: str
    tile: str
    image: Path
    mask: Path
    total_pixels: int
    raw_pixels: Mapping[int, int]

    @property
    def valid_pixels(self) -> int:
        return self.total_pixels

    def merged_pixels(self, raw_ids: Sequence[int]) -> int:
        return sum(self.raw_pixels.get(raw_id, 0) for raw_id in raw_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_classes(path: Path) -> dict[int, tuple[str, str]]:
    classes: dict[int, tuple[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"Invalid class row at {path}:{line_number}")
        raw_id, code, name = int(fields[0]), fields[1], fields[2]
        if raw_id in classes:
            raise ValueError(f"Duplicate raw ID {raw_id} in {path}")
        classes[raw_id] = (code, name)
    expected = {BACKGROUND_ID, *RAW_IDS}
    if set(classes) != expected:
        raise ValueError(f"Class catalog must define exactly IDs 0..37; got {sorted(classes)}")
    return classes


def _read_mask(mask_path: Path, image_path: Path, known_ids: set[int]) -> tuple[int, dict[int, int]]:
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)
    if mask.ndim != 2:
        raise ValueError(f"Expected single-channel class-ID mask: {mask_path}")
    with Image.open(image_path) as image:
        image_size = (image.height, image.width)
    if mask.shape != image_size:
        raise ValueError(f"Image/mask dimensions differ: {image_path} and {mask_path}")
    ids, counts = np.unique(mask, return_counts=True)
    unknown = sorted(int(value) for value in ids if int(value) not in known_ids)
    if unknown:
        raise ValueError(f"Unknown raw ID(s) {unknown} in {mask_path}")
    return int(mask.size), {int(raw_id): int(count) for raw_id, count in zip(ids, counts, strict=True)}


def scan_primary(root: Path, known_ids: set[int]) -> list[Tile]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    records: list[Tile] = []
    for mask_dir in sorted(root.glob("*/*/mask_tiles")):
        source_group = mask_dir.parent.name
        image_dir = mask_dir.parent / "image_tiles"
        if not image_dir.is_dir():
            raise ValueError(f"Missing image tile directory: {image_dir}")
        for mask_path in sorted(mask_dir.glob("*.png")):
            matches = [path for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff") if (path := image_dir / f"{mask_path.stem}{suffix}").is_file()]
            if len(matches) != 1:
                raise ValueError(f"Expected one matching image for {mask_path}; found {len(matches)}")
            image_path = matches[0]
            total, counts = _read_mask(mask_path, image_path, known_ids)
            records.append(Tile(root.name, source_group, image_path.name, image_path, mask_path, total, counts))
    if not records:
        raise ValueError(f"No primary mask tiles found under {root}")
    return records


def scan_dataset114(root: Path, known_ids: set[int]) -> list[Tile]:
    root = root.resolve()
    manifest_path = root / "metadata/manifest.csv"
    if not manifest_path.is_file():
        raise ValueError(f"Missing Dataset114 manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_group", "image", "mask"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Dataset114 manifest must contain {sorted(required)}")
    records: list[Tile] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        image_path, mask_path = root / row["image"], root / row["mask"]
        key = (row["image"], row["mask"])
        if key in seen:
            raise ValueError(f"Duplicate Dataset114 manifest pair: {key}")
        seen.add(key)
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"Missing Dataset114 image/mask pair: {image_path}, {mask_path}")
        total, counts = _read_mask(mask_path, image_path, known_ids)
        records.append(Tile(root.name, row["image_group"], image_path.name, image_path, mask_path, total, counts))
    return records


def _counter_for(tiles: Iterable[Tile], raw_ids: Sequence[int]) -> tuple[Counter[int], Counter[int], dict[int, set[tuple[str, str]]]]:
    pixels: Counter[int] = Counter()
    occurrences: Counter[int] = Counter()
    groups: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for tile in tiles:
        for raw_id in raw_ids:
            count = tile.raw_pixels.get(raw_id, 0)
            pixels[raw_id] += count
            if count:
                occurrences[raw_id] += 1
                groups[raw_id].add((tile.dataset, tile.source_group))
    return pixels, occurrences, groups


def raw_statistics(tiles: Sequence[Tile], classes: Mapping[int, tuple[str, str]]) -> list[dict[str, object]]:
    pixels, occurrences, groups = _counter_for(tiles, RAW_IDS)
    valid_pixels = sum(tile.valid_pixels for tile in tiles)
    tile_count = len(tiles)
    return [{
        "raw_id": raw_id,
        "code": classes[raw_id][0],
        "class_name": classes[raw_id][1],
        "pixel_count": pixels[raw_id],
        "tile_count": occurrences[raw_id],
        "source_group_count": len(groups[raw_id]),
        "pixel_ratio": pixels[raw_id] / valid_pixels if valid_pixels else 0.0,
        "tile_ratio": occurrences[raw_id] / tile_count if tile_count else 0.0,
    } for raw_id in RAW_IDS]


def merged_statistics(tiles: Sequence[Tile]) -> list[dict[str, object]]:
    rows = []
    for target, raw_ids in MERGED_TARGETS.items():
        positive_tiles = [tile for tile in tiles if tile.merged_pixels(raw_ids) > 0]
        rows.append({
            "target_name": target,
            "raw_ids": ";".join(str(raw_id) for raw_id in raw_ids),
            "pixel_count": sum(tile.merged_pixels(raw_ids) for tile in tiles),
            "tile_count": len(positive_tiles),
            "source_group_count": len({(tile.dataset, tile.source_group) for tile in positive_tiles}),
        })
    return rows


def _wide_summary(dataset: str, source_group: str, tiles: Sequence[Tile]) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "source_group": source_group,
        "source_group_count": len({tile.source_group for tile in tiles}),
        "tile_count": len(tiles),
        "total_pixels": sum(tile.total_pixels for tile in tiles),
        "valid_pixels": sum(tile.valid_pixels for tile in tiles),
    }
    for raw_id in RAW_IDS:
        row[f"id_{raw_id}_pixels"] = sum(tile.raw_pixels.get(raw_id, 0) for tile in tiles)
        row[f"id_{raw_id}_tiles"] = sum(tile.raw_pixels.get(raw_id, 0) > 0 for tile in tiles)
    for target, raw_ids in MERGED_TARGETS.items():
        row[f"{target}_pixels"] = sum(tile.merged_pixels(raw_ids) for tile in tiles)
        row[f"{target}_tiles"] = sum(tile.merged_pixels(raw_ids) > 0 for tile in tiles)
    return row


def grouped_statistics(tiles: Sequence[Tile]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_dataset: dict[str, list[Tile]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[Tile]] = defaultdict(list)
    for tile in tiles:
        by_dataset[tile.dataset].append(tile)
        by_group[(tile.dataset, tile.source_group)].append(tile)
    dataset_rows = [_wide_summary(dataset, "", members) for dataset, members in sorted(by_dataset.items())]
    group_rows = [_wide_summary(dataset, group, members) for (dataset, group), members in sorted(by_group.items())]
    return dataset_rows, group_rows


def per_tile_statistics(tiles: Sequence[Tile]) -> list[dict[str, object]]:
    rows = []
    for tile in sorted(tiles, key=lambda item: (item.dataset, item.source_group, item.tile)):
        row: dict[str, object] = {
            "dataset": tile.dataset,
            "source_group": tile.source_group,
            "tile": tile.tile,
            "image": str(tile.image),
            "mask": str(tile.mask),
            "total_pixels": tile.total_pixels,
        }
        for raw_id in RAW_IDS:
            row[f"id_{raw_id}_pixels"] = tile.raw_pixels.get(raw_id, 0)
        for target, raw_ids in MERGED_TARGETS.items():
            row[f"{target}_pixels"] = tile.merged_pixels(raw_ids)
        rows.append(row)
    return rows


def per_dataset_class_statistics(
    tiles: Sequence[Tile],
    classes: Mapping[int, tuple[str, str]],
    dataset_paths: Mapping[str, Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_dataset: dict[str, list[Tile]] = defaultdict(list)
    for tile in tiles:
        by_dataset[tile.dataset].append(tile)
    class_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    for dataset_name, members in sorted(by_dataset.items()):
        total_pixels = sum(tile.total_pixels for tile in members)
        total_rows.append({
            "dataset": dataset_name,
            "dataset_path": str(dataset_paths[dataset_name].resolve()),
            "image_count": len(members),
            "total_pixels": total_pixels,
            "valid_pixels": total_pixels,
            "ignored_pixels": 0,
            "class_count": len(classes),
        })
        for class_id, (class_code, class_name) in sorted(classes.items()):
            pixel_count = sum(tile.raw_pixels.get(class_id, 0) for tile in members)
            class_rows.append({
                "dataset": dataset_name,
                "dataset_path": str(dataset_paths[dataset_name].resolve()),
                "class_id": class_id,
                "class_code": class_code,
                "class_name": class_name,
                "is_ignore": False,
                "pixel_count": pixel_count,
                "pixel_ratio_total": pixel_count / total_pixels if total_pixels else 0.0,
                "pixel_ratio_valid": pixel_count / total_pixels if total_pixels else 0.0,
                "image_count_with_class": sum(tile.raw_pixels.get(class_id, 0) > 0 for tile in members),
                "image_count": len(members),
                "total_pixels": total_pixels,
                "valid_pixels": total_pixels,
            })
    return class_rows, total_rows


def scan_v2_class_statistics(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(root).resolve()
    metadata_path = root / "metadata.json"
    manifest_path = root / "manifest.csv"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Missing dataset_v2_3class metadata or manifest under {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = {int(class_id): str(name) for name, class_id in metadata["classes"].items()}
    ignore_id = int(metadata["ignore_index"])
    if ignore_id in classes:
        raise ValueError(f"Ignore ID {ignore_id} overlaps a class ID in {metadata_path}")
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest_rows = list(csv.DictReader(stream))
    if not manifest_rows or "image" not in manifest_rows[0]:
        raise ValueError(f"Dataset manifest must contain an image column: {manifest_path}")

    allowed_ids = {*classes, ignore_id}
    pixels: Counter[int] = Counter()
    image_occurrences: Counter[int] = Counter()
    for row in manifest_rows:
        image_name = row["image"]
        image_path = root / "images" / image_name
        mask_path = root / "masks" / image_name
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"Missing dataset_v2_3class image/mask pair: {image_path}, {mask_path}")
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image)
        if mask.ndim != 2:
            raise ValueError(f"Expected single-channel class-ID mask: {mask_path}")
        with Image.open(image_path) as image:
            if mask.shape != (image.height, image.width):
                raise ValueError(f"Image/mask dimensions differ: {image_path} and {mask_path}")
        ids, counts = np.unique(mask, return_counts=True)
        unknown = sorted(int(value) for value in ids if int(value) not in allowed_ids)
        if unknown:
            raise ValueError(f"Unknown class ID(s) {unknown} in {mask_path}")
        for class_id, count in zip(ids, counts, strict=True):
            pixels[int(class_id)] += int(count)
            image_occurrences[int(class_id)] += 1

    total_pixels = sum(pixels.values())
    ignored_pixels = pixels[ignore_id]
    valid_pixels = total_pixels - ignored_pixels
    dataset_name = root.name
    class_rows: list[dict[str, object]] = []
    labels = [*sorted(classes.items()), (ignore_id, "ignore")]
    for class_id, class_name in labels:
        pixel_count = pixels[class_id]
        class_rows.append({
            "dataset": dataset_name,
            "dataset_path": str(root),
            "class_id": class_id,
            "class_code": class_name,
            "class_name": class_name,
            "is_ignore": class_id == ignore_id,
            "pixel_count": pixel_count,
            "pixel_ratio_total": pixel_count / total_pixels if total_pixels else 0.0,
            "pixel_ratio_valid": (
                None if class_id == ignore_id
                else pixel_count / valid_pixels if valid_pixels else 0.0
            ),
            "image_count_with_class": image_occurrences[class_id],
            "image_count": len(manifest_rows),
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
        })
    total_row = {
        "dataset": dataset_name,
        "dataset_path": str(root),
        "image_count": len(manifest_rows),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "ignored_pixels": ignored_pixels,
        "class_count": len(classes),
    }
    return class_rows, total_row


def _write_class_statistics_document(
    path: Path,
    class_rows: Sequence[Mapping[str, object]],
    total_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# Dataset class pixel statistics",
        "",
        "Ratios use all mask pixels for `pixel_ratio_total`. "
        "`pixel_ratio_valid` excludes ignore pixels; it is blank for the ignore label.",
        "",
        "## Dataset totals",
        "",
        "| Dataset | Images | Total pixels | Valid pixels | Ignored pixels |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in total_rows:
        lines.append(
            f"| {row['dataset']} | {int(row['image_count']):,} | "
            f"{int(row['total_pixels']):,} | {int(row['valid_pixels']):,} | "
            f"{int(row['ignored_pixels']):,} |"
        )
    for total in total_rows:
        dataset_name = str(total["dataset"])
        lines.extend([
            "",
            f"## {dataset_name}",
            "",
            "| ID | Code | Class | Ignore | Pixels | Total ratio | Valid ratio | Images with class |",
            "|---:|---|---|:---:|---:|---:|---:|---:|",
        ])
        for row in class_rows:
            if row["dataset"] != dataset_name:
                continue
            valid_ratio = row["pixel_ratio_valid"]
            lines.append(
                f"| {row['class_id']} | {row['class_code']} | {row['class_name']} | "
                f"{'yes' if row['is_ignore'] else 'no'} | {int(row['pixel_count']):,} | "
                f"{float(row['pixel_ratio_total']):.8%} | "
                f"{'' if valid_ratio is None else f'{float(valid_ratio):.8%}'} | "
                f"{int(row['image_count_with_class']):,} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_merged_statistics_document(
    path: Path,
    dataset_name: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        f"# {dataset_name} merged-class pixel statistics",
        "",
        "Merged targets follow the raw-label mapping used by the deterioration analysis:",
        "",
        "- `shrinkage_craquelure`: raw IDs 3 + 4",
        "- `scratch_crack`: raw IDs 11 + 1",
        "- `abrasion_loss_rodent_bite`: raw IDs 16 + 2 + 36",
        "",
        "| Merged class | Raw IDs | Pixels | Total ratio | Tiles | Source groups |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_name']} | {row['raw_ids']} | "
            f"{int(row['pixel_count']):,} | {float(row['pixel_ratio_total']):.8%} | "
            f"{int(row['tile_count']):,} | {int(row['source_group_count']):,} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validation_candidates(primary_tiles: Sequence[Tile], validation_groups: int) -> list[dict[str, object]]:
    groups = sorted({tile.source_group for tile in primary_tiles})
    if validation_groups <= 0 or validation_groups >= len(groups):
        raise ValueError(f"validation_groups must be between 1 and {len(groups) - 1}")
    total_pixels = {target: sum(tile.merged_pixels(raw_ids) for tile in primary_tiles) for target, raw_ids in MERGED_TARGETS.items()}
    total_tiles = {target: sum(tile.merged_pixels(raw_ids) > 0 for tile in primary_tiles) for target, raw_ids in MERGED_TARGETS.items()}
    expected_tile_ratio = validation_groups / len(groups)
    candidates = []
    for selected in itertools.combinations(groups, validation_groups):
        selected_set = set(selected)
        members = [tile for tile in primary_tiles if tile.source_group in selected_set]
        pixel_coverage = {
            target: (sum(tile.merged_pixels(raw_ids) for tile in members) / total_pixels[target] if total_pixels[target] else 0.0)
            for target, raw_ids in MERGED_TARGETS.items()
        }
        tile_coverage = {
            target: (sum(tile.merged_pixels(raw_ids) > 0 for tile in members) / total_tiles[target] if total_tiles[target] else 0.0)
            for target, raw_ids in MERGED_TARGETS.items()
        }
        missing = sum(sum(tile.merged_pixels(raw_ids) for tile in members) == 0 for raw_ids in MERGED_TARGETS.values())
        pixel_range = max(pixel_coverage.values()) - min(pixel_coverage.values())
        tile_range = max(tile_coverage.values()) - min(tile_coverage.values())
        validation_tile_ratio = len(members) / len(primary_tiles)
        deviation = abs(validation_tile_ratio - expected_tile_ratio)
        candidates.append({
            "source_groups": ";".join(selected),
            "missing_target_count": missing,
            **{f"{target}_pixel_coverage": pixel_coverage[target] for target in MERGED_TARGETS},
            "pixel_coverage_range": pixel_range,
            **{f"{target}_tile_coverage": tile_coverage[target] for target in MERGED_TARGETS},
            "tile_coverage_range": tile_range,
            "validation_tile_count": len(members),
            "validation_tile_ratio": validation_tile_ratio,
            "validation_tile_ratio_deviation": deviation,
            "selection_score": [missing, pixel_range, tile_range, deviation, list(selected)],
        })
    candidates.sort(key=lambda row: (row["missing_target_count"], row["pixel_coverage_range"], row["tile_coverage_range"], row["validation_tile_ratio_deviation"], row["source_groups"]))
    return candidates


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_hash(tiles: Sequence[Tile]) -> str:
    entries = []
    for tile in sorted(tiles, key=lambda item: (item.dataset, item.source_group, item.tile)):
        entries.append({
            "dataset": tile.dataset,
            "source_group": tile.source_group,
            "tile": tile.tile,
            "image_sha256": _sha256(tile.image),
            "mask_sha256": _sha256(tile.mask),
        })
    return _canonical_hash(entries)


def _normalized_source_identity(value: str) -> str:
    return "".join(character for character in Path(value).stem.casefold() if character.isalnum())


def _image_fingerprint(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = image.convert("L")
        difference_grid = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        thumbnail = np.asarray(gray.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32)
    differences = difference_grid[:, 1:] > difference_grid[:, :-1]
    difference_hash = 0
    for bit in differences.ravel():
        difference_hash = (difference_hash << 1) | int(bit)
    return {
        "file_sha256": _sha256(path),
        "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "difference_hash": difference_hash,
        "thumbnail": thumbnail,
    }


def cross_dataset_leakage_audit(
    primary_tiles: Sequence[Tile],
    secondary_tiles: Sequence[Tile],
    *,
    maximum_dhash_distance: int = 4,
    maximum_thumbnail_rmse: float = 3.0,
) -> dict[str, object]:
    """Find exact identities and conservative near-duplicates across datasets."""

    primary = [(tile, _image_fingerprint(tile.image)) for tile in primary_tiles]
    secondary = [(tile, _image_fingerprint(tile.image)) for tile in secondary_tiles]
    file_index: dict[str, list[Tile]] = defaultdict(list)
    pixel_index: dict[str, list[Tile]] = defaultdict(list)
    identity_index: dict[str, list[Tile]] = defaultdict(list)
    for tile, fingerprint in primary:
        file_index[str(fingerprint["file_sha256"])].append(tile)
        pixel_index[str(fingerprint["pixel_sha256"])].append(tile)
        identity_index[_normalized_source_identity(tile.source_group)].append(tile)
        identity_index[_normalized_source_identity(tile.tile)].append(tile)

    matches: list[dict[str, object]] = []
    quarantined_groups: set[str] = set()
    for secondary_tile, secondary_fingerprint in secondary:
        candidates: dict[tuple[str, str], tuple[Tile, set[str]]] = {}

        def add_candidate(primary_tile: Tile, reason: str) -> None:
            key = (primary_tile.source_group, primary_tile.tile)
            if key not in candidates:
                candidates[key] = (primary_tile, set())
            candidates[key][1].add(reason)

        for primary_tile in file_index[str(secondary_fingerprint["file_sha256"])]:
            add_candidate(primary_tile, "exact_file_sha256")
        for primary_tile in pixel_index[str(secondary_fingerprint["pixel_sha256"])]:
            add_candidate(primary_tile, "exact_rgb_pixels")
        for identity in {
            _normalized_source_identity(secondary_tile.source_group),
            _normalized_source_identity(secondary_tile.tile),
        }:
            for primary_tile in identity_index.get(identity, ()):
                add_candidate(primary_tile, "source_identity")
        for primary_tile, primary_fingerprint in primary:
            distance = (int(secondary_fingerprint["difference_hash"]) ^ int(primary_fingerprint["difference_hash"])).bit_count()
            if distance > maximum_dhash_distance:
                continue
            delta = np.asarray(secondary_fingerprint["thumbnail"]) - np.asarray(primary_fingerprint["thumbnail"])
            rmse = float(np.sqrt(np.mean(np.square(delta))))
            if rmse <= maximum_thumbnail_rmse:
                add_candidate(primary_tile, f"near_duplicate:dhash={distance},thumbnail_rmse={rmse:.3f}")
        for primary_tile, reasons in candidates.values():
            quarantined_groups.add(secondary_tile.source_group)
            matches.append({
                "primary_source_group": primary_tile.source_group,
                "primary_tile": primary_tile.tile,
                "dataset114_source_group": secondary_tile.source_group,
                "dataset114_tile": secondary_tile.tile,
                "reasons": sorted(reasons),
            })
    matches.sort(key=lambda row: (str(row["dataset114_source_group"]), str(row["dataset114_tile"]), str(row["primary_source_group"]), str(row["primary_tile"])))
    return {
        "schema_version": 1,
        "checked_primary_tiles": len(primary_tiles),
        "checked_dataset114_tiles": len(secondary_tiles),
        "criteria": {
            "exact_file_sha256": True,
            "exact_rgb_pixels": True,
            "normalized_source_identity": True,
            "maximum_dhash_distance": maximum_dhash_distance,
            "maximum_thumbnail_rmse": maximum_thumbnail_rmse,
        },
        "suspected_cross_dataset_matches": matches,
        "quarantined_dataset114_source_groups": sorted(quarantined_groups),
        "suspects_found": bool(matches),
        "passed_after_quarantine": True,
    }


def selected_validation(primary_tiles: Sequence[Tile], candidates: Sequence[Mapping[str, object]], class_sha256: str, dataset_sha256: str) -> dict[str, object]:
    selected = candidates[0]
    validation_groups = str(selected["source_groups"]).split(";")
    validation_set = set(validation_groups)
    validation_tiles = sorted(tile.tile for tile in primary_tiles if tile.source_group in validation_set)
    training_tiles = sorted(tile.tile for tile in primary_tiles if tile.source_group not in validation_set)
    training_groups = {tile.source_group for tile in primary_tiles if tile.source_group not in validation_set}
    overlap_groups = sorted(validation_set & training_groups)
    overlap_tiles = sorted(set(validation_tiles) & set(training_tiles))
    split_payload = {
        "validation_source_groups": validation_groups,
        "training_source_groups": sorted(training_groups),
        "validation_tiles": validation_tiles,
        "training_tiles": training_tiles,
    }
    coverage = {
        target: {
            "pixel": selected[f"{target}_pixel_coverage"],
            "tile": selected[f"{target}_tile_coverage"],
        }
        for target in MERGED_TARGETS
    }
    return {
        "schema_version": 1,
        "seed": SEED,
        **split_payload,
        "coverage": coverage,
        "selection_score": selected["selection_score"],
        "dataset_sha256": dataset_sha256,
        "class_sha256": class_sha256,
        "split_sha256": _canonical_hash(split_payload),
        "leakage_audit": {
            "source_group_overlap": overlap_groups,
            "tile_overlap": overlap_tiles,
            "passed": not overlap_groups and not overlap_tiles,
        },
    }


def _group_profiles(tiles: Sequence[Tile]) -> dict[str, dict[str, object]]:
    members: dict[str, list[Tile]] = defaultdict(list)
    for tile in tiles:
        members[tile.source_group].append(tile)
    return {
        group: {
            "tile_count": len(group_tiles),
            "pixels": {
                target: sum(tile.merged_pixels(raw_ids) for tile in group_tiles)
                for target, raw_ids in MERGED_TARGETS.items()
            },
            "positive_tiles": {
                target: sum(tile.merged_pixels(raw_ids) > 0 for tile in group_tiles)
                for target, raw_ids in MERGED_TARGETS.items()
            },
        }
        for group, group_tiles in sorted(members.items())
    }


def _partition_score(
    validation_pixels: Mapping[str, int],
    validation_positive_tiles: Mapping[str, int],
    validation_tile_count: int,
    *,
    total_pixels: Mapping[str, int],
    total_positive_tiles: Mapping[str, int],
    total_tile_count: int,
    target_tile_ratio: float,
) -> tuple[int, float, float, float]:
    missing = sum(
        validation_pixels[target] == 0 or validation_pixels[target] == total_pixels[target]
        for target in MERGED_TARGETS
    )
    pixel_coverage = [
        validation_pixels[target] / total_pixels[target] if total_pixels[target] else 0.0
        for target in MERGED_TARGETS
    ]
    tile_coverage = [
        validation_positive_tiles[target] / total_positive_tiles[target]
        if total_positive_tiles[target] else 0.0
        for target in MERGED_TARGETS
    ]
    return (
        missing,
        max(pixel_coverage) - min(pixel_coverage),
        max(tile_coverage) - min(tile_coverage),
        abs(validation_tile_count / total_tile_count - target_tile_ratio),
    )


def dataset114_training_only(
    tiles: Sequence[Tile],
    cross_dataset_audit: Mapping[str, object],
) -> dict[str, object]:
    """Quarantine suspected overlaps and keep Dataset114 out of validation."""

    quarantined = set(cross_dataset_audit["quarantined_dataset114_source_groups"])
    training_groups = sorted({tile.source_group for tile in tiles if tile.source_group not in quarantined})
    training_tiles = sorted(tile.tile for tile in tiles if tile.source_group not in quarantined)
    excluded_tiles = sorted(tile.tile for tile in tiles if tile.source_group in quarantined)
    split_payload = {
        "training_source_groups": training_groups,
        "validation_source_groups": [],
        "quarantined_source_groups": sorted(quarantined),
        "training_tiles": training_tiles,
        "validation_tiles": [],
        "quarantined_tiles": excluded_tiles,
    }
    return {
        "schema_version": 2,
        "strategy": "training_only_after_cross_dataset_exact_and_near_duplicate_audit",
        **split_payload,
        "split_sha256": _canonical_hash(split_payload),
        "cross_dataset_leakage_audit": dict(cross_dataset_audit),
        "leakage_audit": {
            "dataset114_validation_tiles": 0,
            "quarantined_tiles_excluded": len(excluded_tiles),
            "passed": not split_payload["validation_tiles"],
        },
    }


def primary_only_split(
    primary_tiles: Sequence[Tile],
    primary: Mapping[str, object],
    *,
    dataset_sha256: str,
    class_sha256: str,
) -> dict[str, object]:
    """Build the fixed two-group validation manifest without Dataset114."""

    validation_groups = set(primary["validation_source_groups"])

    def record(tile: Tile) -> dict[str, str]:
        return {
            "dataset": tile.dataset,
            "source_group": tile.source_group,
            "tile": tile.tile,
            "image": str(tile.image),
            "mask": str(tile.mask),
            "image_sha256": _sha256(tile.image),
            "mask_sha256": _sha256(tile.mask),
        }

    training = sorted(
        [record(tile) for tile in primary_tiles if tile.source_group not in validation_groups],
        key=lambda row: (row["dataset"], row["source_group"], row["tile"]),
    )
    validation = sorted(
        [record(tile) for tile in primary_tiles if tile.source_group in validation_groups],
        key=lambda row: (row["dataset"], row["source_group"], row["tile"]),
    )
    payload = {"training": training, "validation": validation}
    membership = {
        partition: [
            {key: row[key] for key in ("dataset", "source_group", "tile")}
            for row in rows
        ]
        for partition, rows in payload.items()
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
        for partition in ("training", "validation")
        for row in payload[partition]
    ]
    train_groups = {(row["dataset"], row["source_group"]) for row in training}
    validation_group_keys = {
        (row["dataset"], row["source_group"]) for row in validation
    }
    train_tiles = {(row["dataset"], row["tile"]) for row in training}
    validation_tiles = {(row["dataset"], row["tile"]) for row in validation}
    return {
        "schema_version": 2,
        "seed": SEED,
        "dataset_sha256": dataset_sha256,
        "class_sha256": class_sha256,
        **payload,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash(adopted_content),
        "dataset114_policy": "excluded_from_training_and_validation",
        "leakage_audit": {
            "source_group_overlap": sorted(train_groups & validation_group_keys),
            "tile_overlap": sorted(train_tiles & validation_tiles),
            "dataset114_tiles": [],
            "passed": not (
                train_groups & validation_group_keys or train_tiles & validation_tiles
            ),
        },
    }


def combined_split(
    primary_tiles: Sequence[Tile],
    secondary_tiles: Sequence[Tile],
    primary: Mapping[str, object],
    secondary: Mapping[str, object],
    *,
    dataset_sha256: str,
    class_sha256: str,
) -> dict[str, object]:
    """Build a primary-only validation split with audited Dataset114 training data."""

    primary_validation = set(primary["validation_source_groups"])
    allowed_secondary_groups = set(secondary["training_source_groups"])

    def record(tile: Tile) -> dict[str, str]:
        return {
            "dataset": tile.dataset,
            "source_group": tile.source_group,
            "tile": tile.tile,
            "image": str(tile.image),
            "mask": str(tile.mask),
            "image_sha256": _sha256(tile.image),
            "mask_sha256": _sha256(tile.mask),
        }

    training = sorted(
        [record(tile) for tile in primary_tiles if tile.source_group not in primary_validation]
        + [record(tile) for tile in secondary_tiles if tile.source_group in allowed_secondary_groups],
        key=lambda row: (row["dataset"], row["source_group"], row["tile"]),
    )
    validation = sorted(
        [record(tile) for tile in primary_tiles if tile.source_group in primary_validation],
        key=lambda row: (row["dataset"], row["source_group"], row["tile"]),
    )
    payload = {"training": training, "validation": validation}
    membership = {
        partition: [
            {key: row[key] for key in ("dataset", "source_group", "tile")}
            for row in rows
        ]
        for partition, rows in payload.items()
    }
    adopted_content = [
        {key: row[key] for key in ("dataset", "source_group", "tile", "image_sha256", "mask_sha256")}
        for partition in ("training", "validation")
        for row in payload[partition]
    ]
    train_groups = {(row["dataset"], row["source_group"]) for row in training}
    validation_groups = {(row["dataset"], row["source_group"]) for row in validation}
    train_tiles = {(row["dataset"], row["tile"]) for row in training}
    validation_tiles = {(row["dataset"], row["tile"]) for row in validation}
    dataset114_validation = [row for row in validation if row["dataset"] == "dataset114"]
    quarantined_groups = set(secondary["quarantined_source_groups"])
    quarantined_present = sorted(
        (row["dataset"], row["source_group"])
        for row in training
        if row["dataset"] == "dataset114" and row["source_group"] in quarantined_groups
    )
    passed = not (
        train_groups & validation_groups
        or train_tiles & validation_tiles
        or dataset114_validation
        or quarantined_present
    )
    return {
        "schema_version": 2,
        "seed": SEED,
        "dataset_sha256": dataset_sha256,
        "class_sha256": class_sha256,
        **payload,
        "split_sha256": _canonical_hash(membership),
        "adopted_dataset_sha256": _canonical_hash(adopted_content),
        "dataset114_policy": "training_only_after_cross_dataset_audit",
        "cross_dataset_leakage_audit": secondary["cross_dataset_leakage_audit"],
        "leakage_audit": {
            "source_group_overlap": sorted(train_groups & validation_groups),
            "tile_overlap": sorted(train_tiles & validation_tiles),
            "dataset114_validation_tiles": [row["tile"] for row in dataset114_validation],
            "quarantined_dataset114_groups_present": quarantined_present,
            "passed": passed,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def analyze_datasets(
    dataset: Path,
    dataset114: Path,
    dataset_v2_3class: Path,
    output: Path,
    *,
    validation_groups: int = 2,
) -> dict[str, object]:
    dataset = Path(dataset)
    dataset114 = Path(dataset114)
    dataset_v2_3class = Path(dataset_v2_3class)
    output = Path(output)
    classes_path = dataset / "classes.txt"
    classes = load_classes(classes_path)
    known_ids = set(classes)
    primary_tiles = scan_primary(dataset, known_ids)
    secondary_tiles = scan_dataset114(dataset114, known_ids)
    tiles = primary_tiles + secondary_tiles
    raw_rows = raw_statistics(tiles, classes)
    merged_rows = merged_statistics(tiles)
    dataset_rows, group_rows = grouped_statistics(tiles)
    tile_rows = per_tile_statistics(tiles)
    class_rows, total_rows = per_dataset_class_statistics(
        tiles,
        classes,
        {dataset.name: dataset, dataset114.name: dataset114},
    )
    v2_class_rows, v2_total_row = scan_v2_class_statistics(dataset_v2_3class)
    class_rows.extend(v2_class_rows)
    total_rows.append(v2_total_row)
    candidate_rows = validation_candidates(primary_tiles, validation_groups)
    class_sha256 = _canonical_hash({
        "classes_file_sha256": _sha256(classes_path),
        "dataset114_id_2_override": "Loss",
        "merged_targets": MERGED_TARGETS,
    })
    dataset_sha256 = _dataset_hash(tiles)
    primary_dataset_sha256 = _dataset_hash(primary_tiles)
    selected = selected_validation(primary_tiles, candidate_rows, class_sha256, dataset_sha256)
    cross_dataset_audit = cross_dataset_leakage_audit(primary_tiles, secondary_tiles)
    secondary_split = dataset114_training_only(secondary_tiles, cross_dataset_audit)
    adopted_split = combined_split(
        primary_tiles,
        secondary_tiles,
        selected,
        secondary_split,
        dataset_sha256=dataset_sha256,
        class_sha256=class_sha256,
    )
    primary_only = primary_only_split(
        primary_tiles,
        selected,
        dataset_sha256=primary_dataset_sha256,
        class_sha256=class_sha256,
    )
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "raw_class_statistics.csv": raw_rows,
        "per_dataset_statistics.csv": dataset_rows,
        "per_source_group_statistics.csv": group_rows,
        "per_tile_statistics.csv": tile_rows,
        "validation_candidates.csv": candidate_rows,
    }
    for name, rows in files.items():
        _write_csv(output / name, rows)
    merge_tiles_by_dataset = {
        dataset.name: primary_tiles,
        dataset114.name: secondary_tiles,
    }
    for total_row in total_rows:
        dataset_name = str(total_row["dataset"])
        dataset_output = output / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        dataset_class_rows = [row for row in class_rows if row["dataset"] == dataset_name]
        _write_csv(dataset_output / "class_pixel_statistics.csv", dataset_class_rows)
        _write_csv(dataset_output / "dataset_pixel_totals.csv", [total_row])
        _write_class_statistics_document(
            dataset_output / "class_pixel_statistics.md",
            dataset_class_rows,
            [total_row],
        )
        if dataset_name in merge_tiles_by_dataset:
            dataset_merged_rows = []
            for row in merged_statistics(merge_tiles_by_dataset[dataset_name]):
                dataset_merged_rows.append({
                    "dataset": dataset_name,
                    "dataset_path": total_row["dataset_path"],
                    **row,
                    "pixel_ratio_total": (
                        int(row["pixel_count"]) / int(total_row["total_pixels"])
                        if int(total_row["total_pixels"]) else 0.0
                    ),
                })
            _write_csv(
                dataset_output / "merged_class_pixel_statistics.csv",
                dataset_merged_rows,
            )
            _write_merged_statistics_document(
                dataset_output / "merged_class_pixel_statistics.md",
                dataset_name,
                dataset_merged_rows,
            )
    (output / "selected_validation.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "dataset114_split.json").write_text(json.dumps(secondary_split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "combined_training_split.json").write_text(json.dumps(adopted_split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "primary_training_split.json").write_text(
        json.dumps(primary_only, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    statistics = {
        "schema_version": 1,
        "datasets": [
            str(dataset.resolve()),
            str(dataset114.resolve()),
            str(dataset_v2_3class.resolve()),
        ],
        "tile_count": len(tiles),
        "source_group_count": len({(tile.dataset, tile.source_group) for tile in tiles}),
        "total_pixels": sum(tile.total_pixels for tile in tiles),
        "raw_classes": raw_rows,
        "merged_targets": merged_rows,
        "per_dataset": dataset_rows,
        "class_pixel_statistics": class_rows,
        "dataset_pixel_totals": total_rows,
        "hashes": {
            "dataset_sha256": dataset_sha256,
            "class_sha256": class_sha256,
            "primary_split_sha256": selected["split_sha256"],
            "dataset114_split_sha256": secondary_split["split_sha256"],
            "combined_split_sha256": adopted_split["split_sha256"],
            "primary_training_split_sha256": primary_only["split_sha256"],
        },
        "selected_validation": selected,
        "dataset114_split": secondary_split,
        "combined_split_summary": {
            "training_tiles": len(adopted_split["training"]),
            "validation_tiles": len(adopted_split["validation"]),
            "leakage_audit": adopted_split["leakage_audit"],
        },
        "primary_training_split_summary": {
            "training_tiles": len(primary_only["training"]),
            "validation_tiles": len(primary_only["validation"]),
            "leakage_audit": primary_only["leakage_audit"],
        },
    }
    (output / "statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset114", type=Path, required=True)
    parser.add_argument("--dataset-v2-3class", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-groups", type=int, default=2)
    args = parser.parse_args()
    result = analyze_datasets(
        args.dataset,
        args.dataset114,
        args.dataset_v2_3class,
        args.output,
        validation_groups=args.validation_groups,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "tile_count": result["tile_count"],
        "selected_validation_groups": result["selected_validation"]["validation_source_groups"],
        "dataset_pixel_totals": result["dataset_pixel_totals"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
