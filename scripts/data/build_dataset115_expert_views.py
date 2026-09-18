"""Build three binary expert views from reviewed Dataset115 D-code masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPERT_SOURCE_CODES: dict[str, tuple[str, ...]] = {
    "shrinkage_craquelure": ("D-03", "D-04"),
    "scratch_crack": ("D-01", "D-11"),
    "loss": ("D-02",),
}
KYT_D04_ONLY_GROUPS = frozenset({"KYT-SC-1R-A9-4", "KYT-SC-1R-2LB1-1"})
CORRECTED_CLASS_NAMES = {"D-01": "Crack", "D-02": "Loss", "D-04": "Craquelure"}
MANIFEST_COLUMNS = (
    "expert",
    "temple",
    "source_group",
    "tile",
    "image",
    "mask",
    "source_codes",
    "foreground_pixels",
    "is_positive",
    "image_sha256",
    "mask_sha256",
)


@dataclass(frozen=True)
class SourceTile:
    temple: str
    source_group: str
    tile: str
    image: str
    image_sha256: str
    masks: dict[str, str]


def _source_codes(expert: str, tile: SourceTile) -> tuple[str, ...]:
    if expert == "shrinkage_craquelure" and tile.source_group in KYT_D04_ONLY_GROUPS:
        return ("D-04",)
    return EXPERT_SOURCE_CODES[expert]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes dataset root: {relative}")
    return path


def _read_source_tiles(dataset: Path) -> tuple[list[SourceTile], dict[Path, str]]:
    manifest = dataset / "metadata" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing source manifest: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("source manifest has no rows")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_mask_hashes: dict[Path, str] = {}
    seen_masks: set[tuple[str, str]] = set()
    for row in rows:
        image = row["image"]
        code = row["mask_class_code"]
        key = (image, code)
        if key in seen_masks:
            raise ValueError(f"duplicate source mask record: image={image}, code={code}")
        seen_masks.add(key)
        image_path = _resolve_under(dataset, image)
        mask_path = _resolve_under(dataset, row["mask"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing source file for image={image}, code={code}")
        if _sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"source image checksum mismatch: {image}")
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError(f"source mask checksum mismatch: {row['mask']}")
        source_mask_hashes[mask_path] = row["mask_sha256"]
        grouped[image].append(row)

    tiles: list[SourceTile] = []
    for image, image_rows in sorted(grouped.items()):
        first = image_rows[0]
        stable_fields = ("temple", "image_group", "image_sha256")
        for row in image_rows[1:]:
            if any(row[field] != first[field] for field in stable_fields):
                raise ValueError(f"inconsistent source rows for image: {image}")
        tiles.append(
            SourceTile(
                temple=first["temple"],
                source_group=first["image_group"],
                tile=Path(image).stem,
                image=image,
                image_sha256=first["image_sha256"],
                masks={row["mask_class_code"]: row["mask"] for row in image_rows},
            )
        )
    return tiles, source_mask_hashes


def _binary_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        array = np.asarray(source.convert("L"), dtype=np.uint8)
    if array.shape != expected_shape:
        raise ValueError(f"mask/image shape mismatch: {path} has {array.shape}, expected {expected_shape}")
    values = set(int(value) for value in np.unique(array))
    if not values.issubset({0, 255}):
        raise ValueError(f"source mask is not binary 0/255: {path}, values={sorted(values)}")
    return array == 255


def _correct_classes(dataset: Path) -> str:
    path = dataset / "metadata" / "classes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = payload.get("categories")
    if not isinstance(categories, list):
        raise ValueError("classes.json has no categories list")
    found: set[str] = set()
    for category in categories:
        code = category.get("code")
        if code in CORRECTED_CLASS_NAMES:
            category["name"] = CORRECTED_CLASS_NAMES[code]
            found.add(code)
    missing = sorted(set(CORRECTED_CLASS_NAMES) - found)
    if missing:
        raise ValueError(f"classes.json is missing required categories: {missing}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return _sha256(path)


def _write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _build_staging(dataset: Path, staging: Path, tiles: list[SourceTile]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for expert in EXPERT_SOURCE_CODES:
        rows: list[dict[str, str | int]] = []
        foreground_total = 0
        positive_tiles = 0
        for tile in tiles:
            image_path = _resolve_under(dataset, tile.image)
            with Image.open(image_path) as image:
                width, height = image.size
            union = np.zeros((height, width), dtype=bool)
            source_codes = _source_codes(expert, tile)
            present_codes = tuple(code for code in source_codes if code in tile.masks)
            for code in present_codes:
                union |= _binary_mask(_resolve_under(dataset, tile.masks[code]), union.shape)

            relative_mask = Path(expert) / "masks" / tile.temple / tile.source_group / f"{tile.tile}.png"
            destination = _resolve_under(staging, relative_mask.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(union.astype(np.uint8) * 255).save(destination)
            foreground_pixels = int(union.sum())
            foreground_total += foreground_pixels
            positive_tiles += foreground_pixels > 0
            rows.append(
                {
                    "expert": expert,
                    "temple": tile.temple,
                    "source_group": tile.source_group,
                    "tile": tile.tile,
                    "image": tile.image,
                    "mask": (Path("expert_views") / relative_mask).as_posix(),
                    "source_codes": ";".join(present_codes),
                    "foreground_pixels": foreground_pixels,
                    "is_positive": int(foreground_pixels > 0),
                    "image_sha256": tile.image_sha256,
                    "mask_sha256": _sha256(destination),
                }
            )
        manifest_path = staging / expert / "manifest.csv"
        _write_manifest(manifest_path, rows)
        summaries[expert] = {
            "source_codes": list(EXPERT_SOURCE_CODES[expert]),
            "merge_rule": "pixelwise_or",
            "source_code_overrides": {
                group: ["D-04"]
                for group in sorted(KYT_D04_ONLY_GROUPS)
            }
            if expert == "shrinkage_craquelure"
            else {},
            "tile_count": len(rows),
            "positive_tile_count": positive_tiles,
            "negative_tile_count": len(rows) - positive_tiles,
            "foreground_pixels": foreground_total,
            "manifest": f"expert_views/{expert}/manifest.csv",
            "manifest_sha256": _sha256(manifest_path),
        }
    return summaries


def _validate_staging(dataset: Path, staging: Path, tiles: list[SourceTile]) -> None:
    expected_images = {tile.image: tile for tile in tiles}
    for expert in EXPERT_SOURCE_CODES:
        manifest_path = staging / expert / "manifest.csv"
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(tiles):
            raise ValueError(f"{expert} manifest has {len(rows)} rows; expected {len(tiles)}")
        if [row["image"] for row in rows] != sorted(expected_images):
            raise ValueError(f"{expert} manifest ordering or image membership is invalid")
        for row in rows:
            tile = expected_images[row["image"]]
            image_path = _resolve_under(dataset, tile.image)
            with Image.open(image_path) as image:
                expected_shape = (image.height, image.width)
            relative = Path(row["mask"])
            if not relative.parts or relative.parts[0] != "expert_views":
                raise ValueError(f"non-canonical generated mask path: {relative}")
            staged_mask = _resolve_under(staging, Path(*relative.parts[1:]).as_posix())
            generated = _binary_mask(staged_mask, expected_shape)
            expected = np.zeros(expected_shape, dtype=bool)
            source_codes = _source_codes(expert, tile)
            present_codes = tuple(code for code in source_codes if code in tile.masks)
            for code in present_codes:
                expected |= _binary_mask(_resolve_under(dataset, tile.masks[code]), expected_shape)
            if not np.array_equal(generated, expected):
                raise ValueError(f"generated union mismatch: expert={expert}, image={tile.image}")
            if row["source_codes"] != ";".join(present_codes):
                raise ValueError(f"source code provenance mismatch: expert={expert}, image={tile.image}")
            if int(row["foreground_pixels"]) != int(expected.sum()):
                raise ValueError(f"foreground count mismatch: expert={expert}, image={tile.image}")
            if _sha256(staged_mask) != row["mask_sha256"]:
                raise ValueError(f"generated mask checksum mismatch: {staged_mask}")


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


def build_expert_views(dataset: str | Path) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    tiles, source_mask_hashes = _read_source_tiles(dataset)
    staging = dataset / f".expert_views.staging-{uuid.uuid4().hex}"
    destination = dataset / "expert_views"
    staging.mkdir(parents=False)
    try:
        summaries = _build_staging(dataset, staging, tiles)
        _validate_staging(dataset, staging, tiles)
        changed_sources = [str(path) for path, digest in source_mask_hashes.items() if _sha256(path) != digest]
        if changed_sources:
            raise RuntimeError(f"source masks changed during build: {changed_sources[:3]}")
        _replace_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    classes_hash = _correct_classes(dataset)
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset),
        "source_manifest": "metadata/manifest.csv",
        "source_manifest_sha256": _sha256(dataset / "metadata" / "manifest.csv"),
        "classes": CORRECTED_CLASS_NAMES,
        "classes_sha256": classes_hash,
        "encoding": {"background": 0, "foreground": 255, "dtype": "uint8"},
        "experts": summaries,
    }
    metadata_path = dataset / "metadata" / "expert_views.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset115_filtered"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = build_expert_views(args.dataset)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
