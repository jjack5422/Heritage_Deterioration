"""Replace active KYT D-04 masks with 7 px variants and preserve coarse sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset115_filtered"
DEFAULT_BACKUP = (
    ROOT
    / "sam3_adapter"
    / "runs"
    / "2026-09-17_craquelure-sam3-adapter-1008_no-kyt-2lb1_epochs60_seed42"
    / "info"
    / "legacy"
    / "kyt_original_coarse_d04"
)
A9_GROUP = "KYT-SC-1R-A9-4"
LB_GROUP = "KYT-SC-1R-2LB1-1"
GROUPS = (A9_GROUP, LB_GROUP)
EXPECTED_D04 = {A9_GROUP: 80, LB_GROUP: 45}
EXPECTED_TILES = {A9_GROUP: 80, LB_GROUP: 62}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binary(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as source:
        array = np.asarray(source.convert("L"), dtype=np.uint8)
    if shape is not None and array.shape != shape:
        raise ValueError(f"mask shape mismatch: {path}: {array.shape} != {shape}")
    values = set(int(value) for value in np.unique(array))
    if not values.issubset({0, 255}):
        raise ValueError(f"mask is not binary: {path}: {sorted(values)}")
    return array == 255


def _save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _zhang_suen(mask: np.ndarray) -> tuple[np.ndarray, int]:
    image = mask.astype(np.uint8).copy()
    completed_iterations = 0
    while True:
        changed = False
        for subiteration in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )
            common = (image == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            if subiteration == 0:
                remove = common & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove = common & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if remove.any():
                image[remove] = 0
                changed = True
        completed_iterations += 1
        if not changed:
            return image.astype(bool), completed_iterations


def _dilate_to_width(skeleton: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or width % 2 == 0:
        raise ValueError(f"width must be a positive odd integer: {width}")
    radius = (width - 1) // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    return ndimage.binary_dilation(skeleton, structure=disk)


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes dataset root: {relative}")
    return path


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if not rows or not fields:
        raise ValueError(f"empty source manifest: {path}")
    return fields, rows


def _write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_kyt_masks(dataset: Path, backup: Path, lb_variants: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    backup = backup.resolve()
    lb_variants = lb_variants.resolve()
    manifest_path = dataset / "metadata" / "manifest.csv"
    fields, rows = _read_manifest(manifest_path)
    if backup.exists():
        raise FileExistsError(f"refusing to replace masks because backup already exists: {backup}")
    if lb_variants.exists():
        raise FileExistsError(f"refusing to overwrite width variants: {lb_variants}")

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_image[row["image"]].append(row)
    group_images = {
        group: sorted(image for image, image_rows in rows_by_image.items() if image_rows[0]["image_group"] == group)
        for group in GROUPS
    }
    for group, images in group_images.items():
        if len(images) != EXPECTED_TILES[group]:
            raise ValueError(f"{group} tile count drifted: {len(images)} != {EXPECTED_TILES[group]}")

    d04_rows = {
        (row["image_group"], row["image"]): row
        for row in rows
        if row["image_group"] in GROUPS and row["mask_class_code"] == "D-04"
    }
    for group in GROUPS:
        count = sum(key[0] == group for key in d04_rows)
        if count != EXPECTED_D04[group]:
            raise ValueError(f"{group} D-04 count drifted: {count} != {EXPECTED_D04[group]}")

    workspace = dataset.parent / f".kyt-d04-{uuid.uuid4().hex}"
    staged_backup = workspace / "backup"
    staged_active = workspace / "active_7px"
    staged_variants = workspace / "lb_variants"
    measurements: list[dict[str, str | int]] = []
    original_manifest = manifest_path.read_bytes()
    workspace.mkdir()
    try:
        backup_records: list[dict[str, str | int]] = []
        generated: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for group in GROUPS:
            for image_relative in group_images[group]:
                image_path = _resolve_under(dataset, image_relative)
                with Image.open(image_path) as image_file:
                    shape = (image_file.height, image_file.width)
                tile = Path(image_relative).stem
                source_row = d04_rows.get((group, image_relative))
                if source_row is None:
                    original = np.zeros(shape, dtype=bool)
                    skeleton = original
                    iterations = 0
                else:
                    source_path = _resolve_under(dataset, source_row["mask"])
                    original = _binary(source_path, shape)
                    skeleton, iterations = _zhang_suen(original)
                    backup_path = staged_backup / group / tile / "D-04.png"
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, backup_path)
                    backup_records.append(
                        {
                            "group": group,
                            "tile": tile,
                            "source_mask": source_row["mask"],
                            "backup_mask": (Path(group) / tile / "D-04.png").as_posix(),
                            "foreground_pixels": int(original.sum()),
                            "sha256": _sha256(source_path),
                        }
                    )
                widths = {width: _dilate_to_width(skeleton, width) for width in (5, 7)}
                generated[(group, image_relative)] = widths
                if group == A9_GROUP:
                    existing = (
                        ROOT
                        / "outputs"
                        / "KYT-SC-1R-A9-4_d04_thinning_preview"
                        / "masks_7px"
                        / f"{tile}.png"
                    )
                    if not np.array_equal(widths[7], _binary(existing, shape)):
                        raise ValueError(f"regenerated A9 7 px mask differs from approved output: {tile}")
                else:
                    for width, mask in widths.items():
                        _save_binary(staged_variants / f"masks_{width}px" / f"{tile}.png", mask)
                    measurements.append(
                        {
                            "tile": tile,
                            "had_source_d04": int(source_row is not None),
                            "original_pixels": int(original.sum()),
                            "skeleton_pixels": int(skeleton.sum()),
                            "skeleton_iterations": iterations,
                            "pixels_5px": int(widths[5].sum()),
                            "pixels_7px": int(widths[7].sum()),
                        }
                    )
                if source_row is not None:
                    _save_binary(staged_active / group / tile / "D-04.png", widths[7])

        (staged_backup / "manifest.json").write_text(
            json.dumps(
                {
                    "method": "Zhang-Suen skeletonization followed by Euclidean disk dilation",
                    "active_width_px": 7,
                    "records": backup_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staged_variants.mkdir(parents=True, exist_ok=True)
        with (staged_variants / "measurements.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
            writer.writeheader()
            writer.writerows(measurements)
        summary = {
            "group": LB_GROUP,
            "method": "Zhang-Suen skeletonization followed by Euclidean disk dilation",
            "tile_count": len(measurements),
            "source_d04_tiles": sum(int(row["had_source_d04"]) for row in measurements),
            "empty_tiles": sum(not int(row["had_source_d04"]) for row in measurements),
            "widths_px": [5, 7],
            "foreground_pixels": {
                f"{width}px": sum(int(row[f"pixels_{width}px"]) for row in measurements)
                for width in (5, 7)
            },
        }
        (staged_variants / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        backup.parent.mkdir(parents=True, exist_ok=True)
        staged_backup.replace(backup)
        lb_variants.parent.mkdir(parents=True, exist_ok=True)
        staged_variants.replace(lb_variants)

        try:
            for (group, image_relative), source_row in d04_rows.items():
                tile = Path(image_relative).stem
                source_path = _resolve_under(dataset, source_row["mask"])
                staged_path = staged_active / group / tile / "D-04.png"
                os.replace(staged_path, source_path)
                source_row["mask_foreground_pixels"] = str(int(generated[(group, image_relative)][7].sum()))
                source_row["mask_sha256"] = _sha256(source_path)

            for group in GROUPS:
                for image_relative in group_images[group]:
                    image_rows = rows_by_image[image_relative]
                    masks = []
                    for row in image_rows:
                        if row["mask_class_code"] == "D-04":
                            mask = generated[(group, image_relative)][7]
                        else:
                            mask = _binary(_resolve_under(dataset, row["mask"]))
                        masks.append(mask)
                    stack = np.stack(masks)
                    union_pixels = str(int(np.any(stack, axis=0).sum()))
                    overlap_pixels = str(int((stack.sum(axis=0) > 1).sum()))
                    for row in image_rows:
                        row["classes_present"] = str(len(image_rows))
                        row["union_foreground_pixels"] = union_pixels
                        row["overlap_pixels"] = overlap_pixels

            temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
            _write_manifest(temporary_manifest, fields, rows)
            os.replace(temporary_manifest, manifest_path)
        except BaseException:
            for record in backup_records:
                source_path = _resolve_under(dataset, str(record["source_mask"]))
                shutil.copy2(backup / str(record["backup_mask"]), source_path)
            manifest_path.write_bytes(original_manifest)
            raise

        return {
            "active_width_px": 7,
            "groups": {group: {"tiles": EXPECTED_TILES[group], "d04_tiles": EXPECTED_D04[group]} for group in GROUPS},
            "backup": str(backup),
            "lb_variants": str(lb_variants),
            "lb_summary": summary,
            "source_manifest_sha256": _sha256(manifest_path),
        }
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument(
        "--lb-variants",
        type=Path,
        default=ROOT / "outputs" / "KYT-SC-1R-2LB1-1_d04_thinning_preview",
    )
    args = parser.parse_args()
    print(json.dumps(prepare_kyt_masks(args.dataset, args.backup, args.lb_variants), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
