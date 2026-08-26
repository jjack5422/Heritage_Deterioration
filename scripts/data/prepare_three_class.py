"""Prepare a reproducible three-class dataset from corrected CVAT XML.

Classes are background=0, crack=1, craquelure=2, and ignore=255 for every
other CVAT label. Splits are assigned by source group, never by tile.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

GROUP_RE = re.compile(r"^(?P<group>.+)_R\d+_C\d+")
TARGETS = {"crack": 1, "craquelure": 2}
IGNORE = 255
SEMANTIC_COLORS = {
    (0, 0, 0): 0,
    (255, 24, 3): 1,
    (102, 255, 102): 2,
    (9, 249, 213): IGNORE,
    (149, 0, 222): IGNORE,
    (236, 236, 0): IGNORE,
    (93, 149, 13): IGNORE,
}


def _decode(rle: str, width: int, height: int) -> np.ndarray:
    runs = [int(x) for x in rle.split(",") if x.strip()]
    if not runs or any(x < 0 for x in runs) or sum(runs) != width * height:
        raise ValueError("invalid CVAT RLE")
    out = np.zeros(width * height, dtype=bool)
    offset = 0
    foreground = False
    for run in runs:
        if foreground:
            out[offset : offset + run] = True
        offset += run
        foreground = not foreground
    return out.reshape(height, width)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(annotation_xml: Path, image_dir: Path, output_dir: Path,
                    *, n_splits: int = 5, seed: int = 42) -> dict:
    annotation_xml, image_dir, output_dir = map(Path, (annotation_xml, image_dir, output_dir))
    annotation_xml = annotation_xml.resolve()
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    root = ET.parse(annotation_xml).getroot()
    records = []
    output_images, output_masks = output_dir / "images", output_dir / "masks"
    semantic_dir = image_dir.parent / "SegmentationClass"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    for image in root.findall("image"):
        name = image.attrib["name"]
        width, height = int(image.attrib["width"]), int(image.attrib["height"])
        semantic = np.asarray(Image.open(semantic_dir / name).convert("RGB"))
        if semantic.shape[:2] != (height, width):
            raise ValueError(f"semantic mask dimensions differ for {name}")
        unknown = np.ones(semantic.shape[:2], dtype=bool)
        mask = np.full(semantic.shape[:2], IGNORE, dtype=np.uint8)
        for color, class_id in SEMANTIC_COLORS.items():
            pixels = np.all(semantic == color, axis=2)
            mask[pixels] = class_id
            unknown &= ~pixels
        if unknown.any():
            raise ValueError(f"unknown semantic mask colours in {name}")
        source = image_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_images / name)
        Image.fromarray(mask, mode="L").save(output_masks / name)
        match = GROUP_RE.match(Path(name).stem)
        group = match.group("group") if match else Path(name).stem
        records.append({"image": name, "group": group, "mask_sha256": _sha256(output_masks / name)})

    groups = sorted({record["group"] for record in records})
    if len(groups) < n_splits:
        raise ValueError(f"{len(groups)} groups cannot form {n_splits} folds")
    # Greedily balance target pixels while keeping assignment deterministic.
    stats = {group: [sum(int(np.count_nonzero(np.asarray(Image.open(output_masks / r["image"])) == c))
                       for r in records if r["group"] == group) for c in (1, 2)] for group in groups}
    groups.sort(key=lambda g: (-max(stats[g]), -sum(stats[g]), g))
    fold_groups = [[] for _ in range(n_splits)]
    fold_totals = [[0, 0] for _ in range(n_splits)]
    for group in groups:
        target = min(range(n_splits), key=lambda i: (fold_totals[i][0] + stats[group][0], fold_totals[i][1] + stats[group][1], len(fold_groups[i]), i))
        fold_groups[target].append(group)
        fold_totals[target] = [fold_totals[target][j] + stats[group][j] for j in range(2)]
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for fold, validation_groups in enumerate(fold_groups):
        validation_groups = set(validation_groups)
        path = splits_dir / f"fold{fold}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image", "group", "partition"))
            for record in sorted(records, key=lambda item: item["image"]):
                partition = "val" if record["group"] in validation_groups else "train"
                writer.writerow((record["image"], record["group"], partition))
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "group", "mask_sha256"))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda item: item["image"]))
    zip_candidate = annotation_xml.parents[3] / "dataset_v2.zip"
    group_schema = {"schema": "group-5fold-v1", "seed": seed, "folds": {}}
    for fold, validation_groups in enumerate(fold_groups):
        val = sorted(validation_groups)
        group_schema["folds"][f"fold{fold}"] = {
            "validation_groups": val,
            "validation_images": sorted(r["image"] for r in records if r["group"] in validation_groups),
        }
    group_json = splits_dir / "group_5fold.json"
    group_json.write_text(json.dumps(group_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "format": "segment-any-crack-3class-v1", "classes": {"background": 0, "crack": 1, "craquelure": 2},
        "ignore_index": IGNORE, "ignored_labels": ["loss", "shrinkage", "flaking", "stain"],
        "image_count": len(records), "group_count": len(groups), "n_splits": n_splits, "seed": seed,
        "annotation_xml": str(annotation_xml), "annotation_sha256": _sha256(annotation_xml),
        "annotation_zip": str(zip_candidate) if zip_candidate.is_file() else None,
        "annotation_zip_sha256": _sha256(zip_candidate) if zip_candidate.is_file() else None,
        "manifest_sha256": _sha256(manifest), "split_sha256": {f"fold{i}": _sha256(splits_dir / f"fold{i}.csv") for i in range(n_splits)},
        "validation_target_pixels": {f"fold{i}": fold_totals[i] for i in range(n_splits)},
        "fold_image_counts": {f"fold{i}": {"train": sum(r["group"] not in set(fold_groups[i]) for r in records), "val": sum(r["group"] in set(fold_groups[i]) for r in records)} for i in range(n_splits)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-xml", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.annotations_xml, args.image_dir, args.output_dir, n_splits=args.folds, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
