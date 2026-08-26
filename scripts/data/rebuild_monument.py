"""Rebuild the monument dataset's derived masks from a CVAT XML export.

The source images remain in the original CVAT upload package.  This utility
decodes the corrected XML export into colour masks and regenerates every
derived index, category/source symlink tree, and representative preview.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


CLASS_IDS = {
    "background": 0,
    "crack": 1,
    "loss": 2,
    "shrinkage": 3,
    "craquelure": 4,
    "flaking": 5,
    "stain": 6,
}
# Semantic masks cannot encode cross-class overlap. Draw in this low-to-high
# precedence order, matching the existing reviewed masks: crack > loss >
# craquelure for the three conflicts in dataset_v2.
OVERLAP_PRIORITY = ("stain", "flaking", "shrinkage", "craquelure", "loss", "crack")
CLASS_COLORS = {
    "background": (0, 0, 0),
    "crack": (255, 24, 3),
    "loss": (9, 249, 213),
    "shrinkage": (149, 0, 222),
    "craquelure": (102, 255, 102),
    "flaking": (236, 236, 0),
    "stain": (93, 149, 13),
}
CLASS_NAMES_ZH = {
    "background": "背景",
    "crack": "裂縫",
    "loss": "缺失",
    "shrinkage": "皺縮",
    "craquelure": "龜裂",
    "flaking": "起甲",
    "stain": "候選汙漬",
}
CATEGORY_DIRS = (
    ("01_裂縫_crack", "crack"),
    ("02_缺失_loss", "loss"),
    ("03_皺縮_shrinkage", "shrinkage"),
    ("04_龜裂_craquelure", "craquelure"),
    ("05_起甲_flaking", "flaking"),
    ("06_候選汙漬_stain", "stain"),
    ("07_僅背景_background_only", "background_only"),
)
PREVIEW_NAMES = {
    "crack": "01_裂縫_crack.jpg",
    "loss": "02_缺失_loss.jpg",
    "shrinkage": "03_皺縮_shrinkage.jpg",
    "craquelure": "04_龜裂_craquelure.jpg",
    "flaking": "05_起甲_flaking.jpg",
    "stain": "06_候選汙漬_stain.jpg",
    "background": "07_僅背景_background_only.jpg",
}
SOURCE_RELATIVE = Path("01_CVAT原始匯出/clean_v2_multiclass_512_cvat")
SOURCE_GROUP_RE = re.compile(r"^(?P<group>.+)_R\d+_C\d+")


@dataclass(frozen=True)
class RenderedMask:
    class_mask: np.ndarray
    cross_class_overlap_pixels: int
    cross_class_overlap_by_pair: dict[str, int]


@dataclass(frozen=True)
class ImageRecord:
    name: str
    source_group: str
    class_counts: dict[str, int]
    disease_pixels: int
    overlap_pixels: int
    overlap_by_pair: dict[str, int]


def decode_cvat_rle(
    *,
    rle: str,
    image_width: int,
    image_height: int,
    left: int,
    top: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Decode CVAT's alternating zero/one runs into an image-sized mask."""

    if min(image_width, image_height, left, top, width, height) < 0:
        raise ValueError("CVAT mask coordinates must be non-negative")
    if width == 0 or height == 0:
        raise ValueError("CVAT mask bounding box must have non-zero dimensions")
    if left + width > image_width or top + height > image_height:
        raise ValueError("CVAT mask bounding box exceeds image dimensions")

    try:
        runs = [int(value.strip()) for value in rle.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("CVAT mask RLE must contain only integers") from error
    if not runs or any(run < 0 for run in runs):
        raise ValueError("CVAT mask RLE must contain non-negative runs")
    if sum(runs) != width * height:
        raise ValueError(
            f"CVAT mask RLE covers {sum(runs)} pixels; expected {width * height}"
        )

    local = np.zeros(width * height, dtype=bool)
    offset = 0
    is_foreground = False
    for run in runs:
        if is_foreground and run:
            local[offset : offset + run] = True
        offset += run
        is_foreground = not is_foreground

    canvas = np.zeros((image_height, image_width), dtype=bool)
    canvas[top : top + height, left : left + width] = local.reshape(height, width)
    return canvas


def render_cvat_image(image: ET.Element) -> RenderedMask:
    """Render CVAT masks into the project's single-label semantic mask format."""

    image_width = int(image.attrib["width"])
    image_height = int(image.attrib["height"])
    layers = {
        label: np.zeros((image_height, image_width), dtype=bool)
        for label in CLASS_IDS
        if label != "background"
    }

    for shape in image.findall("mask"):
        label = shape.attrib.get("label")
        if label not in CLASS_IDS:
            raise ValueError(f"Unknown CVAT label {label!r} in {image.attrib['name']}")
        mask = decode_cvat_rle(
            rle=shape.attrib["rle"],
            image_width=image_width,
            image_height=image_height,
            left=int(shape.attrib["left"]),
            top=int(shape.attrib["top"]),
            width=int(shape.attrib["width"]),
            height=int(shape.attrib["height"]),
        )
        if label != "background":
            layers[label] |= mask

    foreground_layers = np.stack(list(layers.values()), axis=0)
    overlap_pixels = int(np.count_nonzero(foreground_layers.sum(axis=0) > 1))
    overlap_by_pair = {
        f"{first}__{second}": int(np.count_nonzero(layers[first] & layers[second]))
        for index, first in enumerate(OVERLAP_PRIORITY)
        for second in OVERLAP_PRIORITY[index + 1 :]
        if int(np.count_nonzero(layers[first] & layers[second])) > 0
    }
    class_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for label in OVERLAP_PRIORITY:
        class_mask[layers[label]] = CLASS_IDS[label]
    return RenderedMask(
        class_mask=class_mask,
        cross_class_overlap_pixels=overlap_pixels,
        cross_class_overlap_by_pair=overlap_by_pair,
    )


def class_mask_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for label, class_id in CLASS_IDS.items():
        rgb[class_mask == class_id] = CLASS_COLORS[label]
    return rgb


def source_group_for(filename: str) -> str:
    match = SOURCE_GROUP_RE.match(filename)
    if not match:
        raise ValueError(f"Unable to infer source group from filename: {filename}")
    return match.group("group")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_relative_symlink(link: Path, relative_target: str) -> None:
    link.symlink_to(relative_target)


def write_category_tree(root: Path, records: Iterable[ImageRecord]) -> None:
    records = tuple(records)
    for directory_name, label in CATEGORY_DIRS:
        image_dir = root / "02_依病害分類" / directory_name / "原圖"
        mask_dir = root / "02_依病害分類" / directory_name / "遮罩"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        selected = (
            record
            for record in records
            if (record.disease_pixels == 0 if label == "background_only" else record.class_counts[label] > 0)
        )
        for record in selected:
            make_relative_symlink(
                image_dir / record.name,
                f"../../../{SOURCE_RELATIVE.as_posix()}/JPEGImages/{record.name}",
            )
            make_relative_symlink(
                mask_dir / record.name,
                f"../../../{SOURCE_RELATIVE.as_posix()}/SegmentationClass/{record.name}",
            )


def write_source_tree(root: Path, records: Iterable[ImageRecord]) -> None:
    for record in records:
        image_dir = root / "03_依來源群組" / record.source_group / "原圖"
        mask_dir = root / "03_依來源群組" / record.source_group / "遮罩"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        make_relative_symlink(
            image_dir / record.name,
            f"../../../{SOURCE_RELATIVE.as_posix()}/JPEGImages/{record.name}",
        )
        make_relative_symlink(
            mask_dir / record.name,
            f"../../../{SOURCE_RELATIVE.as_posix()}/SegmentationClass/{record.name}",
        )


def write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_statistics(root: Path, records: list[ImageRecord], annotation_xml: Path) -> None:
    info_dir = root / "00_資料說明"
    info_dir.mkdir(parents=True)
    image_pixels = 512 * 512
    ordered_labels = [label for label in CLASS_IDS if label != "background"]

    index_rows: list[list[object]] = []
    for record in records:
        labels = [label for label in ordered_labels if record.class_counts[label] > 0]
        index_rows.append(
            [
                record.name,
                record.source_group,
                "、".join(CLASS_NAMES_ZH[label] for label in labels) or "僅背景",
                ";".join(labels) or "background_only",
                record.disease_pixels,
                f"{record.disease_pixels / image_pixels * 100:.6f}",
                record.class_counts["background"],
                *(record.class_counts[label] for label in ordered_labels),
                f"{SOURCE_RELATIVE.as_posix()}/JPEGImages/{record.name}",
                f"{SOURCE_RELATIVE.as_posix()}/SegmentationClass/{record.name}",
            ]
        )
    write_csv(
        info_dir / "影像索引.csv",
        [
            "檔名",
            "來源群組",
            "中文標籤",
            "英文標籤",
            "病害像素數",
            "病害占比_pct",
            "背景像素數",
            "裂縫_crack_像素數",
            "缺失_loss_像素數",
            "皺縮_shrinkage_像素數",
            "龜裂_craquelure_像素數",
            "起甲_flaking_像素數",
            "候選汙漬_stain_像素數",
            "原圖相對路徑",
            "遮罩相對路徑",
        ],
        index_rows,
    )

    class_rows: list[list[object]] = []
    for label in ordered_labels:
        positive = [record for record in records if record.class_counts[label] > 0]
        pixels = sum(record.class_counts[label] for record in records)
        representative = max(positive, key=lambda record: record.class_counts[label], default=None)
        coverage = [record.class_counts[label] / image_pixels * 100 for record in positive]
        class_rows.append(
            [
                CLASS_NAMES_ZH[label],
                label,
                "否（候選）" if label == "stain" else "是",
                len(positive),
                f"{len(positive) / len(records) * 100:.6f}",
                pixels,
                f"{pixels / (len(records) * image_pixels) * 100:.6f}",
                f"{sum(coverage) / len(coverage):.6f}" if coverage else "",
                f"{max(coverage):.6f}" if coverage else "",
                representative.name if representative else "",
            ]
        )
    background_records = [record for record in records if record.disease_pixels == 0]
    class_rows.append(
        ["僅背景", "background_only", "不適用", len(background_records), f"{len(background_records) / len(records) * 100:.6f}", "", "", "", "", ""],
    )
    write_csv(
        info_dir / "類別統計.csv",
        [
            "中文類別",
            "英文類別",
            "正式類別",
            "含此類別影像數",
            "影像占比_pct",
            "像素數",
            "全資料像素占比_pct",
            "含類別影像平均覆蓋率_pct",
            "最高覆蓋率_pct",
            "代表影像",
        ],
        class_rows,
    )

    group_rows: list[list[object]] = []
    for group in sorted({record.source_group for record in records}):
        members = [record for record in records if record.source_group == group]
        group_rows.append(
            [
                group,
                len(members),
                sum(record.disease_pixels == 0 for record in members),
                *(sum(record.class_counts[label] > 0 for record in members) for label in ordered_labels),
            ]
        )
    write_csv(
        info_dir / "來源群組統計.csv",
        [
            "來源群組",
            "影像數",
            "僅背景影像數",
            "裂縫_crack_影像數",
            "缺失_loss_影像數",
            "皺縮_shrinkage_影像數",
            "龜裂_craquelure_影像數",
            "起甲_flaking_影像數",
            "候選汙漬_stain_影像數",
        ],
        group_rows,
    )

    overlap_by_pair = Counter()
    for record in records:
        overlap_by_pair.update(record.overlap_by_pair)
    report = {
        "annotation_xml": annotation_xml.name,
        "annotation_xml_sha256": sha256(annotation_xml),
        "image_count": len(records),
        "source_group_count": len(group_rows),
        "background_only_count": len(background_records),
        "class_image_counts": {
            label: sum(record.class_counts[label] > 0 for record in records)
            for label in ordered_labels
        },
        "class_pixel_counts": {
            label: sum(record.class_counts[label] for record in records)
            for label in ordered_labels
        },
        "cross_class_overlap_pixels": sum(record.overlap_pixels for record in records),
        "cross_class_overlap_by_pair": dict(sorted(overlap_by_pair.items())),
        "cross_class_overlap_images": [
            {"filename": record.name, "pixels": record.overlap_pixels, "pairs": record.overlap_by_pair}
            for record in records
            if record.overlap_pixels
        ],
        "candidate_labels": ["stain"],
    }
    (info_dir / "修正摘要.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (info_dir / "完整性檢查.txt").write_text(
        "\n".join(
            [
                "古蹟裂縫資料完整性檢查（dataset_v2 修正標註）",
                f"修正標註 XML: {annotation_xml.name}",
                f"修正標註 SHA-256: {report['annotation_xml_sha256']}",
                f"原圖數: {len(records)}",
                f"遮罩數: {len(records)}",
                "一一配對: True",
                "原圖格式: 512x512 RGB PNG（全部）",
                "遮罩格式: 512x512 RGB PNG（全部）",
                "未知遮罩顏色像素: 0",
                f"跨類別重疊像素: {report['cross_class_overlap_pixels']}",
                "重疊處理: 單標籤遮罩依 crack > loss > craquelure > shrinkage > flaking > stain 優先序保留；原始 XML 保留完整重疊資訊。",
                f"僅背景影像數: {len(background_records)}",
                f"來源群組數: {len(group_rows)}",
                "stain 狀態: 候選標註，未列為正式五類劣化",
                "",
            ]
        ),
        encoding="utf-8",
    )


def overlay(image: Image.Image, class_mask: np.ndarray, label: str) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = rgb.copy()
    selected = class_mask == CLASS_IDS[label]
    if label == "background":
        selected = class_mask == CLASS_IDS["background"]
    if selected.any():
        colour = np.asarray(CLASS_COLORS[label], dtype=np.float32)
        result[selected] = result[selected] * 0.52 + colour * 0.48
    return Image.fromarray(result.astype(np.uint8), "RGB")


def write_previews(root: Path, records: list[ImageRecord], image_dir: Path, mask_dir: Path) -> None:
    preview_dir = root / "04_預覽"
    preview_dir.mkdir(parents=True)
    representatives: dict[str, ImageRecord] = {}
    for label in CLASS_IDS:
        if label == "background":
            candidates = [record for record in records if record.disease_pixels == 0]
        else:
            candidates = [record for record in records if record.class_counts[label] > 0]
        if candidates:
            score = (
                (lambda record: record.class_counts["background"])
                if label == "background"
                else (lambda record: record.class_counts[label])
            )
            representatives[label] = max(candidates, key=score)

    tiles: list[Image.Image] = []
    for label in CLASS_IDS:
        record = representatives.get(label)
        if not record:
            continue
        with Image.open(image_dir / record.name) as image, Image.open(mask_dir / record.name) as mask:
            colours = np.asarray(mask.convert("RGB"))
            class_mask = np.zeros(colours.shape[:2], dtype=np.uint8)
            for class_label, colour in CLASS_COLORS.items():
                class_mask[np.all(colours == colour, axis=2)] = CLASS_IDS[class_label]
            rendered = overlay(image, class_mask, label)
            rendered.save(preview_dir / PREVIEW_NAMES[label], quality=92)
            tile = rendered.copy()
            tile.thumbnail((320, 320), Image.Resampling.LANCZOS)
            tiles.append(tile)

    sheet = Image.new("RGB", (960, 960), "black")
    for index, tile in enumerate(tiles):
        x = (index % 3) * 320
        y = (index // 3) * 320
        sheet.paste(tile, (x, y))
    sheet.save(preview_dir / "00_病害總覽.jpg", quality=92)


def build_staging_area(annotation_xml: Path, dataset_root: Path, source_dir: Path) -> tuple[Path, list[ImageRecord]]:
    images_dir = source_dir / "JPEGImages"
    source_names = {path.name for path in images_dir.glob("*.png")}
    xml_root = ET.parse(annotation_xml).getroot()
    xml_images = xml_root.findall("image")
    xml_names = [image.attrib["name"] for image in xml_images]
    if len(xml_names) != len(set(xml_names)):
        raise ValueError("CVAT XML contains duplicate image names")
    if set(xml_names) != source_names:
        missing = sorted(source_names - set(xml_names))
        extra = sorted(set(xml_names) - source_names)
        raise ValueError(f"CVAT/source image mismatch: missing={missing[:3]}, extra={extra[:3]}")

    staging_dir = Path(tempfile.mkdtemp(prefix=".dataset_v2_rebuild_", dir=dataset_root))
    try:
        staged_masks = staging_dir / "SegmentationClass"
        staged_masks.mkdir()
        records: list[ImageRecord] = []
        for image in sorted(xml_images, key=lambda item: item.attrib["name"]):
            filename = image.attrib["name"]
            expected_size = (int(image.attrib["width"]), int(image.attrib["height"]))
            with Image.open(images_dir / filename) as source_image:
                if source_image.mode != "RGB" or source_image.size != expected_size:
                    raise ValueError(f"Source image does not match XML metadata: {filename}")
            rendered = render_cvat_image(image)
            Image.fromarray(class_mask_to_rgb(rendered.class_mask), "RGB").save(staged_masks / filename)
            counts = {
                label: int(np.count_nonzero(rendered.class_mask == class_id))
                for label, class_id in CLASS_IDS.items()
            }
            records.append(
                ImageRecord(
                    name=filename,
                    source_group=source_group_for(filename),
                    class_counts=counts,
                    disease_pixels=rendered.class_mask.size - counts["background"],
                    overlap_pixels=rendered.cross_class_overlap_pixels,
                    overlap_by_pair=rendered.cross_class_overlap_by_pair,
                )
            )

        write_category_tree(staging_dir, records)
        write_source_tree(staging_dir, records)
        write_statistics(staging_dir, records, annotation_xml)
        write_previews(staging_dir, records, images_dir, staged_masks)
        return staging_dir, records
    except Exception:
        shutil.rmtree(staging_dir)
        raise


def activate_staging_area(dataset_root: Path, source_dir: Path, staging_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    old_masks = source_dir / "SegmentationClass"
    transitions = [
        (old_masks, backup_dir / "SegmentationClass"),
        *( (dataset_root / name, backup_dir / name) for name in ("00_資料說明", "02_依病害分類", "03_依來源群組", "04_預覽") ),
    ]
    try:
        for current, backup in transitions:
            shutil.move(current, backup)
        shutil.move(staging_dir / "SegmentationClass", old_masks)
        for name in ("00_資料說明", "02_依病害分類", "03_依來源群組", "04_預覽"):
            shutil.move(staging_dir / name, dataset_root / name)
    except Exception:
        raise RuntimeError(
            f"Activation stopped; previous artifacts are recoverable in {backup_dir}"
        ) from None
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-xml", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("古蹟裂縫"))
    parser.add_argument("--source-relative", type=Path, default=SOURCE_RELATIVE)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_xml = args.annotations_xml.resolve()
    dataset_root = args.dataset_root.resolve()
    source_dir = dataset_root / args.source_relative
    backup_dir = args.backup_dir or dataset_root / "99_修正前備份_dataset_clean_v2"
    if not annotation_xml.is_file():
        raise FileNotFoundError(f"Annotation XML not found: {annotation_xml}")
    if not (source_dir / "JPEGImages").is_dir():
        raise FileNotFoundError(f"Source image directory not found: {source_dir / 'JPEGImages'}")

    staging_dir, records = build_staging_area(annotation_xml, dataset_root, source_dir)
    counts = Counter(label for record in records for label, pixels in record.class_counts.items() if pixels > 0)
    print(f"Validated {len(records)} image/mask pairs; non-empty labels: {dict(sorted(counts.items()))}")
    if args.dry_run:
        shutil.rmtree(staging_dir)
        print("Dry run complete; no dataset files were changed.")
        return 0
    activate_staging_area(dataset_root, source_dir, staging_dir, backup_dir)
    print(f"Rebuilt dataset successfully. Previous artifacts: {backup_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
