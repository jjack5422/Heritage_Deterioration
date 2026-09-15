"""Export manually selected tiles as a grouped five-foreground-class dataset.

The source dataset is never modified. Source class 6 (stain) becomes ignore=255;
background, crack, loss, shrinkage, craquelure, and flaking retain IDs 0-5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets/dataset_clean_v2_prohibited"
DEFAULT_SELECTIONS = ROOT / "outputs/dataset_review_jacky/selected.csv"
DEFAULT_DESTINATION = ROOT / "dataset_jacky"
GROUP_RE = re.compile(r"^(?P<group>.+)_R\d+_C\d+")
SOURCE_CLASS_IDS = {
    "background": 0,
    "crack": 1,
    "loss": 2,
    "shrinkage": 3,
    "craquelure": 4,
    "flaking": 5,
    "stain": 6,
}
TARGET_CLASS_IDS = {
    "background": 0,
    "crack": 1,
    "loss": 2,
    "shrinkage": 3,
    "craquelure": 4,
    "flaking": 5,
}
IGNORE_VALUE = 255


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_group(name: str) -> str:
    match = GROUP_RE.match(Path(name).stem)
    if not match:
        raise ValueError(f"Cannot derive source group from tile name: {name}")
    group = match.group("group")
    if group in {".", ".."} or "/" in group or "\\" in group:
        raise ValueError(f"Unsafe source group in tile name: {name}")
    return group


def _collection(group: str) -> str:
    return re.split(r"[-_]", group, maxsplit=1)[0]


def _read_selection_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("Selection CSV must contain a name column")
        names = [row["name"] for row in reader if not row.get("status") or row["status"] == "keep"]
    if not names:
        raise ValueError("Selection CSV contains no kept images")
    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Selection CSV contains duplicate names: {duplicates[:5]}")
    return names


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _classes_text() -> str:
    return "\n".join(
        [
            "0\tBG\tBackground",
            "1\tD-01\tCrack / Fissure",
            "2\tD-02\tLoss",
            "3\tD-03\tShrinkage",
            "4\tD-04\tCraquelure",
            "5\tD-05\tFlaking",
            "255\tIGNORE\tIgnored source labels (Stain)",
            "",
        ]
    )


def _readme_text(pair_count: int, group_counts: dict[str, int], selection_path: Path) -> str:
    rows = "\n".join(
        f"| `{group}` | {count} |" for group, count in sorted(group_counts.items())
    )
    return f"""# Jacky Five-Class Tile Dataset

## 概要

此資料集由人工複選名單建立，用於五種劣化類別的影像分割訓練。

- Image/mask 配對：{pair_count} 組
- 原始影像組數：{len(group_counts)}
- Mask 格式：8-bit 單通道 class-index PNG
- 人工複選名單：`{selection_path}`
- Raster mask 是 tile-level ground truth；來源沒有逐組可完整對應的 COCO JSON，因此不建立或捏造 `annotations_coco.json`。

## 資料夾結構

```text
dataset_jacky/
├── README.md
├── classes.txt
├── manifest.json
└── <collection>/
    └── <source_group>/
        ├── image_tiles/
        └── mask_tiles/
```

Image 與 mask 保留相同檔名 stem。影像保留來源 PNG，不重新壓縮；mask 為 PNG。

## Mask 編碼

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Crack / Fissure |
| 2 | Loss |
| 3 | Shrinkage |
| 4 | Craquelure |
| 5 | Flaking |
| 255 | Ignore（來源 Stain） |

載入 mask 時必須保留整數類別值；不要將 mask 正規化成 0–1。訓練 loss 應設定 `ignore_index=255`。

## 各原始影像組統計

| Source group | Tiles |
|---|---:|
{rows}
| **合計** | **{pair_count}** |

## 可追溯性

`manifest.json` 記錄來源資料集、人工名單 SHA-256、每個輸出 image/mask 的 SHA-256、類別契約及逐組數量。匯出程序不修改來源資料。
"""


def export_dataset(
    source_root: str | Path,
    selections_csv: str | Path,
    destination_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    selections = Path(selections_csv).resolve()
    destination = Path(destination_root).resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists; refusing to overwrite: {destination}")

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("label_contract", {}).get("class_ids") != SOURCE_CLASS_IDS:
        raise ValueError("source manifest is not the expected seven-class dataset contract")
    source_images = source / "images"
    source_masks = source / "masks"
    names = _read_selection_names(selections)
    missing = [
        name
        for name in names
        if not (source_images / name).is_file() or not (source_masks / f"{Path(name).stem}.png").is_file()
    ]
    if missing:
        raise ValueError(f"Selected image/mask pairs are missing from the source: {missing[:5]}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    records: list[dict[str, str]] = []
    group_counts: Counter[str] = Counter()
    pixel_counts: Counter[int] = Counter()
    try:
        for name in names:
            group = _source_group(name)
            group_counts[group] += 1
            output_group = temporary / _collection(group) / group
            output_images = output_group / "image_tiles"
            output_masks = output_group / "mask_tiles"
            output_images.mkdir(parents=True, exist_ok=True)
            output_masks.mkdir(parents=True, exist_ok=True)

            source_image = source_images / name
            source_mask = source_masks / f"{Path(name).stem}.png"
            output_image = output_images / name
            output_mask = output_masks / source_mask.name
            shutil.copy2(source_image, output_image)
            with Image.open(source_mask) as image:
                mask = np.asarray(image)
            if mask.ndim != 2:
                raise ValueError(f"Expected a single-channel class-ID mask: {name}")
            values, counts = np.unique(mask, return_counts=True)
            unknown = set(values.tolist()) - {*SOURCE_CLASS_IDS.values(), IGNORE_VALUE}
            if unknown:
                raise ValueError(f"Unsupported source label IDs in {name}: {sorted(unknown)}")
            remapped = mask.astype(np.uint8, copy=True)
            remapped[remapped == SOURCE_CLASS_IDS["stain"]] = IGNORE_VALUE
            Image.fromarray(remapped, mode="L").save(output_mask)
            remapped_values, remapped_counts = np.unique(remapped, return_counts=True)
            pixel_counts.update(
                {int(value): int(count) for value, count in zip(remapped_values, remapped_counts)}
            )
            records.append(
                {
                    "name": name,
                    "group": group,
                    "image": str(output_image.relative_to(temporary)),
                    "mask": str(output_mask.relative_to(temporary)),
                    "image_sha256": _sha256(output_image),
                    "mask_sha256": _sha256(output_mask),
                }
            )

        manifest = {
            "schema_version": "jacky-five-class-tile-dataset-v1",
            "pair_count": len(records),
            "group_count": len(group_counts),
            "label_contract": {
                "class_ids": TARGET_CLASS_IDS,
                "ignore_value": IGNORE_VALUE,
                "source_to_target_ids": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 255, "255": 255},
            },
            "pixel_counts": {
                **{name: pixel_counts[class_id] for name, class_id in TARGET_CLASS_IDS.items()},
                "ignore": pixel_counts[IGNORE_VALUE],
            },
            "group_counts": dict(sorted(group_counts.items())),
            "source": {
                "dataset": str(source),
                "manifest_sha256": _sha256(source_manifest_path),
                "selections_csv": str(selections),
                "selections_sha256": _sha256(selections),
            },
            "items": records,
        }
        (temporary / "classes.txt").write_text(_classes_text(), encoding="utf-8")
        (temporary / "README.md").write_text(
            _readme_text(len(records), dict(group_counts), selections), encoding="utf-8"
        )
        _write_json(temporary / "manifest.json", manifest)
        temporary.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="匯出人工複選後的五類分割資料集。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    manifest = export_dataset(args.source, args.selections, args.destination)
    print(json.dumps({"destination": str(args.destination.resolve()), "pair_count": manifest["pair_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
