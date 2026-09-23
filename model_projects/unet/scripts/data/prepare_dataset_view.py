"""Write a verified metadata view so the U-Net trainer can use dataset_v2_3class.

The canonical dataset remains ``manifest.csv`` plus ``group_5fold.json``.  This
tool only creates the JSON contract expected by the ``unet`` project and
never rewrites images, masks, or the canonical CSV split files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASS_IDS = {"background": 0, "crack": 1, "craquelure": 2}
IGNORE_VALUE = 255


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_rows(rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    payload = "".join("\t".join(row[field] for field in fields) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_manifest_hash(payload: dict[str, Any]) -> str:
    serialised = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"image", "group", "mask_sha256"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"{path} must contain exactly {sorted(expected)}")
    rows = sorted(rows, key=lambda row: row["image"])
    names = [row["image"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("manifest.csv contains duplicate image names")
    return rows


def _validate_items(root: Path, rows: list[dict[str, str]]) -> tuple[str, str]:
    image_rows: list[dict[str, str]] = []
    mask_rows: list[dict[str, str]] = []
    allowed = set(CLASS_IDS.values()) | {IGNORE_VALUE}
    for row in rows:
        name = row["image"]
        image_path, mask_path = root / "images" / name, root / "masks" / name
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"missing image/mask pair for {name}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            pixels = np.asarray(mask)
            if image.size != mask.size or pixels.ndim != 2:
                raise ValueError(f"invalid image/mask dimensions for {name}")
        if not set(np.unique(pixels).tolist()).issubset(allowed):
            raise ValueError(f"unsupported mask values in {name}")
        actual_mask_hash = _sha256_file(mask_path)
        if actual_mask_hash != row["mask_sha256"]:
            raise ValueError(f"mask hash differs from canonical manifest for {name}")
        image_rows.append({"image": name, "sha256": _sha256_file(image_path)})
        mask_rows.append({"image": name, "sha256": actual_mask_hash})
    return (
        _sha256_rows(image_rows, ("image", "sha256")),
        _sha256_rows(mask_rows, ("image", "sha256")),
    )


def _load_group_folds(root: Path, names: set[str], groups_by_name: dict[str, str]) -> list[tuple[int, list[str]]]:
    path = root / "splits" / "group_5fold.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, dict) or not folds:
        raise ValueError("group_5fold.json must contain folds")
    result: list[tuple[int, list[str]]] = []
    seen: set[str] = set()
    for key, value in sorted(folds.items()):
        if not key.startswith("fold") or not key[4:].isdigit() or not isinstance(value, dict):
            raise ValueError(f"invalid group fold entry {key!r}")
        index = int(key[4:])
        validation_groups = set(value.get("validation_groups", []))
        validation_names = sorted(value.get("validation_images", []))
        if not validation_groups or not validation_names:
            raise ValueError(f"{key} is empty")
        if set(validation_names) - names:
            raise ValueError(f"{key} contains names outside manifest")
        if {groups_by_name[name] for name in validation_names} != validation_groups:
            raise ValueError(f"{key} groups do not agree with canonical manifest")
        expected_names = {name for name, group in groups_by_name.items() if group in validation_groups}
        if set(validation_names) != expected_names:
            raise ValueError(f"{key} does not contain every tile from its validation groups")
        if seen & set(validation_names):
            raise ValueError("group folds overlap")
        seen.update(validation_names)
        result.append((index, validation_names))
    if seen != names:
        raise ValueError("group folds do not cover the manifest exactly once")
    return result


def prepare_unet_compatibility(dataset_root: str | Path) -> dict[str, Any]:
    """Create the U-Net metadata view while retaining the dataset_v2 source contract."""

    root = Path(dataset_root).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("classes") != CLASS_IDS or metadata.get("ignore_index") != IGNORE_VALUE:
        raise ValueError("dataset is not the expected dataset_v2 three-class contract")
    rows = _read_manifest(root)
    canonical_manifest = root / "manifest.csv"
    if metadata.get("image_count") != len(rows):
        raise ValueError("metadata image_count differs from manifest.csv")
    if metadata.get("manifest_sha256") != _sha256_file(canonical_manifest):
        raise ValueError("metadata manifest_sha256 differs from manifest.csv")
    image_hash, mask_hash = _validate_items(root, rows)
    names = [row["image"] for row in rows]
    groups = {row["image"]: row["group"] for row in rows}
    folds = _load_group_folds(root, set(names), groups)

    manifest: dict[str, Any] = {
        "schema_version": "dataset-v2-unet-compat-v1",
        "view_role": "dataset_v2_3class_unet_compatibility",
        "canonical_manifest_sha256": metadata["manifest_sha256"],
        "annotation_sha256": metadata.get("annotation_sha256"),
        "pair_count": len(rows),
        "training_eligible": True,
        "label_contract": {"class_ids": CLASS_IDS, "ignore_value": IGNORE_VALUE},
        "image_manifest_sha256": image_hash,
        "mask_manifest_sha256": mask_hash,
        "mask_values": [0, 1, 2, 255],
        "group_split_source": "splits/group_5fold.json",
        "group_split_sha256": _sha256_file(root / "splits" / "group_5fold.json"),
    }
    manifest["manifest_sha256"] = _canonical_manifest_hash(manifest)
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "tile_index.json",
        {
            "schema_version": "dataset-v2-unet-tile-index-v1",
            "items": [
                {"tile": row["image"], "source_group": row["group"], "mask_sha256": row["mask_sha256"]}
                for row in rows
            ],
        },
    )
    contract = {
        "class_names": list(CLASS_IDS),
        "ignore_value": IGNORE_VALUE,
        "task_image_manifest_sha256": image_hash,
        "task_mask_manifest_sha256": mask_hash,
    }
    for index, holdout in folds:
        holdout_set = set(holdout)
        _write_json(
            root / "splits" / f"fold{index}.json",
            {
                "schema_version": "dataset-v2-unet-nested-split-v1",
                "outer_fold": index,
                "task": "crack_craquelure_3class",
                "holdout_tiles": holdout,
                "folds": [{"train": [name for name in names if name not in holdout_set]}],
                "data_contract": contract,
            },
        )
    summary = {
        "schema_version": "dataset-v2-unet-compatibility-v1",
        "dataset_root": str(root),
        "pair_count": len(rows),
        "group_count": len(set(groups.values())),
        "fold_count": len(folds),
        "canonical_manifest_sha256": metadata["manifest_sha256"],
        "unet_manifest_sha256": manifest["manifest_sha256"],
        "image_manifest_sha256": image_hash,
        "mask_manifest_sha256": mask_hash,
    }
    _write_json(root / "unet_compatibility.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_unet_compatibility(args.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
