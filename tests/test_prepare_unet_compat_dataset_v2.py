from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_unet_compat_dataset_v2 import prepare_unet_compatibility


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_unet_compatibility_writes_verified_nested_views(tmp_path: Path) -> None:
    root = tmp_path / "dataset_v2_3class"
    images, masks, splits = root / "images", root / "masks", root / "splits"
    images.mkdir(parents=True)
    masks.mkdir()
    splits.mkdir()
    names = (
        "site_a_R1_C01__y00000_x00000.png",
        "site_a_R1_C02__y00000_x00000.png",
        "site_b_R1_C01__y00000_x00000.png",
        "site_b_R1_C02__y00000_x00000.png",
    )
    rows: list[dict[str, str]] = []
    for index, name in enumerate(names):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), "RGB").save(images / name)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        mask[0, 1] = 2
        if index == 0:
            mask[0, 2] = 255
        Image.fromarray(mask, "L").save(masks / name)
        rows.append({"image": name, "group": name.split("_R", 1)[0], "mask_sha256": _sha256(masks / name)})
    _write_csv(root / "manifest.csv", rows, ("image", "group", "mask_sha256"))
    manifest_hash = _sha256(root / "manifest.csv")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "segment-any-crack-3class-v1",
                "classes": {"background": 0, "crack": 1, "craquelure": 2},
                "ignore_index": 255,
                "image_count": len(names),
                "group_count": 2,
                "n_splits": 2,
                "manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    group_folds = {
        "schema": "group-5fold-v1",
        "seed": 42,
        "folds": {
            "fold0": {"validation_groups": ["site_a"], "validation_images": list(names[:2])},
            "fold1": {"validation_groups": ["site_b"], "validation_images": list(names[2:])},
        },
    }
    (splits / "group_5fold.json").write_text(json.dumps(group_folds), encoding="utf-8")

    result = prepare_unet_compatibility(root)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert result["pair_count"] == 4
    assert manifest["training_eligible"] is True
    assert manifest["label_contract"] == {
        "class_ids": {"background": 0, "crack": 1, "craquelure": 2},
        "ignore_value": 255,
    }
    assert [item["tile"] for item in json.loads((root / "tile_index.json").read_text())["items"]] == list(names)

    split0 = json.loads((splits / "fold0.json").read_text(encoding="utf-8"))
    assert split0["holdout_tiles"] == list(names[:2])
    assert split0["folds"][0]["train"] == list(names[2:])
    assert split0["data_contract"]["class_names"] == ["background", "crack", "craquelure"]
    assert split0["data_contract"]["task_mask_manifest_sha256"] == manifest["mask_manifest_sha256"]

