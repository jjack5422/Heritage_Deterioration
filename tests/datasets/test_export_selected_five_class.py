from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.data.export_selected_five_class import SOURCE_CLASS_IDS, export_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_groups_selected_pairs_and_maps_stain_to_ignore(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dataset_jacky"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "label_contract": {"class_ids": SOURCE_CLASS_IDS, "ignore_value": 255},
                "pair_count": 2,
            }
        ),
        encoding="utf-8",
    )
    names = ["KJTHT-SC-L-A4-4_R1_C01__y00000_x00000.png", "MGLST-SC-1L-A3-1_R2_C03__y00000_x00000.png"]
    source_mask_hashes = {}
    for name in names:
        Image.new("RGB", (4, 2), (20, 40, 60)).save(source / "images" / name)
        mask = np.array([[0, 1, 2, 3], [4, 5, 6, 255]], dtype=np.uint8)
        Image.fromarray(mask, mode="L").save(source / "masks" / name)
        source_mask_hashes[name] = _sha256(source / "masks" / name)
    selections = tmp_path / "selected.csv"
    with selections.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "status"])
        writer.writeheader()
        writer.writerow({"name": names[1], "status": "keep"})
        writer.writerow({"name": names[0], "status": "reject"})

    manifest = export_dataset(source, selections, destination)

    output_mask = destination / "MGLST" / "MGLST-SC-1L-A3-1" / "mask_tiles" / names[1]
    output_image = destination / "MGLST" / "MGLST-SC-1L-A3-1" / "image_tiles" / names[1]
    assert output_image.is_file()
    assert np.asarray(Image.open(output_mask)).tolist() == [[0, 1, 2, 3], [4, 5, 255, 255]]
    assert _sha256(source / "masks" / names[1]) == source_mask_hashes[names[1]]
    assert manifest["pair_count"] == 1
    assert manifest["label_contract"]["class_ids"] == {
        "background": 0,
        "crack": 1,
        "loss": 2,
        "shrinkage": 3,
        "craquelure": 4,
        "flaking": 5,
    }
    assert manifest["pixel_counts"]["ignore"] == 2
    assert (destination / "README.md").is_file()
    assert (destination / "classes.txt").is_file()


def test_export_refuses_to_overwrite_destination(tmp_path: Path) -> None:
    destination = tmp_path / "dataset_jacky"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_dataset(tmp_path / "source", tmp_path / "selected.csv", destination)
