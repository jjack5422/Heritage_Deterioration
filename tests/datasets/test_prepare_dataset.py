from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.prepare_three_class import prepare_dataset


def _rle(mask):
    flat = list(mask)
    runs, current, count = [], 0, 0
    for value in flat + [0]:
        value = int(value)
        if value == current:
            count += 1
        else:
            runs.append(count)
            current, count = value, 1
    return ",".join(map(str, runs))


def test_prepare_maps_target_and_non_target_to_ignore_and_keeps_groups(tmp_path):
    source = tmp_path / "src"
    image_dir = source / "JPEGImages"
    semantic_dir = source / "SegmentationClass"
    image_dir.mkdir(parents=True)
    semantic_dir.mkdir(parents=True)
    rows = []
    for index, group in enumerate(("A", "A", "B", "B", "C")):
        name = f"{group}_R1_C{index:02d}.png"
        Image.new("RGB", (4, 2), (20, 30, 40)).save(image_dir / name)
        semantic = Image.new("RGB", (4, 2), (0, 0, 0))
        semantic.putpixel((0, 0), (255, 24, 3))
        semantic.putpixel((1, 0), (102, 255, 102))
        semantic.putpixel((2, 0), (9, 249, 213))
        semantic.save(semantic_dir / name)
        # one crack pixel, one craquelure pixel, one loss pixel
        rows.append(
            f'<image id="{index}" name="{name}" width="4" height="2">'
            f'<mask label="crack" left="0" top="0" width="1" height="1" rle="0,1"/>'
            f'<mask label="craquelure" left="1" top="0" width="1" height="1" rle="0,1"/>'
            f'<mask label="loss" left="2" top="0" width="1" height="1" rle="0,1"/>'
            "</image>"
        )
    xml = source / "annotations.xml"
    xml.write_text("<annotations>" + "".join(rows) + "</annotations>", encoding="utf-8")

    output = tmp_path / "prepared"
    result = prepare_dataset(xml, image_dir, output, n_splits=2, seed=7)
    assert result["image_count"] == 5
    mask = np.asarray(Image.open(output / "masks" / "A_R1_C00.png"))
    assert mask[0, :4].tolist() == [1, 2, 255, 0]
    assert result["annotation_sha256"]
    for fold in range(2):
        lines = (output / "splits" / f"fold{fold}.csv").read_text().splitlines()
        assert lines[0] == "image,group,partition"
        assert len(lines) == 6
    # Group A must occur in exactly one fold's validation partition.
    val_folds = []
    for fold in range(2):
        val_folds.append(
            [line for line in (output / "splits" / f"fold{fold}.csv").read_text().splitlines()[1:] if ",A,val" in line]
        )
    assert sum(bool(items) for items in val_folds) == 1
