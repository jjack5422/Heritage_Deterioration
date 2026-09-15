from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluation.infer_sam3_adapter_dataset import (
    compose_panels,
    discover_pairs,
    make_target,
)


def test_discover_pairs_preserves_source_group_and_matches_stems(tmp_path: Path) -> None:
    group = tmp_path / "temple" / "panel_a"
    images = group / "image_tiles"
    masks = group / "mask_tiles"
    images.mkdir(parents=True)
    masks.mkdir()
    Image.new("RGB", (512, 512)).save(images / "tile.jpg")
    Image.new("L", (512, 512)).save(masks / "tile.png")

    pairs = discover_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0].source_group == "panel_a"
    assert pairs[0].image.stem == pairs[0].mask.stem == "tile"


def test_make_target_can_exclude_crack_from_craquelure_ground_truth() -> None:
    mask = np.array([[0, 1, 2], [3, 4, 11]], dtype=np.uint8)

    target = make_target(mask, raw_ids=(4,))

    np.testing.assert_array_equal(
        target,
        np.array([[False, False, False], [False, True, False]]),
    )


def test_compose_panels_has_fixed_four_panel_order() -> None:
    image = np.full((2, 3, 3), 127, dtype=np.uint8)
    target = np.array([[True, False, False], [False, False, False]])
    prediction = np.array([[False, True, False], [False, False, False]])

    composite = compose_panels(image, target, prediction)

    assert composite.shape == (2, 12, 3)
    np.testing.assert_array_equal(composite[:, :3], image)
    assert tuple(composite[0, 3]) == (244, 63, 94)
    assert tuple(composite[0, 7]) == (6, 182, 212)
