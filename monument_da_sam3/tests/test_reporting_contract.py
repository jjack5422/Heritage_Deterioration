from pathlib import Path

import numpy as np

from monument_da_sam3.reporting import RunLayout, save_concept_qualitative


def test_qualitative_is_namespaced_by_concept(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "fold0")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    first = save_concept_qualitative(layout, concept="crack_craquelure", image_id="tile.png", input_rgb=image, target=mask, prediction=mask)
    second = save_concept_qualitative(layout, concept="loss", image_id="tile.png", input_rgb=image, target=mask, prediction=mask)
    assert first["input_path"] != second["input_path"]
    assert (layout.root / first["overlay_path"]).is_file()
    assert (layout.root / second["overlay_path"]).is_file()


def test_empty_target_has_finite_f1_and_iou(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "fold0")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    empty = np.zeros((4, 4), dtype=bool)
    correct = save_concept_qualitative(
        layout,
        concept="loss",
        image_id="negative.png",
        input_rgb=image,
        target=empty,
        prediction=empty,
    )
    false_positive = empty.copy()
    false_positive[0, 0] = True
    wrong = save_concept_qualitative(
        layout,
        concept="loss",
        image_id="false_positive.png",
        input_rgb=image,
        target=empty,
        prediction=false_positive,
    )
    assert (correct["f1"], correct["iou"]) == (1.0, 1.0)
    assert (wrong["f1"], wrong["iou"]) == (0.0, 0.0)
