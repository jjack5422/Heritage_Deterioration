"""Dataset-format compatibility for the combined expert manifests."""

from pathlib import Path

import numpy as np
from PIL import Image

from sam3_adapter.training_data import ExpertTileDataset
from scripts.data.prepare_combined_expert_splits import _actual_path


def _save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path)


def test_loader_maps_class_index_mask_and_preserves_ignore(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    _save(image, np.zeros((512, 512, 3), dtype=np.uint8))
    values = np.zeros((512, 512), dtype=np.uint8)
    values[0, :4] = [1, 3, 4, 255]
    _save(mask, values)
    dataset = ExpertTileDataset([{
        "dataset": "dataset_jacky",
        "source_group": "source",
        "tile": image.name,
        "image": str(image),
        "mask": str(mask),
        "mask_format": "class_index",
    }], raw_ids=(3, 4), train_augmentation=False)
    assert dataset[0]["target"][0, :4].tolist() == [0, 1, 1, 255]


def test_loader_unions_dataset115_binary_masks(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    first, second = tmp_path / "D-01.png", tmp_path / "D-11.png"
    _save(image, np.zeros((512, 512, 3), dtype=np.uint8))
    first_values = np.zeros((512, 512), dtype=np.uint8)
    second_values = np.zeros((512, 512), dtype=np.uint8)
    first_values[0, 0] = 255
    second_values[0, 1] = 255
    _save(first, first_values)
    _save(second, second_values)
    dataset = ExpertTileDataset([{
        "dataset": "dataset115_filtered",
        "source_group": "source",
        "tile": image.name,
        "image": str(image),
        "positive_masks": [str(first), str(second)],
        "mask_format": "binary_multilabel",
    }], raw_ids=(1,), train_augmentation=False)
    assert dataset[0]["target"][0, :3].tolist() == [1, 1, 0]


def test_metadata_apostrophe_path_resolves_exported_underscore(tmp_path: Path) -> None:
    actual = tmp_path / "KJWTomh-SC-M-A7_-1" / "tile.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"image")
    assert _actual_path(tmp_path, "KJWTomh-SC-M-A7'-1/tile.jpg") == actual.resolve()
