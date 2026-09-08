"""512px two-channel targets and synchronized geometric augmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_SIZE = 512
ALLOWED_RAW_LABELS = frozenset((0, 1, 2, 3, 4, 5, 255))


class DataContractError(ValueError):
    """Raised when an image/mask pair cannot satisfy the approved target contract."""


def make_multilabel_targets(raw_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if raw_mask.ndim != 2:
        raise DataContractError(f"raw mask must be HxW, got {tuple(raw_mask.shape)}")
    values = {int(value) for value in torch.unique(raw_mask).tolist()}
    unknown = values - ALLOWED_RAW_LABELS
    if unknown:
        raise DataContractError(f"raw mask contains unknown labels: {sorted(unknown)}")
    targets = torch.stack(((raw_mask == 1), (raw_mask == 2))).float()
    crack_valid = (raw_mask == 0) | (raw_mask == 1)
    loss_valid = (raw_mask == 0) | (raw_mask == 2)
    valid_masks = torch.stack((crack_valid, loss_valid))
    class_present = (targets.bool() & valid_masks).flatten(1).any(dim=1)
    return targets, valid_masks, class_present


class MonumentDeteriorationDataset(Dataset[dict[str, Any]]):
    """Load immutable dataset members as RGB [0,1] and multi-label masks."""

    def __init__(
        self,
        root: str | Path,
        names: Sequence[str],
        source_groups: dict[str, str],
        *,
        train_augmentation: bool,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.names = tuple(names)
        self.source_groups = source_groups
        self.train_augmentation = train_augmentation
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.names[index]
        image_path = self.root / "images" / name
        mask_path = self.root / "masks" / name
        with Image.open(image_path) as source_image:
            if source_image.mode != "RGB":
                raise DataContractError(f"source image must be RGB: {name} ({source_image.mode})")
            image = np.asarray(source_image, dtype=np.uint8).copy()
        with Image.open(mask_path) as source_mask:
            mask = np.asarray(source_mask, dtype=np.int64).copy()
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise DataContractError(f"image and mask must be 512x512: {name}")
        if self.train_augmentation:
            if self.rng.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if self.rng.random() < 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        targets, valid_masks, class_present = make_multilabel_targets(torch.from_numpy(mask))
        try:
            source_group = self.source_groups[name]
        except KeyError as error:
            raise DataContractError(f"missing source_group for {name}") from error
        return {
            "image": image_tensor,
            "targets": targets,
            "valid_masks": valid_masks,
            "class_present": class_present,
            "image_id": name,
            "source_group": source_group,
        }
