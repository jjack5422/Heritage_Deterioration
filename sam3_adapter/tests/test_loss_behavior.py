"""Behavioral checks for expert segmentation losses."""

import torch
import torch.nn.functional as F

from sam3_adapter.losses import per_image_weighted_bce_dice_loss


def test_per_image_dice_gives_each_positive_image_equal_weight() -> None:
    probability = torch.tensor(
        [
            [[[0.9, 0.9, 0.9, 0.9]]],
            [[[0.1, 0.01, 0.01, 0.01]]],
        ],
        dtype=torch.float64,
    )
    logits = torch.logit(probability)
    target = torch.tensor(
        [
            [[[1, 1, 1, 1]]],
            [[[1, 0, 0, 0]]],
        ]
    )

    actual = per_image_weighted_bce_dice_loss(
        logits,
        target,
        ignore_value=255,
        positive_weight=2.0,
        dice_weight=0.65,
    )

    target_float = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits.flatten(),
        target_float.flatten(),
        pos_weight=torch.tensor(2.0, dtype=logits.dtype),
    )
    intersection = (probability * target_float).sum(dim=(1, 2, 3))
    denominator = probability.square().sum(dim=(1, 2, 3)) + target_float.square().sum(dim=(1, 2, 3))
    per_image_dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    expected = bce + 0.65 * per_image_dice.mean()

    torch.testing.assert_close(actual, expected)


def test_empty_targets_use_bce_without_dice() -> None:
    logits = torch.tensor([[[[2.0, -2.0]]], [[[1.0, -1.0]]]])
    target = torch.zeros_like(logits, dtype=torch.long)

    actual = per_image_weighted_bce_dice_loss(
        logits,
        target,
        ignore_value=255,
        positive_weight=2.0,
        dice_weight=0.65,
    )
    expected = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))

    torch.testing.assert_close(actual, expected)
