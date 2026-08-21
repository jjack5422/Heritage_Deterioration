"""Unit tests for the H0 native-SAM2 hierarchical segmentation contract."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from sam2_sac.h0_core import (
    HIERARCHY_CLASS_NAMES,
    binary_bce_dice_loss,
    expand_no_prompt_embeddings,
    freeze_except_layer_norm,
    hierarchy_probabilities,
    make_stage_target,
)


def test_hierarchy_probabilities_are_normalized_and_mutually_exclusive() -> None:
    union_logits = torch.tensor([[[[0.0, 1.0]]]])
    type_logits = torch.tensor([[[[0.0, -1.0]]]])

    probabilities = hierarchy_probabilities(union_logits, type_logits)

    assert HIERARCHY_CLASS_NAMES == ("background", "crack", "craquelure")
    assert probabilities.shape == (1, 3, 1, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, 1, 2))
    assert torch.allclose(probabilities[:, 0], 1.0 - torch.sigmoid(union_logits[:, 0]))
    assert torch.allclose(
        probabilities[:, 1], torch.sigmoid(union_logits[:, 0]) * torch.sigmoid(type_logits[:, 0])
    )
    assert torch.allclose(
        probabilities[:, 2],
        torch.sigmoid(union_logits[:, 0]) * (1.0 - torch.sigmoid(type_logits[:, 0])),
    )


def test_stage_target_keeps_only_merged_crack_as_foreground() -> None:
    source = torch.tensor([[[0, 1, 2, 3, 4, 5, 255]]])

    foreground = make_stage_target(source, stage="foreground", ignore_value=255)

    assert foreground.tolist() == [[[0, 1, 255, 255, 255, 255, 255]]]


def test_binary_loss_ignores_unsupervised_pixels() -> None:
    logits = torch.tensor([[[[0.0, 50.0]]]])
    target = torch.tensor([[[[0, 255]]]])

    baseline = binary_bce_dice_loss(logits, target, ignore_value=255, positive_weight=1.0)
    perturbed = binary_bce_dice_loss(
        torch.tensor([[[[0.0, -50.0]]]]),
        target,
        ignore_value=255,
        positive_weight=1.0,
    )

    assert torch.isfinite(baseline)
    assert torch.allclose(baseline, perturbed)


def test_binary_loss_requires_an_explicit_fold_positive_weight() -> None:
    parameter = inspect.signature(binary_bce_dice_loss).parameters["positive_weight"]

    assert parameter.default is inspect.Parameter.empty


def test_binary_loss_applies_the_training_fold_positive_weight_to_bce_only() -> None:
    logits = torch.zeros((1, 1, 1, 2))
    target = torch.tensor([[[[0, 1]]]])

    loss = binary_bce_dice_loss(
        logits,
        target,
        ignore_value=255,
        dice_weight=0.0,
        positive_weight=4.0,
    )

    assert torch.allclose(loss, torch.log(torch.tensor(2.0)) * 2.5)


def test_only_layer_norm_parameters_are_trainable() -> None:
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.LayerNorm((4, 1, 1)), nn.Linear(4, 2))

    names = freeze_except_layer_norm(module)

    assert names == ("1.bias", "1.weight")
    assert all(parameter.requires_grad for name, parameter in module.named_parameters() if name in names)
    assert all(not parameter.requires_grad for name, parameter in module.named_parameters() if name not in names)


def test_no_prompt_embeddings_expand_to_the_image_batch_without_changing_values() -> None:
    sparse = torch.ones((1, 0, 256))
    dense = torch.full((1, 256, 32, 32), 3.0)

    expanded_sparse, expanded_dense = expand_no_prompt_embeddings(sparse, dense, batch_size=4)

    assert expanded_sparse.shape == (4, 0, 256)
    assert expanded_dense.shape == (4, 256, 32, 32)
    assert torch.equal(expanded_dense[0], dense[0])
    assert torch.equal(expanded_dense[-1], dense[0])
