"""Unit tests for the native-SAM2 binary segmentation contract."""

from __future__ import annotations

import torch
from torch import nn

from sam2_adapter.h0_core import (
    binary_bce_dice_loss,
    expand_no_prompt_embeddings,
    freeze_except_layer_norm,
)


def test_binary_loss_ignores_unsupervised_pixels() -> None:
    logits = torch.tensor([[[[0.0, 50.0]]]])
    target = torch.tensor([[[[0, 255]]]])

    baseline = binary_bce_dice_loss(logits, target, ignore_value=255)
    perturbed = binary_bce_dice_loss(
        torch.tensor([[[[0.0, -50.0]]]]), target, ignore_value=255
    )

    assert torch.isfinite(baseline)
    assert torch.allclose(baseline, perturbed)


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
