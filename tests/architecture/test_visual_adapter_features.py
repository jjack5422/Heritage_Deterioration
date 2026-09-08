from __future__ import annotations

import pytest
import torch

from dual_adapter_sam3.visual_adapter import (
    HandcraftedFeaturePyramid,
    VisualAdapterBank,
    VisualAdapterConfig,
    fft_high_pass,
)


def test_fft_high_pass_suppresses_constant_and_preserves_high_frequency() -> None:
    constant = torch.ones(2, 3, 32, 32, dtype=torch.float16)
    filtered = fft_high_pass(constant, area_ratio=0.25)

    assert filtered.shape == constant.shape
    assert filtered.dtype == torch.float32
    assert filtered.device == constant.device
    assert torch.count_nonzero(filtered) == 0

    checkerboard = torch.arange(32).view(1, 1, 1, 32) % 2
    checkerboard = checkerboard.expand(1, 3, 32, 32).float()
    response = fft_high_pass(checkerboard, area_ratio=0.25)
    assert torch.isfinite(response).all()
    assert torch.count_nonzero(response) > 0


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_fft_high_pass_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="area_ratio"):
        fft_high_pass(torch.zeros(1, 3, 16, 16), area_ratio=ratio)


@pytest.mark.parametrize(
    "images,match",
    [
        (torch.zeros(3, 16, 16), r"\[B,3,H,W\]"),
        (torch.zeros(1, 1, 16, 16), r"\[B,3,H,W\]"),
        (torch.full((1, 3, 16, 16), float("nan")), "finite"),
    ],
)
def test_fft_high_pass_rejects_invalid_images(images: torch.Tensor, match: str) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=match):
        fft_high_pass(images, area_ratio=0.25)


def test_handcrafted_pyramid_has_fixed_four_stage_shapes() -> None:
    pyramid = HandcraftedFeaturePyramid(VisualAdapterConfig())
    with torch.no_grad():
        outputs = pyramid(torch.rand(2, 3, 512, 512))

    assert [tuple(item.shape) for item in outputs] == [
        (2, 32, 128, 128),
        (2, 32, 64, 64),
        (2, 32, 32, 32),
        (2, 32, 16, 16),
    ]
    assert all(torch.isfinite(item).all() for item in outputs)


def test_visual_adapter_is_zero_initialized_identity_with_fixed_stage_mapping() -> None:
    bank = VisualAdapterBank(VisualAdapterConfig())
    images = torch.rand(1, 3, 32, 32)
    pyramid = bank.prepare(images)
    tokens = torch.randn(1, 2, 2, 1024)

    assert [bank.stage_for_block(index) for index in range(32)] == [
        *(0 for _ in range(8)),
        *(1 for _ in range(8)),
        *(2 for _ in range(8)),
        *(3 for _ in range(8)),
    ]
    for projection in bank.stage_up_projections:
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0

    for block_index in range(32):
        actual = bank.inject(tokens, pyramid, block_index=block_index)
        torch.testing.assert_close(actual, tokens, rtol=0, atol=0)


def test_visual_adapter_second_backward_reaches_every_parameter_tensor() -> None:
    torch.manual_seed(7)
    bank = VisualAdapterBank(VisualAdapterConfig())
    optimizer = torch.optim.SGD(bank.parameters(), lr=0.1)
    images = torch.rand(1, 3, 32, 32)
    base_tokens = torch.randn(1, 2, 2, 1024)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        pyramid = bank.prepare(images)
        tokens = base_tokens
        for block_index in range(32):
            tokens = bank.inject(tokens, pyramid, block_index=block_index)
        tokens.square().mean().backward()
        optimizer.step()

    step()
    step()

    missing = []
    zero = []
    for name, parameter in bank.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            missing.append(name)
        elif torch.count_nonzero(parameter.grad) == 0:
            zero.append(name)
    assert missing == []
    assert zero == []


@pytest.mark.parametrize("block_index", [-1, 32])
def test_visual_adapter_rejects_invalid_block_index(block_index: int) -> None:
    bank = VisualAdapterBank(VisualAdapterConfig())
    pyramid = bank.prepare(torch.rand(1, 3, 32, 32))
    with pytest.raises(IndexError, match="block_index"):
        bank.inject(torch.rand(1, 2, 2, 1024), pyramid, block_index=block_index)


def test_visual_adapter_rejects_bad_tokens_and_nonfinite_intermediates() -> None:
    bank = VisualAdapterBank(VisualAdapterConfig())
    pyramid = bank.prepare(torch.rand(1, 3, 32, 32))

    with pytest.raises(ValueError, match=r"\[B,H,W,1024\]"):
        bank.inject(torch.rand(1, 4, 1024), pyramid, block_index=0)
    with pytest.raises(ValueError, match=r"\[B,H,W,1024\]"):
        bank.inject(torch.rand(1, 2, 2, 32), pyramid, block_index=0)

    corrupted = list(pyramid)
    corrupted[0] = torch.full_like(corrupted[0], float("inf"))
    with pytest.raises(FloatingPointError, match="finite"):
        bank.inject(torch.rand(1, 2, 2, 1024), tuple(corrupted), block_index=0)
