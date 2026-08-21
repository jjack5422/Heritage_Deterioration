"""Behavioral contract for binary merged-crack SAM2-Adapter training."""

from __future__ import annotations

import torch
from torch import nn

from sam2_adapter.adapter_core import (
    StageAdapterBank,
    high_pass_filter,
    make_binary_target,
)
from sam2_adapter.adapter_model import AdapterHieraTrunk, configure_adapter_training


class _FakePatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).permute(0, 2, 3, 1)


class _FakeBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.projection = nn.Linear(dim, dim_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.dim != self.dim_out:
            tokens = tokens[:, ::2, ::2]
        return self.projection(tokens)


class _FakeHiera(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed()
        self.blocks = nn.ModuleList(
            (_FakeBlock(8, 8), _FakeBlock(8, 16), _FakeBlock(16, 16))
        )
        self.stage_ends = [0, 2]
        self.return_interm_layers = True
        self.channel_list = [16, 8]

    def _get_pos_embed(self, spatial_shape: tuple[int, int]) -> torch.Tensor:
        return torch.zeros((1, *spatial_shape, 8))


def test_binary_target_scores_only_background_and_merged_crack() -> None:
    source = torch.tensor([[[0, 1, 2, 5, 255]]])

    target = make_binary_target(source, ignore_value=255)

    assert target.tolist() == [[[0, 1, 255, 255, 255]]]


def test_high_pass_filter_removes_a_constant_image_without_changing_shape() -> None:
    image = torch.full((2, 3, 32, 32), 0.75, dtype=torch.float32)

    filtered = high_pass_filter(image, rate=0.25)

    assert filtered.shape == image.shape
    assert filtered.dtype == torch.float32
    assert torch.isfinite(filtered).all()
    assert float(filtered.abs().max()) < 1e-5


def test_stage_adapter_bank_matches_each_hiera_stage_and_block_shape() -> None:
    adapters = StageAdapterBank(
        stage_dims=(8, 16),
        block_stage_indices=(0, 0, 1),
        scale_factor=4,
    )
    images = torch.randn(2, 3, 32, 32)
    handcrafted = adapters.prepare_handcrafted(images)
    stage0 = torch.randn(2, 8, 8, 8)
    stage1 = torch.randn(2, 4, 4, 16)

    prompt0 = adapters.begin_stage(stage0, handcrafted[0], stage_index=0)
    injected0 = adapters.inject(stage0, prompt0, block_index=0)
    prompt1 = adapters.begin_stage(stage1, handcrafted[1], stage_index=1)
    injected1 = adapters.inject(stage1, prompt1, block_index=2)

    assert handcrafted[0].shape == (2, 8, 8, 2)
    assert handcrafted[1].shape == (2, 4, 4, 4)
    assert injected0.shape == stage0.shape
    assert injected1.shape == stage1.shape


def test_adapter_hiera_trunk_preserves_native_multiscale_outputs() -> None:
    trunk = AdapterHieraTrunk(_FakeHiera(), scale_factor=4)

    outputs = trunk(torch.randn(2, 3, 32, 32))

    assert [tuple(output.shape) for output in outputs] == [
        (2, 8, 8, 8),
        (2, 16, 4, 4),
    ]
    assert trunk.channel_list == [16, 8]


def test_training_configuration_exposes_only_adapters_and_mask_decoder() -> None:
    class FakeSAM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = nn.Module()
            self.image_encoder.trunk = AdapterHieraTrunk(_FakeHiera(), scale_factor=4)
            self.sam_mask_decoder = nn.Module()
            self.sam_mask_decoder.mask_tokens = nn.Linear(16, 1)
            self.sam_mask_decoder.iou_prediction_head = nn.Linear(16, 1)
            self.sam_mask_decoder.pred_obj_score_head = nn.Linear(16, 1)
            self.sam_prompt_encoder = nn.Linear(4, 4)
            self.memory_encoder = nn.Linear(4, 4)

    sam = FakeSAM()

    names = configure_adapter_training(sam)

    assert names
    assert all(
        name.startswith("image_encoder.trunk.adapters.")
        or name.startswith("sam_mask_decoder.mask_tokens.")
        for name in names
    )
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in sam.named_parameters()
    )
