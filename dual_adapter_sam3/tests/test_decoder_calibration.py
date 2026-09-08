from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dual_adapter_sam3.decoder_calibration import (
    calibration_state,
    configure_decoder_calibration,
    expected_calibration_state_keys,
    expected_decoder_trainable_keys,
    load_decoder_calibration_checkpoint,
    save_decoder_calibration_checkpoint,
)
from dual_adapter_sam3.decoder_training import (
    enable_full_pixel_decoder,
    expected_full_pixel_decoder_keys,
)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2)
        self.norm2 = nn.LayerNorm(2)
        self.norm3 = nn.LayerNorm(2)
        self.da_moe = nn.Module()
        self.da_moe.router = nn.Linear(2, 2)
        self.da_moe.experts = nn.ModuleList([nn.Linear(2, 2)])


class _CalibrationModel(nn.Module):
    model_variant = "da_sam3"

    def __init__(self) -> None:
        super().__init__()
        self.model_contract = {"model_variant": "da_sam3", "rank": 8}
        self.sam3 = nn.Module()
        self.sam3.transformer = nn.Module()
        self.sam3.transformer.encoder = nn.Module()
        self.sam3.transformer.encoder.layers = nn.ModuleList([_Layer()])
        self.sam3.segmentation_head = nn.Module()
        self.sam3.segmentation_head.pixel_decoder = nn.Module()
        self.sam3.segmentation_head.pixel_decoder.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(2, 2, 3, padding=1),
                nn.Conv2d(2, 2, 3, padding=1),
                nn.Conv2d(2, 2, 3, padding=1),
            ]
        )
        self.sam3.segmentation_head.pixel_decoder.norms = nn.ModuleList(
            [nn.GroupNorm(1, 2), nn.GroupNorm(1, 2), nn.GroupNorm(1, 2)]
        )
        self.sam3.segmentation_head.semantic_seg_head = nn.Conv2d(2, 1, 1)
        self.sam3.segmentation_head.instance_seg_head = nn.Conv2d(2, 2, 1)
        self.sam3.frozen_weight = nn.Parameter(torch.ones(2))


def _optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    return optimizer, scheduler


def test_decoder_calibration_trains_only_last_spatial_block_and_semantic_head() -> None:
    model = _CalibrationModel()

    names = configure_decoder_calibration(model)

    assert names == expected_decoder_trainable_keys(model)
    assert len(names) == 6
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in model.named_parameters()
    )
    assert any("conv_layers.1" in name for name in names)
    assert any("norms.1" in name for name in names)
    assert any("semantic_seg_head" in name for name in names)
    assert not any(
        "conv_layers.0" in name
        or "conv_layers.2" in name
        or "instance_seg_head" in name
        for name in names
    )


def test_full_decoder_scope_trains_both_executed_stages_and_semantic_head() -> None:
    model = _CalibrationModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    names = enable_full_pixel_decoder(model)

    assert names == expected_full_pixel_decoder_keys(model)
    assert len(names) == 10
    assert any("conv_layers.0" in name for name in names)
    assert any("conv_layers.1" in name for name in names)
    assert any("norms.0" in name for name in names)
    assert any("norms.1" in name for name in names)
    assert any("semantic_seg_head" in name for name in names)
    assert not any("conv_layers.2" in name or "norms.2" in name for name in names)
    assert not any("instance_seg_head" in name for name in names)
    assert all(dict(model.named_parameters())[name].requires_grad for name in names)


def test_calibration_state_contains_source_adaptation_and_decoder_delta() -> None:
    model = _CalibrationModel()
    configure_decoder_calibration(model)

    state = calibration_state(model)

    assert set(state) == set(expected_calibration_state_keys(model))
    assert any("da_moe" in name for name in state)
    assert set(expected_decoder_trainable_keys(model)).issubset(state)
    assert not any("instance_seg_head" in name or "frozen_weight" in name for name in state)


def test_calibration_checkpoint_round_trip_and_exact_key_validation(tmp_path: Path) -> None:
    source = _CalibrationModel()
    configure_decoder_calibration(source)
    optimizer, scheduler = _optimizer_scheduler(source)
    path = tmp_path / "best.pt"
    save_decoder_calibration_checkpoint(
        path,
        source,
        epoch=3,
        validation_segmentation_loss=0.2,
        registry_hash="prompt",
        split_hash="split",
        source_checkpoint_sha256="source-sha",
        optimizer=optimizer,
        scheduler=scheduler,
    )

    target = _CalibrationModel()
    for parameter in target.parameters():
        parameter.data.zero_()
    loaded = load_decoder_calibration_checkpoint(
        path,
        target,
        expected_prompt_hash="prompt",
        expected_split_hash="split",
        expected_source_checkpoint_sha256="source-sha",
    )

    assert loaded["checkpoint_type"] == "da_sam3_decoder_calibration"
    for name, tensor in calibration_state(target).items():
        torch.testing.assert_close(tensor, loaded["model_state"][name])

    malformed = dict(loaded)
    malformed["model_state"] = dict(loaded["model_state"])
    malformed["model_state"].pop(next(iter(malformed["model_state"])))
    bad_path = tmp_path / "bad.pt"
    torch.save(malformed, bad_path)
    with pytest.raises(RuntimeError, match="keys mismatch"):
        load_decoder_calibration_checkpoint(
            bad_path,
            _CalibrationModel(),
            expected_prompt_hash="prompt",
            expected_split_hash="split",
            expected_source_checkpoint_sha256="source-sha",
        )
