from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from dual_adapter_sam3.train import (
    _adaptation_state,
    _expected_adaptation_keys,
    _load_adaptation,
    _save_checkpoint,
)


class _CheckpointLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2)
        self.norm2 = nn.LayerNorm(2)
        self.norm3 = nn.LayerNorm(2)
        self.da_moe = nn.Module()
        self.da_moe.router = nn.Linear(2, 2)
        self.da_moe.experts = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.base_ffn = nn.Linear(2, 2)


class _CheckpointModel(nn.Module):
    def __init__(self, model_variant: str) -> None:
        super().__init__()
        self.model_variant = model_variant
        self.model_contract = {
            "model_variant": model_variant,
            "checkpoint_sha256": "official-checkpoint",
            "rank": 8,
        }
        self.sam3 = nn.Module()
        self.sam3.transformer = nn.Module()
        self.sam3.transformer.encoder = nn.Module()
        self.sam3.transformer.encoder.layers = nn.ModuleList([_CheckpointLayer()])
        self.sam3.backbone = nn.Module()
        self.sam3.backbone.vision_backbone = nn.Module()
        self.sam3.backbone.vision_backbone.trunk = nn.Module()
        self.sam3.backbone.vision_backbone.trunk.base_weight = nn.Parameter(torch.ones(2))
        if model_variant == "visual_da_sam3":
            self.sam3.backbone.vision_backbone.trunk.visual_adapter = nn.Sequential(
                nn.Linear(2, 2),
                nn.GELU(),
                nn.Linear(2, 2),
            )
        self.sam3.decoder_weight = nn.Parameter(torch.ones(2))


def _optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _save(path: Path, model: _CheckpointModel) -> dict:
    optimizer, scheduler = _optimizer_scheduler(model)
    _save_checkpoint(
        path,
        model,
        stage="stage1",
        epoch=3,
        validation_segmentation_loss=0.25,
        registry_hash="prompt-hash",
        split_hash="split-hash",
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return torch.load(path, map_location="cpu", weights_only=True)


def test_adaptation_state_is_exactly_variant_specific() -> None:
    legacy = _CheckpointModel("da_sam3")
    visual = _CheckpointModel("visual_da_sam3")

    legacy_keys = set(_adaptation_state(legacy))
    visual_keys = set(_adaptation_state(visual))

    assert legacy_keys == set(_expected_adaptation_keys(legacy))
    assert visual_keys == set(_expected_adaptation_keys(visual))
    assert not any("visual_adapter" in key for key in legacy_keys)
    assert any("visual_adapter" in key for key in visual_keys)
    assert not any("base_ffn" in key or "base_weight" in key or "decoder_weight" in key for key in visual_keys)


@pytest.mark.parametrize("model_variant", ["da_sam3", "visual_da_sam3"])
def test_schema2_checkpoint_round_trip(tmp_path: Path, model_variant: str) -> None:
    source = _CheckpointModel(model_variant)
    checkpoint_path = tmp_path / f"{model_variant}.pt"
    checkpoint = _save(checkpoint_path, source)

    assert checkpoint["schema_version"] == 2
    assert checkpoint["model_variant"] == model_variant
    target = _CheckpointModel(model_variant)
    for name, parameter in target.named_parameters():
        if name in checkpoint["adaptation_state"]:
            parameter.data.zero_()
    loaded = _load_adaptation(
        checkpoint_path,
        target,
        expected_prompt_hash="prompt-hash",
        expected_split_hash="split-hash",
    )

    assert loaded["model_variant"] == model_variant
    for name, tensor in _adaptation_state(target).items():
        torch.testing.assert_close(tensor, checkpoint["adaptation_state"][name])


def test_schema1_without_variant_loads_only_legacy(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    checkpoint = _save(path, _CheckpointModel("da_sam3"))
    checkpoint["schema_version"] = 1
    checkpoint.pop("model_variant")
    checkpoint["model_contract"].pop("model_variant")
    torch.save(checkpoint, path)

    loaded = _load_adaptation(
        path,
        _CheckpointModel("da_sam3"),
        expected_prompt_hash="prompt-hash",
        expected_split_hash="split-hash",
    )
    assert loaded["schema_version"] == 1

    with pytest.raises(RuntimeError, match="da_sam3.*visual_da_sam3"):
        _load_adaptation(
            path,
            _CheckpointModel("visual_da_sam3"),
            expected_prompt_hash="prompt-hash",
            expected_split_hash="split-hash",
        )


@pytest.mark.parametrize("source_variant,target_variant", [("da_sam3", "visual_da_sam3"), ("visual_da_sam3", "da_sam3")])
def test_checkpoint_rejects_variant_mismatch(
    tmp_path: Path,
    source_variant: str,
    target_variant: str,
) -> None:
    path = tmp_path / "mismatch.pt"
    _save(path, _CheckpointModel(source_variant))
    with pytest.raises(RuntimeError, match=f"{source_variant}.*{target_variant}"):
        _load_adaptation(
            path,
            _CheckpointModel(target_variant),
            expected_prompt_hash="prompt-hash",
            expected_split_hash="split-hash",
        )


@pytest.mark.parametrize("mutation,match", [("missing", "missing"), ("unexpected", "unexpected"), ("shape", "shape")])
def test_checkpoint_rejects_inexact_adaptation_state_without_partial_load(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    path = tmp_path / f"{mutation}.pt"
    checkpoint = _save(path, _CheckpointModel("visual_da_sam3"))
    first_key = sorted(checkpoint["adaptation_state"])[0]
    if mutation == "missing":
        checkpoint["adaptation_state"].pop(first_key)
    elif mutation == "unexpected":
        checkpoint["adaptation_state"]["unexpected.weight"] = torch.ones(1)
    else:
        checkpoint["adaptation_state"][first_key] = torch.ones(99)
    torch.save(checkpoint, path)

    target = _CheckpointModel("visual_da_sam3")
    before = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}
    with pytest.raises(RuntimeError, match=match):
        _load_adaptation(
            path,
            target,
            expected_prompt_hash="prompt-hash",
            expected_split_hash="split-hash",
        )
    after = dict(target.named_parameters())
    for name, tensor in before.items():
        torch.testing.assert_close(after[name], tensor)


@pytest.mark.parametrize("field", ["prompt", "split", "model"])
def test_checkpoint_rejects_contract_mismatch(tmp_path: Path, field: str) -> None:
    path = tmp_path / f"{field}.pt"
    checkpoint = _save(path, _CheckpointModel("da_sam3"))
    if field == "prompt":
        checkpoint["prompt_contract_sha256"] = "other"
    elif field == "split":
        checkpoint["split_contract_sha256"] = "other"
    else:
        checkpoint["model_contract"] = deepcopy(checkpoint["model_contract"])
        checkpoint["model_contract"]["rank"] = 99
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="contract"):
        _load_adaptation(
            path,
            _CheckpointModel("da_sam3"),
            expected_prompt_hash="prompt-hash",
            expected_split_hash="split-hash",
        )
