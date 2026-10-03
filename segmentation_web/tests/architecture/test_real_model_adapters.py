from __future__ import annotations

import pytest
import torch
import json
from pathlib import Path

from adapters.checkpoint_loading import (
    CheckpointContractError,
    validate_sam2_checkpoint,
    validate_sam3_checkpoint,
    validate_unet_checkpoint,
    load_expert_checkpoint,
)


def _expert_weight(tmp_path: Path) -> Path:
    config_root = tmp_path / "config"
    config_root.mkdir()
    (config_root / "args.json").write_text(json.dumps({
        "expert": "loss", "model_input_size": 1008,
        "sam2_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "scale_factor": 32, "highpass_rate": 0.25,
    }))
    (config_root / "model.json").write_text(json.dumps({
        "expert": "loss", "model_input_size": 1008,
        "source_and_metric_size": 512, "backbone_input_size": 1024,
        "base_checkpoint_sha256": "b" * 64,
        "adapter_metadata": {"scale_factor": 32, "highpass_rate": 0.25},
    }))
    checkpoint = tmp_path / "artifacts/checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({
        "schema_version": 1, "expert": "loss", "foreground_raw_ids": (2,),
        "base_checkpoint_sha256": "b" * 64,
        "adaptation_state": {"decoder.weight": torch.ones(1)},
    }, checkpoint)
    return checkpoint


@pytest.mark.parametrize("architecture", ["sam2_adapter", "sam3_adapter"])
def test_expert_checkpoint_requires_matching_class_and_training_input_contract(tmp_path, architecture):
    path = _expert_weight(tmp_path)
    checkpoint = load_expert_checkpoint(path, expected_expert="loss", architecture=architecture)
    assert checkpoint.input_size == 1008
    with pytest.raises(CheckpointContractError, match="expert"):
        load_expert_checkpoint(path, expected_expert="scratch_crack", architecture=architecture)
    config = tmp_path / "config/args.json"
    args = json.loads(config.read_text())
    args["model_input_size"] = 512
    config.write_text(json.dumps(args))
    with pytest.raises(CheckpointContractError, match="1008"):
        load_expert_checkpoint(path, expected_expert="loss", architecture=architecture)


def test_expert_checkpoint_rejects_missing_configuration_and_wrong_raw_classes(tmp_path):
    path = _expert_weight(tmp_path)
    (tmp_path / "config/args.json").unlink()
    with pytest.raises(CheckpointContractError, match="config"):
        load_expert_checkpoint(path, expected_expert="loss", architecture="sam2_adapter")
    payload = torch.load(path, weights_only=False)
    payload["foreground_raw_ids"] = (1,)
    torch.save(payload, path)
    with pytest.raises(CheckpointContractError, match="foreground classes"):
        load_expert_checkpoint(path, expected_expert="loss", architecture="sam2_adapter")


def test_sam2_checkpoint_contract_accepts_completed_foreground_schema() -> None:
    state = {"decoder.weight": torch.ones(1)}
    validated = validate_sam2_checkpoint(
        {
            "schema_version": 2,
            "task": "foreground",
            "image_size": 512,
            "base_checkpoint_sha256": "a" * 64,
            "sam2_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "adapter": {"scale_factor": 32},
            "adaptation_state": state,
        },
        image_size=512,
    )

    assert validated.adaptation_state is state
    assert validated.base_checkpoint_sha256 == "a" * 64


def test_sam3_checkpoint_contract_requires_schema_hash_and_state() -> None:
    state = {"model.adapter.weight": torch.ones(1)}
    validated = validate_sam3_checkpoint(
        {
            "schema_version": 1,
            "base_checkpoint_sha256": "b" * 64,
            "adaptation_state": state,
        }
    )

    assert validated.adaptation_state is state
    with pytest.raises(CheckpointContractError, match="schema_version"):
        validate_sam3_checkpoint(
            {
                "schema_version": 2,
                "base_checkpoint_sha256": "b" * 64,
                "adaptation_state": state,
            }
        )


@pytest.mark.parametrize(
    ("expected_backbone", "actual_backbone"),
    [("resnet50", "convnext_large"), ("convnext_large", "resnet50")],
)
def test_unet_checkpoint_contract_rejects_wrong_registered_backbone(
    expected_backbone: str,
    actual_backbone: str,
) -> None:
    payload = {
        "args": {
            "backbone": actual_backbone,
            "encoder": actual_backbone,
            "class_names": "background,foreground",
        },
        "class_names": ["background", "foreground"],
        "task": {"name": "foreground", "source_class_ids": (1,)},
        "model": {"segmentation_head.weight": torch.ones(1)},
    }

    with pytest.raises(CheckpointContractError, match="backbone"):
        validate_unet_checkpoint(payload, expected_backbone=expected_backbone)


def test_unet_checkpoint_contract_returns_encoder_and_model_state() -> None:
    state = {"segmentation_head.weight": torch.ones(1)}
    payload = {
        "args": {
            "backbone": "convnext_large",
            "encoder": "tu-convnext_large",
            "class_names": "background,foreground",
        },
        "class_names": ["background", "foreground"],
        "task": {"name": "foreground", "source_class_ids": (1,)},
        "model": state,
    }

    validated = validate_unet_checkpoint(
        payload,
        expected_backbone="convnext_large",
    )

    assert validated.encoder == "tu-convnext_large"
    assert validated.model_state is state
    assert validated.class_names == ("background", "foreground")
