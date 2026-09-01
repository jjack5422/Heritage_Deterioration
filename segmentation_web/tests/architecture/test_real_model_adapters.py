from __future__ import annotations

import pytest
import torch

from adapters.checkpoint_loading import (
    CheckpointContractError,
    validate_sam2_checkpoint,
    validate_sam3_checkpoint,
    validate_unet_checkpoint,
)


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
