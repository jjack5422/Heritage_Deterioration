"""Trainable scope and checkpoint contract for decoder calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .checkpoints import expected_adaptation_keys
from .model import DualAdapterSam3


CALIBRATION_EXPERT = "da_sam3_decoder_calibration"
CALIBRATION_CHECKPOINT_TYPE = "da_sam3_decoder_calibration"
ACTIVE_PIXEL_DECODER_STAGE = 1


def _decoder_modules(model: DualAdapterSam3) -> tuple[nn.Module, nn.Module, nn.Module]:
    segmentation_head = getattr(model.sam3, "segmentation_head", None)
    if segmentation_head is None:
        raise RuntimeError("DA-SAM3 has no segmentation head")
    pixel_decoder = getattr(segmentation_head, "pixel_decoder", None)
    if pixel_decoder is None:
        raise RuntimeError("DA-SAM3 segmentation head has no pixel decoder")
    conv_layers = getattr(pixel_decoder, "conv_layers", None)
    norms = getattr(pixel_decoder, "norms", None)
    semantic_head = getattr(segmentation_head, "semantic_seg_head", None)
    if not isinstance(conv_layers, nn.ModuleList) or len(conv_layers) != 3:
        raise RuntimeError(
            "official SAM3 PixelDecoder must expose exactly three convolution stages"
        )
    if not isinstance(norms, nn.ModuleList) or len(norms) != len(conv_layers):
        raise RuntimeError("PixelDecoder convolution/normalization stages do not match")
    if not isinstance(semantic_head, nn.Module):
        raise RuntimeError("segmentation head exposes no semantic_seg_head")
    # The image detector supplies three FPN maps, so PixelDecoder performs two
    # fusion/upsampling iterations and executes indices 0 and 1. Index 2 is
    # declared by the upstream builder but is not reached on this model path.
    return (
        conv_layers[ACTIVE_PIXEL_DECODER_STAGE],
        norms[ACTIVE_PIXEL_DECODER_STAGE],
        semantic_head,
    )


def expected_decoder_trainable_keys(model: DualAdapterSam3) -> tuple[str, ...]:
    """Return the exact six tensors in the boundary-focused calibration scope."""

    target_ids = {
        id(parameter)
        for module in _decoder_modules(model)
        for parameter in module.parameters()
    }
    keys = tuple(
        name for name, parameter in model.named_parameters() if id(parameter) in target_ids
    )
    if len(keys) != 6:
        raise RuntimeError(
            f"decoder calibration expected six parameter tensors, found {len(keys)}: {keys}"
        )
    return keys


def configure_decoder_calibration(model: DualAdapterSam3) -> tuple[str, ...]:
    """Freeze the model, then enable the final PixelDecoder block and semantic head."""

    keys = expected_decoder_trainable_keys(model)
    enabled = set(keys)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in enabled)
        parameter.grad = None
    model._train_stage = "decoder_calibration"
    return keys


def expected_calibration_state_keys(model: DualAdapterSam3) -> tuple[str, ...]:
    """Return source DA parameters plus the decoder-calibration delta."""

    keys = tuple(dict.fromkeys((*expected_adaptation_keys(model), *expected_decoder_trainable_keys(model))))
    if len(keys) != len(set(keys)):
        raise RuntimeError("calibration checkpoint keys are not unique")
    return keys


def calibration_state(model: DualAdapterSam3) -> dict[str, Tensor]:
    named = dict(model.named_parameters())
    return {
        name: named[name].detach().cpu()
        for name in expected_calibration_state_keys(model)
    }


def save_decoder_calibration_checkpoint(
    path: Path,
    model: DualAdapterSam3,
    *,
    epoch: int,
    validation_segmentation_loss: float,
    registry_hash: str,
    split_hash: str,
    source_checkpoint_sha256: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": CALIBRATION_CHECKPOINT_TYPE,
            "model_variant": model.model_variant,
            "stage": "decoder_calibration",
            "epoch": int(epoch),
            "validation_segmentation_loss": float(validation_segmentation_loss),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "trainable_names": list(expected_decoder_trainable_keys(model)),
            "model_state": calibration_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "prompt_contract_sha256": registry_hash,
            "split_contract_sha256": split_hash,
            "model_contract": model.model_contract,
        },
        path,
    )


def _validate_tensor_state(
    state: Mapping[str, Any],
    model: DualAdapterSam3,
) -> dict[str, Tensor]:
    expected = set(expected_calibration_state_keys(model))
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"calibration state keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    named = dict(model.named_parameters())
    validated: dict[str, Tensor] = {}
    for name in sorted(expected):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise RuntimeError(f"calibration state {name!r} is not a tensor")
        parameter = named[name]
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise RuntimeError(
                f"calibration state shape mismatch for {name}: "
                f"checkpoint={tuple(tensor.shape)} model={tuple(parameter.shape)}"
            )
        if tensor.dtype != parameter.dtype:
            raise RuntimeError(
                f"calibration state dtype mismatch for {name}: "
                f"checkpoint={tensor.dtype} model={parameter.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"calibration state contains non-finite tensor: {name}")
        validated[name] = tensor
    return validated


def load_decoder_calibration_checkpoint(
    path: Path,
    model: DualAdapterSam3,
    *,
    expected_prompt_hash: str,
    expected_split_hash: str,
    expected_source_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("schema_version", -1)) != 1:
        raise RuntimeError("unsupported decoder-calibration checkpoint schema")
    if checkpoint.get("checkpoint_type") != CALIBRATION_CHECKPOINT_TYPE:
        raise RuntimeError("checkpoint is not a DA-SAM3 decoder-calibration checkpoint")
    if checkpoint.get("model_variant") != "da_sam3" or model.model_variant != "da_sam3":
        raise RuntimeError("decoder calibration supports only the da_sam3 model variant")
    if checkpoint.get("prompt_contract_sha256") != expected_prompt_hash:
        raise RuntimeError("decoder-calibration prompt contract mismatch")
    if checkpoint.get("split_contract_sha256") != expected_split_hash:
        raise RuntimeError("decoder-calibration split contract mismatch")
    if checkpoint.get("model_contract") != model.model_contract:
        raise RuntimeError("decoder-calibration model contract mismatch")
    if (
        expected_source_checkpoint_sha256 is not None
        and checkpoint.get("source_checkpoint_sha256")
        != expected_source_checkpoint_sha256
    ):
        raise RuntimeError("decoder-calibration source checkpoint mismatch")
    raw_state = checkpoint.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("decoder-calibration model state is missing or malformed")
    state = _validate_tensor_state(raw_state, model)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected decoder-calibration checkpoint keys: {incompatible.unexpected_keys}"
        )
    return checkpoint


__all__ = [
    "CALIBRATION_EXPERT",
    "calibration_state",
    "configure_decoder_calibration",
    "expected_calibration_state_keys",
    "expected_decoder_trainable_keys",
    "load_decoder_calibration_checkpoint",
    "save_decoder_calibration_checkpoint",
]
