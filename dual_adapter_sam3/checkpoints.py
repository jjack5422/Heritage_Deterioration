"""Strict, training-independent DA-SAM3 adaptation checkpoint handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .decoder_training import expected_full_pixel_decoder_keys
from .model import DualAdapterSam3


def is_adaptation_parameter(name: str, *, model_variant: str) -> bool:
    """Return whether a parameter belongs to the saved adaptation state."""

    is_expert = ".da_moe.experts." in name
    is_router = ".da_moe.router." in name
    is_fusion_norm = ".transformer.encoder.layers." in name and any(
        f".{norm}." in name for norm in ("norm1", "norm2", "norm3")
    )
    is_visual_adapter = (
        ".visual_adapter." in name and model_variant == "visual_da_sam3"
    )
    return is_expert or is_router or is_fusion_norm or is_visual_adapter


def expected_adaptation_keys(model: DualAdapterSam3) -> tuple[str, ...]:
    """Return the exact checkpoint key contract for one model variant."""

    model_variant = str(model.model_variant)
    if model_variant not in ("da_sam3", "visual_da_sam3"):
        raise RuntimeError(
            f"unsupported model variant for adaptation state: {model_variant!r}"
        )
    keys = tuple(
        name
        for name, _parameter in model.named_parameters()
        if is_adaptation_parameter(name, model_variant=model_variant)
    )
    if bool(getattr(model, "full_pixel_decoder", False)):
        keys = tuple(dict.fromkeys((*keys, *expected_full_pixel_decoder_keys(model))))
    if not keys:
        raise RuntimeError(f"{model_variant} model exposes no adaptation parameters")
    if model_variant == "visual_da_sam3" and not any(
        ".visual_adapter." in name for name in keys
    ):
        raise RuntimeError(
            "visual_da_sam3 model exposes no Visual Adapter parameters"
        )
    return keys


def adaptation_state(model: DualAdapterSam3) -> dict[str, Tensor]:
    """Copy only adaptation parameters into a CPU checkpoint state."""

    expected = expected_adaptation_keys(model)
    named = dict(model.named_parameters())
    return {name: named[name].detach().cpu() for name in expected}


def checkpoint_model_variant(checkpoint: Mapping[str, Any]) -> str:
    """Resolve and validate the model variant encoded by a checkpoint."""

    schema_version = int(checkpoint.get("schema_version", -1))
    if schema_version == 1:
        variant = str(checkpoint.get("model_variant", "da_sam3"))
        if variant != "da_sam3":
            raise RuntimeError(
                f"schema 1 checkpoint cannot declare model variant {variant!r}"
            )
        return variant
    if schema_version == 2:
        if "model_variant" not in checkpoint:
            raise RuntimeError("schema 2 checkpoint is missing model_variant")
        variant = str(checkpoint["model_variant"])
        if variant not in ("da_sam3", "visual_da_sam3"):
            raise RuntimeError(
                f"schema 2 checkpoint has unsupported model variant {variant!r}"
            )
        return variant
    raise RuntimeError(
        f"unsupported adaptation checkpoint schema version: {schema_version}"
    )


def _normalized_checkpoint_model_contract(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_variant: str,
) -> dict[str, Any]:
    raw = checkpoint.get("model_contract")
    if not isinstance(raw, Mapping):
        raise RuntimeError("checkpoint model contract is missing or malformed")
    contract = dict(raw)
    if int(checkpoint["schema_version"]) == 1 and checkpoint_variant == "da_sam3":
        contract.setdefault("model_variant", "da_sam3")
    return contract


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    model: DualAdapterSam3,
    *,
    expected_prompt_hash: str,
    expected_split_hash: str,
) -> dict[str, Tensor]:
    """Validate variant, data contracts, keys, shapes, dtypes, and finiteness."""

    checkpoint_variant = checkpoint_model_variant(checkpoint)
    requested_variant = str(model.model_variant)
    if checkpoint_variant != requested_variant:
        raise RuntimeError(
            f"checkpoint model variant {checkpoint_variant!r} does not match requested "
            f"model variant {requested_variant!r}"
        )
    if checkpoint.get("prompt_contract_sha256") != expected_prompt_hash:
        raise RuntimeError("checkpoint prompt contract mismatch")
    if checkpoint.get("split_contract_sha256") != expected_split_hash:
        raise RuntimeError("checkpoint split contract mismatch")
    saved_contract = _normalized_checkpoint_model_contract(
        checkpoint,
        checkpoint_variant=checkpoint_variant,
    )
    if saved_contract != model.model_contract:
        raise RuntimeError("checkpoint model contract mismatch")

    raw_state = checkpoint.get("adaptation_state")
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("checkpoint adaptation state is missing or malformed")
    state = dict(raw_state)
    expected_keys = set(expected_adaptation_keys(model))
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"adaptation state keys mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    named_parameters = dict(model.named_parameters())
    for name in sorted(expected_keys):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise RuntimeError(f"adaptation state {name!r} is not a tensor")
        parameter = named_parameters[name]
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise RuntimeError(
                f"adaptation state shape mismatch for {name}: "
                f"checkpoint={tuple(tensor.shape)} model={tuple(parameter.shape)}"
            )
        if tensor.dtype != parameter.dtype:
            raise RuntimeError(
                f"adaptation state dtype mismatch for {name}: "
                f"checkpoint={tensor.dtype} model={parameter.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"adaptation state contains non-finite tensor: {name}"
            )
    return state


def load_adaptation_checkpoint(
    path: Path,
    model: DualAdapterSam3,
    *,
    expected_prompt_hash: str,
    expected_split_hash: str,
) -> dict[str, Any]:
    """Load a validated adaptation checkpoint without training dependencies."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = validate_checkpoint_contract(
        checkpoint,
        model,
        expected_prompt_hash=expected_prompt_hash,
        expected_split_hash=expected_split_hash,
    )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected adaptation checkpoint keys: {unexpected}")
    return checkpoint
