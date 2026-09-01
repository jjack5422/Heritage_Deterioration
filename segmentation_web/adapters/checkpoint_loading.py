"""Strict checkpoint contracts shared by the real web model adapters."""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class CheckpointContractError(RuntimeError):
    """Raised when a checkpoint is incompatible with its registered model."""


@dataclass(frozen=True, slots=True)
class AdaptationCheckpoint:
    adaptation_state: dict[str, Tensor]
    base_checkpoint_sha256: str
    schema_version: int
    image_size: int | None = None
    model_config: str | None = None
    adapter_metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class UnetCheckpoint:
    model_state: dict[str, Tensor]
    backbone: str
    encoder: str
    class_names: tuple[str, ...]


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Read one task checkpoint on CPU and require a dictionary payload."""

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointContractError(f"Unable to read checkpoint: {path.name}") from exc
    if not isinstance(payload, dict):
        raise CheckpointContractError("checkpoint payload must be a dictionary")
    return payload


def sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint without loading its tensors."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointContractError(f"Unable to read base checkpoint: {path}") from exc
    return digest.hexdigest()


def verify_base_checkpoint(path: Path, expected_sha256: str) -> None:
    """Require an existing base checkpoint with the recorded training digest."""

    if not path.is_file():
        raise CheckpointContractError(f"Base checkpoint not found: {path}")
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise CheckpointContractError(
            "base checkpoint SHA-256 does not match the task checkpoint"
        )


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointContractError("base_checkpoint_sha256 must be a SHA-256 digest")
    if any(character not in string.hexdigits for character in value):
        raise CheckpointContractError("base_checkpoint_sha256 must be a SHA-256 digest")
    return value.lower()


def _tensor_state(payload: dict[str, Any], key: str) -> dict[str, Tensor]:
    state = payload.get(key)
    if not isinstance(state, dict) or not state:
        raise CheckpointContractError(f"checkpoint {key} must be a non-empty dictionary")
    invalid = [
        name
        for name, value in state.items()
        if not isinstance(name, str) or not isinstance(value, Tensor)
    ]
    if invalid:
        raise CheckpointContractError(f"checkpoint {key} contains non-tensor entries")
    return state


def validate_sam2_checkpoint(
    payload: dict[str, Any],
    *,
    image_size: int,
) -> AdaptationCheckpoint:
    """Validate the completed binary SAM2-Adapter checkpoint schema."""

    if payload.get("schema_version") != 2:
        raise CheckpointContractError("SAM2 checkpoint schema_version must be 2")
    if payload.get("task") != "foreground":
        raise CheckpointContractError("SAM2 checkpoint task must be foreground")
    if payload.get("image_size") != image_size:
        raise CheckpointContractError(
            f"SAM2 checkpoint image_size must be {image_size}"
        )
    config = payload.get("sam2_config")
    if not isinstance(config, str) or not config:
        raise CheckpointContractError("SAM2 checkpoint sam2_config is missing")
    adapter = payload.get("adapter")
    if not isinstance(adapter, dict) or not adapter:
        raise CheckpointContractError("SAM2 checkpoint adapter metadata is missing")
    return AdaptationCheckpoint(
        adaptation_state=_tensor_state(payload, "adaptation_state"),
        base_checkpoint_sha256=_sha256(payload.get("base_checkpoint_sha256")),
        schema_version=2,
        image_size=image_size,
        model_config=config,
        adapter_metadata=adapter,
    )


def validate_sam3_checkpoint(payload: dict[str, Any]) -> AdaptationCheckpoint:
    """Validate the completed SAM3-Adapter task checkpoint schema."""

    if payload.get("schema_version") != 1:
        raise CheckpointContractError("SAM3 checkpoint schema_version must be 1")
    return AdaptationCheckpoint(
        adaptation_state=_tensor_state(payload, "adaptation_state"),
        base_checkpoint_sha256=_sha256(payload.get("base_checkpoint_sha256")),
        schema_version=1,
    )


def _class_names(payload: dict[str, Any], args: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("class_names", args.get("class_names"))
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        names = tuple(str(part) for part in value)
    else:
        names = ()
    if names != ("background", "foreground"):
        raise CheckpointContractError(
            "U-Net checkpoint classes must be background,foreground"
        )
    return names


def validate_unet_checkpoint(
    payload: dict[str, Any],
    *,
    expected_backbone: str,
) -> UnetCheckpoint:
    """Validate a binary foreground U-Net and its registered backbone."""

    args = payload.get("args")
    if not isinstance(args, dict):
        raise CheckpointContractError("U-Net checkpoint args are missing")
    backbone = args.get("backbone")
    if backbone != expected_backbone:
        raise CheckpointContractError(
            f"U-Net checkpoint backbone must be {expected_backbone}, got {backbone}"
        )
    encoder = args.get("encoder")
    if not isinstance(encoder, str) or not encoder:
        raise CheckpointContractError("U-Net checkpoint encoder is missing")
    task = payload.get("task")
    if not isinstance(task, dict) or task.get("name") != "foreground":
        raise CheckpointContractError("U-Net checkpoint task must be foreground")
    return UnetCheckpoint(
        model_state=_tensor_state(payload, "model"),
        backbone=expected_backbone,
        encoder=encoder,
        class_names=_class_names(payload, args),
    )
