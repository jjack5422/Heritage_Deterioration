"""Strict checkpoint contracts shared by the real web model adapters."""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True, slots=True)
class ExpertCheckpoint:
    adaptation: AdaptationCheckpoint
    args: dict[str, Any]
    model_metadata: dict[str, Any]
    expert: str
    input_size: int


def load_expert_checkpoint(
    path: Path, *, expected_expert: str, architecture: str
) -> ExpertCheckpoint:
    """Validate an expert weight and its training-time preprocessing contract."""

    raw_ids = {"scratch_crack": (1,), "loss": (2,), "shrinkage_craquelure": (3, 4)}
    payload = load_checkpoint_payload(path)
    if payload.get("expert") != expected_expert:
        raise CheckpointContractError("Checkpoint expert does not match selected model")
    foreground_ids = payload.get("foreground_raw_ids")
    if (
        not isinstance(foreground_ids, (tuple, list))
        or tuple(foreground_ids) != raw_ids[expected_expert]
    ):
        raise CheckpointContractError("Checkpoint foreground classes do not match expert")
    adaptation = validate_sam3_checkpoint(payload)  # Both expert runs use schema 1.
    config_root = path.parent.parent.parent / "config"
    try:
        args = json.loads((config_root / "args.json").read_text(encoding="utf-8"))
        metadata = json.loads((config_root / "model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointContractError(
            "Expert checkpoint requires config/args.json and model.json"
        ) from exc
    if not isinstance(args, dict) or not isinstance(metadata, dict):
        raise CheckpointContractError("Expert configuration must contain JSON objects")
    if args.get("expert") != expected_expert or metadata.get("expert") != expected_expert:
        raise CheckpointContractError("Run configuration expert does not match checkpoint")
    if args.get("model_input_size") != 1008 or metadata.get("model_input_size") != 1008:
        raise CheckpointContractError("Selected expert requires 1008-pixel model input")
    if metadata.get("source_and_metric_size") != 512:
        raise CheckpointContractError("Selected expert requires 512-pixel source tiles")
    if metadata.get("base_checkpoint_sha256") != adaptation.base_checkpoint_sha256:
        raise CheckpointContractError("Run metadata base checkpoint hash does not match")
    if architecture == "sam2_adapter":
        if metadata.get("backbone_input_size") != 1024:
            raise CheckpointContractError("SAM2 expert requires a 1024-pixel padded backbone")
        adapter_metadata = metadata.get("adapter_metadata")
        if (
            not isinstance(adapter_metadata, dict)
            or not isinstance(args.get("scale_factor"), int)
            or not isinstance(args.get("highpass_rate"), (int, float))
            or args.get("sam2_config") != "configs/sam2.1/sam2.1_hiera_l.yaml"
            or args.get("scale_factor") != adapter_metadata.get("scale_factor")
            or args.get("highpass_rate") != adapter_metadata.get("highpass_rate")
        ):
            raise CheckpointContractError("SAM2 expert adapter configuration does not match")
    return ExpertCheckpoint(adaptation, args, metadata, expected_expert, 1008)


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
