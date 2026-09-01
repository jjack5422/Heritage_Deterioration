"""Registered segmentation models and safe checkpoint discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from config import settings


ALLOWED_CHECKPOINT_EXTENSIONS = frozenset({".pth", ".pt", ".ckpt"})
BUILT_IN_WEIGHT = "built-in"

MODELS: dict[str, dict[str, Any]] = {
    "dummy": {
        "label": "Dummy Segmentation",
        "weights_subdir": None,
        "adapter": "dummy",
    },
    "sam2_adapter": {
        "label": "SAM2 Adapter",
        "weights_subdir": "sam2_adapter",
        "adapter": "sam2_adapter",
    },
    "sam3_adapter": {
        "label": "SAM3 Adapter",
        "weights_subdir": "sam3_adapter",
        "adapter": "sam3_adapter",
    },
    "resunet50": {
        "label": "ResUNet50",
        "weights_subdir": "resunet50",
        "adapter": "resunet50",
    },
    "convnext_unet": {
        "label": "ConvNeXt-Large U-Net",
        "weights_subdir": "convnext_unet",
        "adapter": "convnext_unet",
    },
}


class RegistryError(ValueError):
    """Base class for safe, user-facing registry errors."""


class UnknownModelError(RegistryError):
    """Raised when a model identifier is not registered."""


class InvalidWeightError(RegistryError):
    """Raised for unsafe, unsupported, or missing checkpoints."""


def _model(model_id: str) -> dict[str, Any]:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise UnknownModelError(f"Unknown model: {model_id}") from exc


def _model_dir(
    model_id: str,
    subdir: str,
    *,
    model_root: Path | str | None,
    weight_roots: Mapping[str, Path | str] | None,
) -> Path:
    configured_roots = weight_roots
    if configured_roots is None and model_root is None:
        configured_roots = settings.model_weight_roots()
    if configured_roots is not None and model_id in configured_roots:
        return Path(configured_roots[model_id]).resolve()
    root = Path(model_root if model_root is not None else settings.model_root)
    return (root / subdir).resolve()


def get_models() -> list[dict[str, str]]:
    """Return public identifiers and labels for all registered models."""

    return [
        {"id": model_id, "label": definition["label"]}
        for model_id, definition in MODELS.items()
    ]


def get_weights(
    model_id: str,
    model_root: Path | str | None = None,
    *,
    weight_roots: Mapping[str, Path | str] | None = None,
) -> list[str]:
    """Return sorted compatible checkpoints for one registered model."""

    definition = _model(model_id)
    subdir = definition["weights_subdir"]
    if subdir is None:
        return [BUILT_IN_WEIGHT]

    model_dir = _model_dir(
        model_id,
        subdir,
        model_root=model_root,
        weight_roots=weight_roots,
    )
    if not model_dir.is_dir():
        return []

    weights: list[str] = []
    for candidate in model_dir.iterdir():
        if candidate.suffix.lower() not in ALLOWED_CHECKPOINT_EXTENSIONS:
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(model_dir)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        if resolved.is_file():
            weights.append(candidate.name)
    return sorted(weights, key=str.casefold)


def get_weight_path(
    model_id: str,
    weight_name: str,
    model_root: Path | str | None = None,
    *,
    weight_roots: Mapping[str, Path | str] | None = None,
) -> Path | None:
    """Resolve a selected checkpoint while preventing path traversal."""

    definition = _model(model_id)
    subdir = definition["weights_subdir"]
    if subdir is None:
        if weight_name != BUILT_IN_WEIGHT:
            raise InvalidWeightError(
                f"Invalid weight for {model_id}: expected {BUILT_IN_WEIGHT}"
            )
        return None

    if not weight_name or Path(weight_name).name != weight_name:
        raise InvalidWeightError("Invalid checkpoint name")
    if Path(weight_name).suffix.lower() not in ALLOWED_CHECKPOINT_EXTENSIONS:
        raise InvalidWeightError("Unsupported checkpoint extension")

    model_dir = _model_dir(
        model_id,
        subdir,
        model_root=model_root,
        weight_roots=weight_roots,
    )
    candidate = (model_dir / weight_name).resolve()
    try:
        candidate.relative_to(model_dir)
    except ValueError as exc:
        raise InvalidWeightError("Checkpoint path escapes the model directory") from exc
    if not candidate.is_file():
        raise InvalidWeightError(f"Checkpoint not found: {weight_name}")
    return candidate


def get_adapter_name(model_id: str) -> str:
    """Return the adapter implementation key for a registered model."""

    return str(_model(model_id)["adapter"])
