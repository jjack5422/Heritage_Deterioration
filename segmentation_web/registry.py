"""Registered segmentation models and safe checkpoint discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from config import settings


ALLOWED_CHECKPOINT_EXTENSIONS = frozenset({".pth", ".pt", ".ckpt"})
BUILT_IN_WEIGHT = "built-in"
HYBRID_MODEL_ID = "hybrid_three_experts"
HYBRID_CLASS_ID = "three_deteriorations"
HYBRID_WEIGHT = "three_best"
HYBRID_COMPONENTS = (
    {"class": "craquelure", "label": "龜裂", "model": "sam3_adapter_craquelure",
     "weight": "best.pt", "color": (204, 121, 167)},
    {"class": "crack", "label": "裂縫", "model": "sam2_adapter_crack",
     "weight": "best.pt", "color": (0, 114, 178)},
    {"class": "loss", "label": "缺失", "model": "sam3_adapter_loss",
     "weight": "best.pt", "color": (230, 159, 0)},
)
DA_SAM3_CLASSES = (
    {"id": "crack_craquelure", "label": "裂縫／龜裂"},
    {"id": "loss", "label": "缺失"},
)
DETERIORATION_CLASSES = (
    {"id": "craquelure", "label": "龜裂（craquelure）"},
    {"id": "loss", "label": "缺失（loss）"},
    {"id": "crack", "label": "裂縫（crack）"},
    DA_SAM3_CLASSES[0],
    {"id": HYBRID_CLASS_ID, "label": "龜裂＋裂縫＋缺失"},
)
EXPERT_CLASSES = {
    "craquelure": "shrinkage_craquelure",
    "loss": "loss",
    "crack": "scratch_crack",
}

MODELS: dict[str, dict[str, Any]] = {
    "dummy": {
        "label": "Dummy Segmentation",
        "weights_subdir": None,
        "adapter": "dummy",
    },
    **{
        f"{architecture}_{category['id']}": {
            "label": f"{label} {category['id']}",
            "weights_subdir": f"{architecture}_{category['id']}",
            "adapter": architecture,
            "expert": EXPERT_CLASSES[category["id"]],
            "preferred_weight": "best.pt",
            "deterioration_classes": (category,),
        }
        for architecture, label in (
            ("sam3_adapter", "SAM3 Adapter"), ("sam2_adapter", "SAM2 Adapter")
        )
        for category in DETERIORATION_CLASSES[:3]
    },
    "da_sam3": {
        "label": "DA-SAM3",
        "weights_subdir": "da_sam3",
        "adapter": "da_sam3",
        "preferred_weight": "stage2_best.pt",
        "deterioration_classes": DA_SAM3_CLASSES,
    },
    "resunet50": {
        "label": "ResUNet50",
        "weights_subdir": "resunet50",
        "adapter": "resunet50",
        "deterioration_classes": (DETERIORATION_CLASSES[0],),
    },
    "convnext_unet": {
        "label": "ConvNeXt-Large U-Net",
        "weights_subdir": "convnext_unet",
        "adapter": "convnext_unet",
        "deterioration_classes": (DETERIORATION_CLASSES[0],),
    },
    HYBRID_MODEL_ID: {
        "label": "三類混合模型（SAM3 龜裂＋SAM2 裂縫＋SAM3 缺失）",
        "weights_subdir": None,
        "virtual_weight": HYBRID_WEIGHT,
        "adapter": HYBRID_MODEL_ID,
        "deterioration_classes": (DETERIORATION_CLASSES[-1],),
        "components": HYBRID_COMPONENTS,
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


def get_models() -> list[dict[str, Any]]:
    """Return public identifiers and labels for all registered models."""

    models: list[dict[str, Any]] = []
    for model_id, definition in MODELS.items():
        model = {"id": model_id, "label": definition["label"]}
        classes = definition.get("deterioration_classes")
        if classes:
            model["deterioration_classes"] = [dict(item) for item in classes]
        if "components" in definition:
            model["components"] = [dict(item) for item in definition["components"]]
        models.append(model)
    return models


def get_deterioration_classes(model_id: str) -> tuple[dict[str, str], ...]:
    """Return the fixed selectable classes for a model, if it has any."""

    classes = _model(model_id).get("deterioration_classes", ())
    return tuple(dict(item) for item in classes)


def resolve_deterioration_class(
    model_id: str,
    deterioration_class: str | None,
) -> str | None:
    """Validate a model-dependent deterioration-class selection."""

    classes = get_deterioration_classes(model_id)
    if not classes:
        if deterioration_class:
            raise RegistryError(
                "Deterioration class is not supported for selected model"
            )
        return None
    if not deterioration_class:
        if len(classes) == 1:
            return classes[0]["id"]
        raise RegistryError("Deterioration class is required for DA-SAM3")
    allowed = {item["id"] for item in classes}
    if deterioration_class not in allowed:
        raise RegistryError("Invalid deterioration class")
    return deterioration_class


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
        if model_id == HYBRID_MODEL_ID:
            if not all(
                item["weight"] in get_weights(
                    item["model"], model_root=model_root, weight_roots=weight_roots
                )
                for item in HYBRID_COMPONENTS
            ):
                return []
        return [definition.get("virtual_weight", BUILT_IN_WEIGHT)]

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
    preferred = definition.get("preferred_weight")
    return sorted(
        weights,
        key=lambda name: (name != preferred, name.casefold()),
    )


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
        virtual_weight = definition.get("virtual_weight", BUILT_IN_WEIGHT)
        if weight_name != virtual_weight:
            raise InvalidWeightError(
                f"Invalid weight for {model_id}: expected {virtual_weight}"
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
