"""Thread-safe model lifecycle and segmentation inference orchestration."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from adapters import (
    ConvNextUnetAdapter,
    DASAM3Adapter,
    DummyAdapter,
    ResUNetAdapter,
    SAM2Adapter,
    SAM3Adapter,
    SegmentationAdapter,
)
from config import Settings, settings
from imaging.image_processing import binary_mask_to_pil
from imaging.deterioration_overlay import compose_deterioration_masks
from registry import (
    MODELS,
    HYBRID_MODEL_ID,
    HYBRID_CLASS_ID,
    HYBRID_WEIGHT,
    HYBRID_COMPONENTS,
    InvalidWeightError,
    get_adapter_name,
    get_weight_path,
    resolve_deterioration_class,
)


LOGGER = logging.getLogger(__name__)
AdapterFactory = Callable[[], SegmentationAdapter]
WeightResolver = Callable[[str, str, Path], Path | None]


def device_information() -> dict[str, Any]:
    """Report CUDA availability without preventing CPU-only startup."""

    cuda_available = bool(torch.cuda.is_available())
    return {
        "cuda": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "device": "cuda" if cuda_available else "cpu",
    }


def _default_weight_resolver(
    model_id: str, weight_name: str, model_root: Path
) -> Path | None:
    return get_weight_path(model_id, weight_name, model_root=model_root)


class HybridInferenceError(RuntimeError):
    """A named component failed; no partial hybrid result is returned."""


class InferenceManager:
    """Cache one model/weight pair and serialize inference through one lock."""

    def __init__(
        self,
        *,
        model_root: Path | str | None = None,
        runtime_settings: Settings | None = None,
        adapter_factories: dict[str, AdapterFactory] | None = None,
        weight_resolver: WeightResolver | None = None,
    ) -> None:
        app_settings = runtime_settings or settings
        self.model_root = Path(model_root or app_settings.model_root)
        self._weight_roots = (
            None if model_root is not None else app_settings.model_weight_roots()
        )
        expert_adapters = {"sam2_adapter": SAM2Adapter, "sam3_adapter": SAM3Adapter}
        self._adapter_factories = adapter_factories or {
            "dummy": DummyAdapter,
            **{
                model_id: partial(
                    expert_adapters[definition["adapter"]],
                    settings=app_settings,
                    expert=definition["expert"],
                )
                for model_id, definition in MODELS.items()
                if "expert" in definition
            },
            "da_sam3": lambda: DASAM3Adapter(settings=app_settings),
            "resunet50": lambda: ResUNetAdapter(settings=app_settings),
            "convnext_unet": lambda: ConvNextUnetAdapter(settings=app_settings),
        }
        if weight_resolver is not None:
            self._weight_resolver = weight_resolver
        else:
            self._weight_resolver = lambda model_id, weight_name, root: get_weight_path(
                model_id,
                weight_name,
                model_root=root if self._weight_roots is None else None,
                weight_roots=self._weight_roots,
            )
        self._inference_lock = threading.Lock()
        self._cached_adapter: SegmentationAdapter | None = None
        self._cached_key: tuple[str, str] | None = None

    @property
    def cached_key(self) -> tuple[str, str] | None:
        """Expose the active cache key for diagnostics and tests."""

        return self._cached_key

    def _adapter_key(self, model_id: str) -> str:
        if model_id in self._adapter_factories:
            return model_id
        return get_adapter_name(model_id)

    def _load_adapter(
        self, model_id: str, weight_name: str
    ) -> SegmentationAdapter:
        requested_key = (model_id, weight_name)
        if self._cached_adapter is not None and self._cached_key == requested_key:
            LOGGER.info("Reusing cached model model=%s weight=%s", model_id, weight_name)
            return self._cached_adapter

        weight_path = self._weight_resolver(model_id, weight_name, self.model_root)
        if self._cached_adapter is not None:
            LOGGER.info(
                "Switching model old=%s new=%s", self._cached_key, requested_key
            )
            self._cached_adapter.unload()
            self._cached_adapter = None
            self._cached_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        adapter_key = self._adapter_key(model_id)
        try:
            factory = self._adapter_factories[adapter_key]
        except KeyError as exc:
            raise ValueError(f"No adapter factory registered for {model_id}") from exc
        adapter = factory()
        LOGGER.info("Loading model model=%s weight=%s", model_id, weight_name)
        try:
            adapter.prepare_runtime()
            adapter.load(weight_path)
        except Exception:
            adapter.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        self._cached_adapter = adapter
        self._cached_key = requested_key
        return adapter

    def predict(
        self,
        model_id: str,
        weight_name: str,
        image: Image.Image,
        threshold: float = 0.5,
        deterioration_class: str | None = None,
    ) -> dict[str, Any]:
        """Load or reuse an adapter and return normalized inference outputs."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if model_id in MODELS:
            deterioration_class = resolve_deterioration_class(
                model_id, deterioration_class
            )
        if model_id == HYBRID_MODEL_ID:
            return self.predict_hybrid(
                image, weight_name=weight_name,
                thresholds={item["class"]: threshold for item in HYBRID_COMPONENTS},
            )
        with self._inference_lock:
            return self._predict_locked(
                model_id, weight_name, image, threshold, deterioration_class
            )

    def _predict_locked(
        self, model_id: str, weight_name: str, image: Image.Image,
        threshold: float, deterioration_class: str | None,
    ) -> dict[str, Any]:
        """Run one model while the caller owns the entire inference lock."""

        adapter_class = deterioration_class
        if model_id in MODELS and get_adapter_name(model_id) != "da_sam3":
            adapter_class = None
        started = time.perf_counter()
        LOGGER.info(
            "Inference start model=%s weight=%s class=%s threshold=%.2f",
            model_id,
            weight_name,
            deterioration_class,
            threshold,
        )
        adapter = self._load_adapter(model_id, weight_name)
        with torch.inference_mode():
            if adapter_class is None:
                prediction = adapter.predict(image.convert("RGB"), threshold)
            else:
                prediction = adapter.predict(
                    image.convert("RGB"),
                    threshold,
                    deterioration_class=adapter_class,
                )
        latency_ms = (time.perf_counter() - started) * 1000
        mask = binary_mask_to_pil(prediction["mask"])
        overlay = prediction["overlay"].convert("RGB")
        device = str(device_information()["device"])
        LOGGER.info(
            "Inference success model=%s weight=%s latency_ms=%.2f device=%s",
            model_id,
            weight_name,
            latency_ms,
            device,
        )
        return {
            "mask": mask,
            "overlay": overlay,
            "latency_ms": latency_ms,
            "model": model_id,
            "weight": weight_name,
            "deterioration_class": deterioration_class,
            "device": device,
            "metadata": {
                **prediction.get("metadata", {}),
                "deterioration_class": deterioration_class,
            },
        }

    def predict_hybrid(
        self, image: Image.Image, *, weight_name: str = HYBRID_WEIGHT,
        thresholds: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run the fixed three experts atomically and keep every binary mask."""

        if weight_name != HYBRID_WEIGHT:
            raise InvalidWeightError("Hybrid model requires the three_best weight preset")
        selected = {item["class"]: 0.5 for item in HYBRID_COMPONENTS}
        if thresholds is not None:
            if set(thresholds) - set(selected):
                raise ValueError("Invalid hybrid thresholds")
            try:
                selected.update({key: float(value) for key, value in thresholds.items()})
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid hybrid thresholds") from exc
        if any(not 0 <= value <= 1 for value in selected.values()):
            raise ValueError("Invalid hybrid thresholds")

        with self._inference_lock:
            started = time.perf_counter()
            source = image.convert("RGB")
            # Check every required weight before starting the first expert.
            for component in HYBRID_COMPONENTS:
                try:
                    self._weight_resolver(component["model"], component["weight"], self.model_root)
                except Exception as exc:
                    raise HybridInferenceError(
                        f"Hybrid inference failed for {component['model']}"
                    ) from exc
            components = []
            for component in HYBRID_COMPONENTS:
                try:
                    result = self._predict_locked(
                        component["model"], component["weight"], source,
                        selected[component["class"]], component["class"],
                    )
                except Exception as exc:
                    LOGGER.exception("Hybrid component failed model=%s", component["model"])
                    if self._cached_adapter is not None:
                        self._cached_adapter.unload()
                    self._cached_adapter = None
                    self._cached_key = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise HybridInferenceError(
                        f"Hybrid inference failed for {component['model']}"
                    ) from exc
                # Only masks travel to the UI; per-class overlays are drawn there.
                result.pop("overlay")
                result.update({
                    "class": component["class"], "label": component["label"],
                    "color": component["color"], "threshold": selected[component["class"]],
                })
                components.append(result)
            composed = compose_deterioration_masks(
                source,
                {item["class"]: item["mask"] for item in components},
                {item["class"]: item["color"] for item in components},
            )
            return {
                "model": HYBRID_MODEL_ID, "weight": HYBRID_WEIGHT,
                "deterioration_class": HYBRID_CLASS_ID, "thresholds": selected,
                "device": str(device_information()["device"]),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "mask": composed["mask"], "overlay": composed["overlay"],
                "components": components,
                "metadata": {"overlap_pixels": composed["overlap_pixels"]},
            }

    def unload(self) -> None:
        """Release the currently cached adapter, if any."""

        with self._inference_lock:
            if self._cached_adapter is not None:
                self._cached_adapter.unload()
            self._cached_adapter = None
            self._cached_key = None
