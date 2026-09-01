"""Thread-safe model lifecycle and segmentation inference orchestration."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from adapters import (
    ConvNextUnetAdapter,
    DummyAdapter,
    ResUNetAdapter,
    SAM2Adapter,
    SAM3Adapter,
    SegmentationAdapter,
)
from config import Settings, settings
from imaging.image_processing import binary_mask_to_pil
from registry import get_adapter_name, get_weight_path


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
        self._adapter_factories = adapter_factories or {
            "dummy": DummyAdapter,
            "sam2_adapter": lambda: SAM2Adapter(settings=app_settings),
            "sam3_adapter": lambda: SAM3Adapter(settings=app_settings),
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
    ) -> dict[str, Any]:
        """Load or reuse an adapter and return normalized inference outputs."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        with self._inference_lock:
            started = time.perf_counter()
            LOGGER.info(
                "Inference start model=%s weight=%s threshold=%.2f",
                model_id,
                weight_name,
                threshold,
            )
            adapter = self._load_adapter(model_id, weight_name)
            with torch.inference_mode():
                prediction = adapter.predict(image.convert("RGB"), threshold)
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
                "device": device,
                "metadata": prediction.get("metadata", {}),
            }

    def unload(self) -> None:
        """Release the currently cached adapter, if any."""

        with self._inference_lock:
            if self._cached_adapter is not None:
                self._cached_adapter.unload()
            self._cached_adapter = None
            self._cached_key = None
