from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from adapters.base import SegmentationAdapter
from inference import InferenceManager


class RecordingAdapter(SegmentationAdapter):
    def __init__(self, tracker: dict[str, int], delay: float = 0.0) -> None:
        self.tracker = tracker
        self.delay = delay

    def load(self, weight_path: Path | None) -> None:
        self.tracker["loads"] = self.tracker.get("loads", 0) + 1

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        self.tracker["active"] = self.tracker.get("active", 0) + 1
        self.tracker["max_active"] = max(
            self.tracker.get("max_active", 0), self.tracker["active"]
        )
        time.sleep(self.delay)
        self.tracker["active"] -= 1
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        return {
            "mask": mask,
            "overlay": image.copy(),
            "metadata": {"width": image.width, "height": image.height},
        }

    def unload(self) -> None:
        self.tracker["unloads"] = self.tracker.get("unloads", 0) + 1


class FailingAdapter(RecordingAdapter):
    def load(self, weight_path: Path | None) -> None:
        super().load(weight_path)
        raise RuntimeError("checkpoint rejected")


class ClassRecordingAdapter(RecordingAdapter):
    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        deterioration_class: str | None = None,
    ) -> dict:
        self.tracker.setdefault("classes", []).append(deterioration_class)
        return super().predict(image, threshold)


class RuntimeRecordingAdapter(RecordingAdapter):
    def prepare_runtime(self) -> None:
        self.tracker.setdefault("events", []).append("prepare")

    def load(self, weight_path: Path | None) -> None:
        self.tracker.setdefault("events", []).append("load")
        super().load(weight_path)


def _resolver(model_id: str, weight_name: str, model_root: Path) -> Path:
    return model_root / model_id / weight_name


def test_repeated_model_and_weight_reuses_cached_adapter(tmp_path: Path) -> None:
    tracker: dict[str, int] = {}
    manager = InferenceManager(
        model_root=tmp_path,
        adapter_factories={"alpha": lambda: RecordingAdapter(tracker)},
        weight_resolver=_resolver,
    )
    image = Image.new("RGB", (8, 6), "white")

    first = manager.predict("alpha", "one.pth", image, 0.5)
    second = manager.predict("alpha", "one.pth", image, 0.5)

    assert tracker["loads"] == 1
    assert tracker.get("unloads", 0) == 0
    assert first["mask"].mode == "L"
    assert second["model"] == "alpha"
    assert second["weight"] == "one.pth"


def test_switching_deterioration_class_reuses_same_loaded_model(tmp_path: Path) -> None:
    tracker: dict = {}
    manager = InferenceManager(
        model_root=tmp_path,
        adapter_factories={"da_sam3": lambda: ClassRecordingAdapter(tracker)},
        weight_resolver=_resolver,
    )
    image = Image.new("RGB", (8, 6), "white")

    crack = manager.predict(
        "da_sam3",
        "stage2_best.pt",
        image,
        deterioration_class="crack_craquelure",
    )
    loss = manager.predict(
        "da_sam3",
        "stage2_best.pt",
        image,
        deterioration_class="loss",
    )

    assert tracker["loads"] == 1
    assert tracker["classes"] == ["crack_craquelure", "loss"]
    assert crack["deterioration_class"] == "crack_craquelure"
    assert loss["deterioration_class"] == "loss"


def test_runtime_is_prepared_before_adapter_load(tmp_path: Path) -> None:
    tracker: dict = {}
    manager = InferenceManager(
        model_root=tmp_path,
        adapter_factories={"alpha": lambda: RuntimeRecordingAdapter(tracker)},
        weight_resolver=_resolver,
    )

    manager.predict("alpha", "one.pth", Image.new("RGB", (4, 4)))

    assert tracker["events"] == ["prepare", "load"]


def test_model_switch_unloads_previous_adapter(tmp_path: Path) -> None:
    alpha: dict[str, int] = {}
    beta: dict[str, int] = {}
    factories: dict[str, Callable[[], SegmentationAdapter]] = {
        "alpha": lambda: RecordingAdapter(alpha),
        "beta": lambda: RecordingAdapter(beta),
    }
    manager = InferenceManager(
        model_root=tmp_path,
        adapter_factories=factories,
        weight_resolver=_resolver,
    )
    image = Image.new("RGB", (4, 4), "black")

    manager.predict("alpha", "one.pth", image)
    manager.predict("beta", "two.pth", image)

    assert alpha["unloads"] == 1
    assert beta["loads"] == 1


def test_failed_load_is_cleaned_up_and_not_cached(tmp_path: Path) -> None:
    tracker: dict[str, int] = {}
    manager = InferenceManager(
        model_root=tmp_path,
        adapter_factories={"alpha": lambda: FailingAdapter(tracker)},
        weight_resolver=_resolver,
    )

    try:
        manager.predict("alpha", "broken.pt", Image.new("RGB", (4, 4)))
    except RuntimeError as exc:
        assert str(exc) == "checkpoint rejected"
    else:  # pragma: no cover - explicit diagnostic
        raise AssertionError("failing adapter should propagate its load error")

    assert tracker["loads"] == 1
    assert tracker["unloads"] == 1
    assert manager.cached_key is None


def test_inference_lock_serializes_concurrent_requests(tmp_path: Path) -> None:
    tracker: dict[str, int] = {}
    manager = InferenceManager(
        model_root=tmp_path,
        adapter_factories={"alpha": lambda: RecordingAdapter(tracker, delay=0.04)},
        weight_resolver=_resolver,
    )
    image = Image.new("RGB", (4, 4), "black")
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            start.wait()
            manager.predict("alpha", "one.pth", image)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert errors == []
    assert tracker["max_active"] == 1
