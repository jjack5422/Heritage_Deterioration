"""Opt-in RTX integration test for all real web model checkpoints."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from api import create_app
from config import WORKSPACE_ROOT, load_settings
from inference import InferenceManager
from registry import HYBRID_COMPONENTS, HYBRID_MODEL_ID, HYBRID_WEIGHT, resolve_deterioration_class


RUN_REAL_MODELS = os.getenv("RUN_REAL_MODEL_TESTS") == "1"
MODEL_IDS = (
    "sam3_adapter_craquelure",
    "sam3_adapter_loss",
    "sam3_adapter_crack",
    "sam2_adapter_loss",
    "sam2_adapter_crack",
    "sam2_adapter_craquelure",
    "resunet50",
    "convnext_unet",
)
REPRESENTATIVE_IMAGE = WORKSPACE_ROOT / (
    "datasets/dataset_clean_v2_merged_craquelure/images/"
    "MGLST-DT-1R-A2-1_R1_C04__y00512_x00512.png"
)


@pytest.mark.skipif(not RUN_REAL_MODELS, reason="set RUN_REAL_MODEL_TESTS=1 for GPU integration")
def test_real_hybrid_api_preserves_all_three_expert_predictions() -> None:
    assert torch.cuda.is_available(), "RUN_REAL_MODEL_TESTS=1 requires CUDA"
    settings = load_settings()
    manager = InferenceManager(runtime_settings=settings)
    image = Image.open(REPRESENTATIVE_IMAGE).convert("RGB")
    try:
        expected = {
            item["class"]: np.asarray(manager.predict(
                item["model"], item["weight"], image, .5
            )["mask"]).copy()
            for item in HYBRID_COMPONENTS
        }
        app = create_app(settings=settings, inference_manager=manager)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        stream.seek(0)
        response = app.test_client().post("/api/infer", data={
            "image": (stream, "wall.png"), "model": HYBRID_MODEL_ID,
            "weight": HYBRID_WEIGHT,
        }, headers={"X-API-Key": settings.internal_api_key} if settings.private_demo_mode else {})
        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert body["thresholds"] == {item["class"]: .5 for item in HYBRID_COMPONENTS}
        with Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"]))) as colored:
            assert colored.mode == "RGB"
            assert colored.size == image.size
        for item in body["components"]:
            with Image.open(io.BytesIO(base64.b64decode(item["mask_png_base64"]))) as mask:
                np.testing.assert_array_equal(mask, expected[item["class"]])
        overlap = np.stack([value > 0 for value in expected.values()]).sum(axis=0) > 1
        assert body["overlap_pixels"] == int(overlap.sum())
        assert manager.cached_key == ("sam3_adapter_loss", "best.pt")
        print(f"real_hybrid_success latency_ms={body['latency_ms']} overlap_pixels={body['overlap_pixels']}", flush=True)
    finally:
        manager.unload()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="set RUN_REAL_MODEL_TESTS=1 to load multi-gigabyte GPU models",
)
def test_real_models_load_switch_and_infer_through_web_contract() -> None:
    if not torch.cuda.is_available():
        pytest.fail("RUN_REAL_MODEL_TESTS=1 requires CUDA")
    settings = load_settings()
    manager = InferenceManager(runtime_settings=settings)
    image = Image.open(REPRESENTATIVE_IMAGE).convert("RGB")
    observations: dict[str, dict[str, object]] = {}
    app = create_app(settings=settings, inference_manager=manager)
    app.config.update(TESTING=True)
    client = app.test_client()

    for model_id in MODEL_IDS:
        print(f"real_model_start model={model_id}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        result = manager.predict(model_id, "best.pt", image, threshold=0.5)
        mask = np.asarray(result["mask"])
        metadata = result["metadata"]
        assert result["mask"].size == image.size, model_id
        assert result["overlay"].size == image.size, model_id
        assert set(np.unique(mask)).issubset({0, 255}), model_id
        assert math.isfinite(float(metadata["probability_min"])), model_id
        assert math.isfinite(float(metadata["probability_max"])), model_id
        if model_id.startswith(("sam2_adapter_", "sam3_adapter_")):
            assert metadata["model_input_size"] == 1008
            assert metadata["expert"] == {
                "craquelure": "shrinkage_craquelure", "loss": "loss", "crack": "scratch_crack"
            }[result["deterioration_class"]]
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        stream.seek(0)
        response = client.post(
            "/api/infer",
            headers={"X-API-Key": settings.internal_api_key} if settings.private_demo_mode else {},
            data={
                "image": (stream, "representative.png"), "model": model_id,
                "weight": "best.pt", "threshold": "0.5",
                "deterioration_class": resolve_deterioration_class(model_id, None),
            }, content_type="multipart/form-data",
        )
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["deterioration_class"] == result["deterioration_class"]
        assert base64.b64decode(response.get_json()["mask_png_base64"]).startswith(b"\x89PNG")
        observations[model_id] = {
            "latency_ms": result["latency_ms"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "foreground_pixels": int((mask > 0).sum()),
        }
        print(
            f"real_model_success model={model_id} "
            f"observation={json.dumps(observations[model_id], sort_keys=True)}",
            flush=True,
        )

    app = create_app(settings=settings, inference_manager=manager)
    app.config.update(TESTING=True)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    response = app.test_client().post(
        "/api/infer",
        data={
            "image": (stream, "representative.png"),
            "model": "convnext_unet",
            "weight": "best.pt",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
        headers={"X-API-Key": settings.internal_api_key} if settings.private_demo_mode else {},
    )
    assert response.status_code == 200, response.get_json()
    assert base64.b64decode(response.get_json()["mask_png_base64"]).startswith(
        b"\x89PNG"
    )
    assert set(observations) == set(MODEL_IDS)
    print(f"real_model_summary={json.dumps(observations, sort_keys=True)}", flush=True)
    manager.unload()
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="set RUN_REAL_MODEL_TESTS=1 to load multi-gigabyte GPU models",
)
def test_real_da_sam3_infers_both_classes_with_one_cached_model() -> None:
    if not torch.cuda.is_available():
        pytest.fail("RUN_REAL_MODEL_TESTS=1 requires CUDA")
    settings = load_settings()
    manager = InferenceManager(runtime_settings=settings)
    image = Image.open(REPRESENTATIVE_IMAGE).convert("RGB")
    observations: dict[str, dict[str, object]] = {}

    try:
        torch.cuda.reset_peak_memory_stats()
        for deterioration_class in ("crack_craquelure", "loss"):
            result = manager.predict(
                "da_sam3",
                "stage2_best.pt",
                image,
                threshold=0.5,
                deterioration_class=deterioration_class,
            )
            mask = np.asarray(result["mask"])
            metadata = result["metadata"]
            assert result["mask"].size == image.size
            assert result["overlay"].size == image.size
            assert set(np.unique(mask)).issubset({0, 255})
            assert math.isfinite(float(metadata["probability_min"]))
            assert math.isfinite(float(metadata["probability_max"]))
            assert metadata["model_variant"] == "visual_da_sam3"
            assert metadata["full_pixel_decoder"] is True
            assert metadata["deterioration_class"] == deterioration_class
            assert manager.cached_key == ("da_sam3", "stage2_best.pt")
            observations[deterioration_class] = {
                "latency_ms": result["latency_ms"],
                "foreground_pixels": int((mask > 0).sum()),
                "probability_min": float(metadata["probability_min"]),
                "probability_max": float(metadata["probability_max"]),
            }
        observations["peak_allocated_mib"] = {
            "value": torch.cuda.max_memory_allocated() / 1024**2
        }
        print(
            f"real_da_sam3_summary={json.dumps(observations, sort_keys=True)}",
            flush=True,
        )
    finally:
        manager.unload()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="set RUN_REAL_MODEL_TESTS=1 to load multi-gigabyte GPU models",
)
def test_real_sam3_runtimes_switch_in_both_directions() -> None:
    if not torch.cuda.is_available():
        pytest.fail("RUN_REAL_MODEL_TESTS=1 requires CUDA")
    settings = load_settings()
    manager = InferenceManager(runtime_settings=settings)
    image = Image.open(REPRESENTATIVE_IMAGE).convert("RGB")
    cases = (
        ("da_sam3", "stage2_best.pt", "crack_craquelure"),
        ("sam3_adapter_crack", "best.pt", "crack"),
        ("da_sam3", "stage2_best.pt", "loss"),
    )

    try:
        for model_id, weight, deterioration_class in cases:
            result = manager.predict(
                model_id,
                weight,
                image,
                threshold=0.5,
                deterioration_class=deterioration_class,
            )
            mask = np.asarray(result["mask"])
            assert result["mask"].size == image.size
            assert result["overlay"].size == image.size
            assert set(np.unique(mask)).issubset({0, 255})
            print(
                f"runtime_switch_success model={model_id} "
                f"class={deterioration_class} latency_ms={result['latency_ms']:.1f}",
                flush=True,
            )
    finally:
        manager.unload()
        torch.cuda.empty_cache()
