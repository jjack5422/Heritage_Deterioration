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


RUN_REAL_MODELS = os.getenv("RUN_REAL_MODEL_TESTS") == "1"
MODEL_IDS = (
    "sam2_adapter",
    "sam3_adapter",
    "resunet50",
    "convnext_unet",
)
REPRESENTATIVE_IMAGE = WORKSPACE_ROOT / (
    "datasets/dataset_clean_v2_merged_craquelure/images/"
    "MGLST-DT-1R-A2-1_R1_C04__y00512_x00512.png"
)


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

    for model_id in MODEL_IDS:
        print(f"real_model_start model={model_id}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        result = manager.predict(model_id, "best.pt", image, threshold=0.5)
        mask = np.asarray(result["mask"])
        metadata = result["metadata"]
        assert result["mask"].size == image.size, model_id
        assert result["overlay"].size == image.size, model_id
        assert set(np.unique(mask)).issubset({0, 255}), model_id
        assert int((mask > 0).sum()) > 0, model_id
        assert math.isfinite(float(metadata["probability_min"])), model_id
        assert math.isfinite(float(metadata["probability_max"])), model_id
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
    )
    assert response.status_code == 200, response.get_json()
    assert base64.b64decode(response.get_json()["mask_png_base64"]).startswith(
        b"\x89PNG"
    )
    assert set(observations) == set(MODEL_IDS)
    print(f"real_model_summary={json.dumps(observations, sort_keys=True)}", flush=True)
    manager.unload()
    torch.cuda.empty_cache()
