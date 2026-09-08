import base64
import io
from pathlib import Path

from PIL import Image

from api import create_app
from config import Settings
from inference import InferenceManager


class _RecordingInferenceManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(
        self,
        model_id,
        weight_name,
        image,
        threshold,
        deterioration_class=None,
    ):
        self.calls.append(
            {
                "model": model_id,
                "weight": weight_name,
                "deterioration_class": deterioration_class,
            }
        )
        mask = Image.new("L", image.size, 0)
        return {
            "mask": mask,
            "overlay": image.copy(),
            "latency_ms": 1.0,
            "model": model_id,
            "weight": weight_name,
            "deterioration_class": deterioration_class,
            "device": "cpu",
        }


def _settings(
    model_root: Path,
    max_upload_mb: int = 10,
    *,
    private_demo_mode: bool = False,
    internal_api_key: str | None = None,
    inference_rate_limit_requests: int = 30,
    max_image_pixels: int = 2048 * 2048,
    max_image_side: int = 2048,
) -> Settings:
    return Settings(
        model_root=model_root,
        flask_host="127.0.0.1",
        flask_port=5000,
        gradio_host="127.0.0.1",
        gradio_port=7860,
        flask_api_url="http://127.0.0.1:5000",
        max_upload_mb=max_upload_mb,
        request_timeout_seconds=120.0,
        private_demo_mode=private_demo_mode,
        gradio_auth_username="meeting" if private_demo_mode else None,
        gradio_auth_password=(
            "strong-demo-password" if private_demo_mode else None
        ),
        internal_api_key=internal_api_key,
        inference_rate_limit_requests=inference_rate_limit_requests,
        max_image_pixels=max_image_pixels,
        max_image_side=max_image_side,
    )


def _png_file(size: tuple[int, int] = (12, 9)) -> io.BytesIO:
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format="PNG")
    stream.seek(0)
    return stream


def _client(tmp_path: Path, max_upload_mb: int = 10, **setting_overrides):
    settings = _settings(
        tmp_path,
        max_upload_mb=max_upload_mb,
        **setting_overrides,
    )
    manager = InferenceManager(model_root=tmp_path)
    app = create_app(settings=settings, inference_manager=manager)
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_and_models_endpoints(tmp_path: Path) -> None:
    client = _client(tmp_path)

    health = client.get("/api/health")
    models = client.get("/api/models")

    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"
    assert health.get_json()["device"] in {"cpu", "cuda"}
    assert models.status_code == 200
    assert models.get_json()[0]["id"] == "dummy"


def test_dummy_weights_endpoint(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/api/models/dummy/weights")

    assert response.status_code == 200
    assert response.get_json() == ["built-in"]


def test_dummy_inference_returns_pngs_and_metadata(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/infer",
        data={
            "image": (_png_file(), "input.png"),
            "model": "dummy",
            "weight": "built-in",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["model"] == "dummy"
    assert body["weight"] == "built-in"
    assert body["threshold"] == 0.5
    assert body["latency_ms"] >= 0
    assert base64.b64decode(body["mask_png_base64"]).startswith(b"\x89PNG")
    assert base64.b64decode(body["overlay_png_base64"]).startswith(b"\x89PNG")


def test_da_sam3_inference_requires_and_forwards_selected_class(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    manager = _RecordingInferenceManager()
    app = create_app(settings=settings, inference_manager=manager)
    app.config.update(TESTING=True)
    client = app.test_client()

    missing = client.post(
        "/api/infer",
        data={
            "image": (_png_file(), "input.png"),
            "model": "da_sam3",
            "weight": "stage2_best.pt",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
    )
    invalid = client.post(
        "/api/infer",
        data={
            "image": (_png_file(), "input.png"),
            "model": "da_sam3",
            "weight": "stage2_best.pt",
            "threshold": "0.5",
            "deterioration_class": "other",
        },
        content_type="multipart/form-data",
    )
    accepted = client.post(
        "/api/infer",
        data={
            "image": (_png_file(), "input.png"),
            "model": "da_sam3",
            "weight": "stage2_best.pt",
            "threshold": "0.5",
            "deterioration_class": "loss",
        },
        content_type="multipart/form-data",
    )

    assert missing.status_code == 400
    assert missing.get_json()["error"] == (
        "Deterioration class is required for DA-SAM3"
    )
    assert invalid.status_code == 400
    assert invalid.get_json()["error"] == "Invalid deterioration class"
    assert accepted.status_code == 200
    assert accepted.get_json()["deterioration_class"] == "loss"
    assert manager.calls == [
        {
            "model": "da_sam3",
            "weight": "stage2_best.pt",
            "deterioration_class": "loss",
        }
    ]


def test_inference_rejects_invalid_inputs(tmp_path: Path) -> None:
    client = _client(tmp_path)

    invalid_threshold = client.post(
        "/api/infer",
        data={
            "image": (_png_file(), "input.png"),
            "model": "dummy",
            "weight": "built-in",
            "threshold": "1.5",
        },
        content_type="multipart/form-data",
    )
    malformed_image = client.post(
        "/api/infer",
        data={
            "image": (io.BytesIO(b"broken"), "input.png"),
            "model": "dummy",
            "weight": "built-in",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
    )
    unknown_model = client.get("/api/models/not-real/weights")
    unsafe_weight = client.post(
        "/api/infer",
        data={
            "image": (_png_file(), "input.png"),
            "model": "dummy",
            "weight": "../../secret.pth",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
    )

    assert invalid_threshold.status_code == 400
    assert invalid_threshold.get_json()["error"] == "Invalid threshold"
    assert malformed_image.status_code == 400
    assert malformed_image.get_json()["error"] == "Invalid image"
    assert unknown_model.status_code == 400
    assert unknown_model.get_json()["error"] == "Unknown model"
    assert unsafe_weight.status_code == 400
    assert "Invalid weight" in unsafe_weight.get_json()["error"]


def test_oversized_upload_returns_json_error(tmp_path: Path) -> None:
    client = _client(tmp_path, max_upload_mb=1)

    response = client.post(
        "/api/infer",
        data={
            "image": (io.BytesIO(b"x" * (1024 * 1024 + 1)), "large.png"),
            "model": "dummy",
            "weight": "built-in",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert response.get_json()["error"] == "Upload too large"


def test_private_demo_api_requires_matching_internal_key(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        private_demo_mode=True,
        internal_api_key="a" * 32,
    )

    missing = client.get("/api/models")
    wrong = client.get("/api/models", headers={"X-API-Key": "wrong"})
    accepted = client.get("/api/models", headers={"X-API-Key": "a" * 32})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.get_json()["error"] == "Unauthorized"
    assert accepted.status_code == 200


def test_inference_rate_limit_rejects_excess_requests(tmp_path: Path) -> None:
    client = _client(tmp_path, inference_rate_limit_requests=1)
    request_data = {
        "image": (_png_file(), "input.png"),
        "model": "dummy",
        "weight": "built-in",
        "threshold": "0.5",
    }

    accepted = client.post(
        "/api/infer",
        data=request_data,
        content_type="multipart/form-data",
    )
    rejected = client.post(
        "/api/infer",
        data={
            **request_data,
            "image": (_png_file(), "input.png"),
        },
        content_type="multipart/form-data",
    )

    assert accepted.status_code == 200
    assert rejected.status_code == 429
    assert rejected.get_json()["error"] == "Rate limit exceeded"
    assert int(rejected.headers["Retry-After"]) >= 1


def test_decoded_image_dimensions_are_limited(tmp_path: Path) -> None:
    client = _client(tmp_path, max_image_pixels=64, max_image_side=8)

    response = client.post(
        "/api/infer",
        data={
            "image": (_png_file((9, 8)), "input.png"),
            "model": "dummy",
            "weight": "built-in",
            "threshold": "0.5",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert response.get_json()["error"] == "Image dimensions too large"
