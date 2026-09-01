from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

import ui as ui_module
from config import Settings
from ui import (
    PRECISION_LAB_CSS,
    ApiClientError,
    InferenceApiClient,
    create_ui,
    format_service_status,
    private_demo_auth,
    precision_lab_theme,
)


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def _png_base64(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_api_client_loads_models_and_model_dependent_weights() -> None:
    session = FakeSession(
        [
            FakeResponse([{"id": "dummy", "label": "Dummy Segmentation"}]),
            FakeResponse(["built-in"]),
        ]
    )
    client = InferenceApiClient("http://127.0.0.1:5000", session=session)

    assert client.get_models() == [
        {"id": "dummy", "label": "Dummy Segmentation"}
    ]
    assert client.get_weights("dummy") == ["built-in"]
    assert session.calls[1][1].endswith("/api/models/dummy/weights")


def test_api_client_sends_internal_api_key_without_exposing_it_in_payload() -> None:
    session = FakeSession(
        [FakeResponse([{"id": "dummy", "label": "Dummy Segmentation"}])]
    )
    client = InferenceApiClient(
        "http://127.0.0.1:5000",
        api_key="b" * 32,
        session=session,
    )

    client.get_models()

    _, _, kwargs = session.calls[0]
    assert kwargs["headers"] == {"X-API-Key": "b" * 32}


def test_api_client_loads_health_status() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "status": "ok",
                    "cuda": True,
                    "device": "cuda",
                    "gpu": "Example GPU",
                }
            )
        ]
    )
    client = InferenceApiClient("http://127.0.0.1:5000", session=session)

    assert client.get_health() == {
        "status": "ok",
        "cuda": True,
        "device": "cuda",
        "gpu": "Example GPU",
    }
    assert session.calls[0][1].endswith("/api/health")


def test_api_client_posts_multipart_image_and_decodes_outputs() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "status": "success",
                    "model": "dummy",
                    "weight": "built-in",
                    "threshold": 0.5,
                    "device": "cpu",
                    "latency_ms": 3.2,
                    "mask_png_base64": _png_base64("black"),
                    "overlay_png_base64": _png_base64("red"),
                }
            )
        ]
    )
    client = InferenceApiClient("http://127.0.0.1:5000", session=session)

    result = client.infer(Image.new("RGB", (4, 3), "white"), "dummy", "built-in", 0.5)

    assert result["mask"].size == (4, 3)
    assert result["overlay"].size == (4, 3)
    assert result["metadata"].startswith("模型：dummy")
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url.endswith("/api/infer")
    assert kwargs["data"]["model"] == "dummy"
    assert "image" in kwargs["files"]


def test_gradio_blocks_build_without_contacting_flask() -> None:
    session = FakeSession([])
    client = InferenceApiClient("http://127.0.0.1:5000", session=session)

    demo = create_ui(api_client=client)

    assert type(demo).__name__ == "Blocks"
    assert session.calls == []
    assert "--lab-accent" in PRECISION_LAB_CSS
    config = str(demo.get_config_file())
    assert "古蹟劣化偵測" in config
    assert "精準分割實驗室" not in config
    assert "看清每一道" not in config
    assert "開始分割" in config
    assert "輸入影像" in config
    assert "step-number" not in config
    assert "privacy-note" not in config
    assert demo._queue.max_size == 2
    assert all(
        dependency["api_visibility"] == "private"
        for dependency in demo.get_config_file()["dependencies"]
    )


def test_private_demo_auth_returns_validated_gradio_credentials() -> None:
    settings = Settings(
        model_root=Path("/data/models"),
        flask_host="127.0.0.1",
        flask_port=5000,
        gradio_host="127.0.0.1",
        gradio_port=7860,
        flask_api_url="http://127.0.0.1:5000",
        max_upload_mb=10,
        request_timeout_seconds=120.0,
        private_demo_mode=True,
        gradio_auth_username="meeting",
        gradio_auth_password="strong-demo-password",
        internal_api_key="d" * 32,
    )

    assert private_demo_auth(settings) == ("meeting", "strong-demo-password")


def test_main_uses_no_custom_login_message(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        model_root=Path("/data/models"),
        flask_host="127.0.0.1",
        flask_port=5000,
        gradio_host="127.0.0.1",
        gradio_port=7860,
        flask_api_url="http://127.0.0.1:5000",
        max_upload_mb=10,
        request_timeout_seconds=120.0,
        private_demo_mode=True,
        gradio_auth_username="meeting",
        gradio_auth_password="1234",
        internal_api_key="e" * 32,
    )
    captured: dict[str, object] = {}

    class FakeDemo:
        def launch(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(ui_module, "default_settings", settings)
    monkeypatch.setattr(ui_module, "create_ui", FakeDemo)
    monkeypatch.setattr(ui_module, "precision_lab_theme", object)

    ui_module.main()

    assert captured["auth_message"] is None


def test_result_panel_has_no_decorative_dash_rail() -> None:
    assert ".result-panel::before" not in PRECISION_LAB_CSS
    assert "repeating-linear-gradient" not in PRECISION_LAB_CSS


def _contrast_ratio(foreground: str, background: str) -> float:
    def relative_luminance(color: str) -> float:
        channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted(
        [relative_luminance(foreground), relative_luminance(background)],
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_dropdown_theme_has_readable_light_menu_colors() -> None:
    theme = precision_lab_theme()

    assert theme.background_fill_primary == "#ffffff"
    assert theme.background_fill_secondary == "#f7f7f8"
    assert _contrast_ratio(
        theme.body_text_color,
        theme.background_fill_primary,
    ) >= 4.5


def test_ui_scopes_dropdown_and_slider_reset_styles() -> None:
    demo = create_ui(
        api_client=InferenceApiClient(
            "http://127.0.0.1:5000",
            session=FakeSession([]),
        )
    )
    config = str(demo.get_config_file())

    assert "model-dropdown" in config
    assert "weight-dropdown" in config
    assert "threshold-slider" in config
    assert "#precision-lab #threshold-slider .reset-button" in PRECISION_LAB_CSS


def test_service_status_is_accessible_and_escapes_server_text() -> None:
    online = format_service_status(
        {
            "status": "ok",
            "device": "cuda",
            "gpu": "Example <GPU>",
        }
    )
    offline = format_service_status(None)

    assert "API 已連線" in online
    assert "Example &lt;GPU&gt;" in online
    assert 'role="status"' in online
    assert "lab-status-light" not in online
    assert "API 未連線" in offline


def test_minimal_ui_css_is_white_scoped_and_undecorated() -> None:
    assert "--lab-bg: #ffffff" in PRECISION_LAB_CSS
    assert "html,\nbody" not in PRECISION_LAB_CSS
    assert ".gradio-container" not in PRECISION_LAB_CSS
    assert "box-shadow:" not in PRECISION_LAB_CSS
    assert "linear-gradient" not in PRECISION_LAB_CSS


def test_login_css_has_high_contrast_text_and_inputs() -> None:
    assert 'body:has(input[type="password"])' in PRECISION_LAB_CSS
    assert "color: #1c2430 !important" in PRECISION_LAB_CSS
    assert "background: #ffffff !important" in PRECISION_LAB_CSS


def test_api_client_localizes_known_server_error_without_changing_api() -> None:
    session = FakeSession([FakeResponse({"error": "Invalid image"}, 400)])
    client = InferenceApiClient("http://127.0.0.1:5000", session=session)

    with pytest.raises(ApiClientError, match="圖片格式無效"):
        client.infer(
            Image.new("RGB", (4, 3), "white"),
            "dummy",
            "built-in",
            0.5,
        )

    assert session.calls[0][1].endswith("/api/infer")
