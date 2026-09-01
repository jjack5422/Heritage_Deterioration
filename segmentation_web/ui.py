"""Gradio UI that delegates every inference operation to the Flask API."""

from __future__ import annotations

import html
import io
import logging
from typing import Any
from urllib.parse import quote

import gradio as gr
import requests
from PIL import Image

from config import (
    Settings,
    settings as default_settings,
    validate_private_demo_settings,
)
from imaging.image_processing import (
    base64_png_to_pil,
    configure_image_decompression_limit,
)


LOGGER = logging.getLogger(__name__)


MODEL_LABELS_ZH = {
    "dummy": "測試分割模型（Dummy）",
    "sam2_adapter": "SAM2 Adapter 分割模型",
    "sam3_adapter": "SAM3 Adapter 分割模型",
    "resunet50": "ResUNet50 分割模型",
    "convnext_unet": "ConvNeXt-Large U-Net 分割模型",
}

API_ERROR_MESSAGES_ZH = {
    "Upload too large": "上傳圖片超過大小限制",
    "Unknown model": "找不到指定的模型",
    "Image is required": "請先上傳圖片",
    "Model is required": "請選擇模型",
    "Weight is required": "找不到可用的模型權重",
    "Invalid image": "圖片格式無效，請上傳 JPEG、PNG 或 WEBP 圖片",
    "Invalid threshold": "遮罩閾值必須介於 0.00 到 1.00 之間",
    "Inference failed": "推論失敗，請稍後再試",
    "Unauthorized": "推論服務驗證失敗，請重新啟動私人展示服務",
    "Rate limit exceeded": "推論次數已達上限，請稍後再試",
    "Image dimensions too large": "圖片尺寸過大，最大為 2048×2048",
}


PRECISION_LAB_CSS = r"""
#precision-lab {
    --lab-bg: #ffffff;
    --lab-surface: #ffffff;
    --lab-subtle: #f7f7f8;
    --lab-line: #d8dde3;
    --lab-line-strong: #b8c0ca;
    --lab-text: #1c2430;
    --lab-muted: #5f6b7a;
    --lab-accent: #2563eb;
    --lab-accent-strong: #1d4ed8;
    --lab-success: #067647;
    --lab-danger: #b42318;
    width: 100%;
    min-height: 100vh;
    min-height: 100dvh;
    margin: 0;
    padding: 24px;
    color: var(--lab-text);
    background: var(--lab-bg);
    font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans TC", sans-serif;
}

#precision-lab main,
#precision-lab .main,
#precision-lab > .contain {
    width: min(100%, 1280px);
    margin: 0 auto;
    padding: 0 !important;
}

#precision-lab footer {
    display: none !important;
}

#precision-lab * {
    box-sizing: border-box;
}

#precision-lab button,
#precision-lab [role="button"] {
    min-height: 44px;
    cursor: pointer;
}

#precision-lab :focus-visible {
    outline: 3px solid var(--lab-accent) !important;
    outline-offset: 2px !important;
}

#precision-lab .lab-header {
    align-items: center;
    gap: 20px;
    margin: 0 0 20px;
    padding: 0 0 16px;
    border-bottom: 1px solid var(--lab-line);
}

#precision-lab .brand-block h1 {
    margin: 0;
    color: var(--lab-text);
    font-size: 1.25rem;
    font-weight: 650;
    letter-spacing: -0.015em;
    line-height: 1.4;
}

#precision-lab #service-status {
    align-self: center;
    min-width: 240px;
}

#precision-lab .lab-service-status {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 8px;
    align-items: baseline;
    justify-content: flex-end;
    color: var(--lab-muted);
    font-size: 0.82rem;
    line-height: 1.5;
}

#precision-lab .lab-status-label {
    color: var(--lab-danger);
    font-weight: 650;
}

#precision-lab .lab-service-status.is-online .lab-status-label {
    color: var(--lab-success);
}

#precision-lab .lab-status-detail {
    overflow: hidden;
    max-width: 230px;
    color: var(--lab-muted);
    text-overflow: ellipsis;
    white-space: nowrap;
}

#precision-lab .lab-workbench {
    gap: 16px;
    align-items: stretch;
}

#precision-lab .control-panel,
#precision-lab .result-panel {
    min-width: 0 !important;
    padding: 20px;
    border: 1px solid var(--lab-line);
    border-radius: 4px;
    background: var(--lab-surface);
}

#precision-lab .section-heading {
    margin-bottom: 16px;
}

#precision-lab .section-heading h2 {
    margin: 0;
    color: var(--lab-text);
    font-size: 1rem;
    font-weight: 650;
    line-height: 1.5;
}

#precision-lab .block,
#precision-lab .form {
    border-color: var(--lab-line);
}

#precision-lab .source-stage,
#precision-lab .output-stage {
    overflow: hidden;
    border: 1px solid var(--lab-line) !important;
    border-radius: 3px !important;
    background: var(--lab-subtle) !important;
}

#precision-lab .field-pair {
    gap: 12px;
}

#precision-lab .threshold-note {
    margin: 4px 0 12px;
    color: var(--lab-muted);
    font-size: 0.78rem;
    line-height: 1.5;
}

#precision-lab #run-button {
    min-height: 48px;
    margin-top: 4px;
    border: 1px solid var(--lab-accent) !important;
    border-radius: 3px !important;
    color: #ffffff !important;
    background: var(--lab-accent) !important;
    font-weight: 650;
    transition: background-color 140ms ease, border-color 140ms ease;
}

#precision-lab #run-button:hover {
    border-color: var(--lab-accent-strong) !important;
    background: var(--lab-accent-strong) !important;
}

#precision-lab #run-button:active {
    background: #1e40af !important;
}

#precision-lab .secondary-results {
    gap: 12px;
    margin-top: 12px;
}

#precision-lab .telemetry-panel {
    margin-top: 12px;
}

#precision-lab .telemetry-panel textarea {
    min-height: 128px;
    color: var(--lab-text);
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.78rem;
    line-height: 1.65;
}

#precision-lab label,
#precision-lab .label-wrap {
    color: var(--lab-text) !important;
    font-weight: 600;
}

#precision-lab .lab-dropdown .wrap,
#precision-lab .lab-dropdown input {
    color: var(--lab-text) !important;
    background: var(--lab-surface) !important;
}

#precision-lab .options {
    border: 1px solid var(--lab-line-strong) !important;
    color: var(--lab-text) !important;
    background: var(--lab-surface) !important;
}

#precision-lab .options .item {
    color: var(--lab-text) !important;
}

#precision-lab .options .item:hover,
#precision-lab .options .active {
    color: var(--lab-text) !important;
    background: var(--lab-subtle) !important;
}

#precision-lab #threshold-slider .reset-button {
    display: inline-flex !important;
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    align-items: center;
    justify-content: center;
}

#precision-lab #threshold-slider .reset-button svg {
    width: 14px;
    height: 14px;
}

#precision-lab input,
#precision-lab textarea,
#precision-lab .wrap {
    transition: border-color 140ms ease, background-color 140ms ease;
}

@media (max-width: 900px) {
    #precision-lab {
        padding: 18px 16px 24px;
    }

    #precision-lab .lab-header {
        align-items: flex-start;
    }

    #precision-lab .lab-header > * {
        min-width: 0 !important;
    }

    #precision-lab #service-status {
        min-width: 0;
    }

    #precision-lab .lab-workbench {
        flex-direction: column;
    }
}

@media (max-width: 560px) {
    #precision-lab {
        padding: 14px 12px 20px;
    }

    #precision-lab .lab-header {
        flex-direction: column !important;
        gap: 8px;
        align-items: stretch;
    }

    #precision-lab .lab-header > * {
        width: 100% !important;
        flex: 1 1 auto !important;
    }

    #precision-lab .lab-service-status {
        justify-content: flex-start;
    }

    #precision-lab .control-panel,
    #precision-lab .result-panel {
        padding: 14px;
    }

    #precision-lab .field-pair,
    #precision-lab .secondary-results {
        flex-direction: column;
    }
}

@media (prefers-reduced-motion: reduce) {
    #precision-lab *,
    #precision-lab *::before,
    #precision-lab *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
    }
}

/* Gradio renders authentication before the Blocks root, so target that page
   through its password field without changing the inference workspace. */
body:has(input[type="password"]) {
    color: #1c2430 !important;
    background: #ffffff !important;
}

body:has(input[type="password"]) .wrap {
    color: #1c2430 !important;
    background: #ffffff !important;
}

body:has(input[type="password"]) label,
body:has(input[type="password"]) .label-wrap,
body:has(input[type="password"]) h2 {
    color: #1c2430 !important;
}

body:has(input[type="password"]) input[type="text"],
body:has(input[type="password"]) input[type="password"] {
    color: #1c2430 !important;
    caret-color: #1c2430 !important;
    background: #ffffff !important;
    border-color: #b8c0ca !important;
}

body:has(input[type="password"]) input::placeholder {
    color: #5f6b7a !important;
}

body:has(input[type="password"]) input:focus {
    border-color: #2563eb !important;
    outline: 3px solid rgb(37 99 235 / 20%) !important;
    outline-offset: 1px;
}

body:has(input[type="password"]) button {
    color: #ffffff !important;
    background: #2563eb !important;
    border-color: #2563eb !important;
}

body:has(input[type="password"]) button:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
"""


def format_service_status(health: dict[str, Any] | None) -> str:
    """Render an accessible API/GPU status badge for the Gradio header."""

    if health and health.get("status") == "ok":
        gpu = health.get("gpu") or str(health.get("device", "cpu")).upper()
        detail = html.escape(str(gpu))
        return (
            '<div class="lab-service-status is-online" role="status" '
            'aria-live="polite">'
            '<span class="lab-status-label">API 已連線</span>'
            f'<span class="lab-status-detail" title="{detail}">｜{detail}</span>'
            "</div>"
        )
    return (
        '<div class="lab-service-status is-offline" role="status" '
        'aria-live="polite">'
        '<span class="lab-status-label">API 未連線</span>'
        '<span class="lab-status-detail">｜請先啟動 api.py，再重新整理頁面</span>'
        "</div>"
    )


def _localized_api_error(message: str, status_code: int) -> str:
    if message in API_ERROR_MESSAGES_ZH:
        return API_ERROR_MESSAGES_ZH[message]
    if message.startswith("Invalid weight"):
        return "模型權重無效，請重新選擇"
    if status_code >= 500:
        return "推論服務目前無法完成請求，請稍後再試"
    return "送出的資料無效，請檢查設定後再試"


def precision_lab_theme():
    """Return the offline-capable theme used when launching Gradio."""

    return gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="sm",
        font=("system-ui", "Noto Sans TC", "Segoe UI", "sans-serif"),
        font_mono=("ui-monospace", "SFMono-Regular", "Consolas", "monospace"),
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f7f7f8",
        background_fill_secondary_dark="#f7f7f8",
        body_text_color="#1c2430",
        body_text_color_dark="#1c2430",
        body_text_color_subdued="#5f6b7a",
        body_text_color_subdued_dark="#5f6b7a",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#d8dde3",
        block_border_color_dark="#d8dde3",
        block_label_background_fill="#ffffff",
        block_label_background_fill_dark="#ffffff",
        block_label_text_color="#1c2430",
        block_label_text_color_dark="#1c2430",
        border_color_primary="#d8dde3",
        border_color_primary_dark="#d8dde3",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_background_fill_focus="#ffffff",
        input_border_color="#d8dde3",
        input_border_color_dark="#d8dde3",
        input_border_color_focus="#2563eb",
        input_border_color_focus_dark="#2563eb",
        slider_color="#2563eb",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_dark="#2563eb",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
    )


class ApiClientError(RuntimeError):
    """Safe, user-facing error raised for Flask communication failures."""


class InferenceApiClient:
    """Small HTTP client used by Gradio instead of importing inference code."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _payload(self, response: requests.Response) -> Any:
        try:
            payload = response.json()
        except (requests.JSONDecodeError, ValueError) as exc:
            raise ApiClientError("Flask API 回傳的資料格式無效") from exc
        if response.status_code >= 400:
            if isinstance(payload, dict):
                message = payload.get("error", "Flask API request failed")
            else:
                message = "Flask API request failed"
            raise ApiClientError(
                _localized_api_error(str(message), response.status_code)
            )
        return payload

    def get_health(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout_seconds,
                headers=self._headers(),
            )
            payload = self._payload(response)
        except requests.RequestException as exc:
            raise ApiClientError("無法連線至 Flask 推論 API") from exc
        if not isinstance(payload, dict):
            raise ApiClientError("Flask API 回傳的服務狀態格式無效")
        return payload

    def get_models(self) -> list[dict[str, str]]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/models",
                timeout=self.timeout_seconds,
                headers=self._headers(),
            )
            payload = self._payload(response)
        except requests.RequestException as exc:
            raise ApiClientError("無法連線至 Flask 推論 API") from exc
        if not isinstance(payload, list):
            raise ApiClientError("Flask API 回傳的模型清單格式無效")
        return payload

    def get_weights(self, model_id: str) -> list[str]:
        if not model_id:
            return []
        safe_model_id = quote(model_id, safe="")
        try:
            response = self.session.get(
                f"{self.base_url}/api/models/{safe_model_id}/weights",
                timeout=self.timeout_seconds,
                headers=self._headers(),
            )
            payload = self._payload(response)
        except requests.RequestException as exc:
            raise ApiClientError("無法連線至 Flask 推論 API") from exc
        if not isinstance(payload, list):
            raise ApiClientError("Flask API 回傳的權重清單格式無效")
        return [str(weight) for weight in payload]

    def infer(
        self,
        image: Image.Image,
        model_id: str,
        weight_name: str,
        threshold: float,
    ) -> dict[str, Any]:
        if image is None:
            raise ApiClientError("請先上傳圖片")
        if not model_id:
            raise ApiClientError("請選擇模型")
        if not weight_name:
            raise ApiClientError("找不到相容的模型權重")

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        try:
            response = self.session.post(
                f"{self.base_url}/api/infer",
                files={"image": ("upload.png", buffer.getvalue(), "image/png")},
                data={
                    "model": model_id,
                    "weight": weight_name,
                    "threshold": str(float(threshold)),
                },
                timeout=self.timeout_seconds,
                headers=self._headers(),
            )
            payload = self._payload(response)
        except requests.RequestException as exc:
            raise ApiClientError("無法連線至 Flask 推論 API") from exc
        if not isinstance(payload, dict):
            raise ApiClientError("Flask API 回傳的推論結果格式無效")

        try:
            mask = base64_png_to_pil(str(payload["mask_png_base64"])).convert("L")
            overlay = base64_png_to_pil(str(payload["overlay_png_base64"]))
            metadata = (
                f"模型：{payload['model']}\n"
                f"模型權重：{payload['weight']}\n"
                f"遮罩閾值：{float(payload['threshold']):.2f}\n"
                f"執行裝置：{payload['device']}\n"
                f"推論時間：{float(payload['latency_ms']):.1f} 毫秒"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiClientError("Flask API 回傳的推論資料不完整") from exc
        return {"mask": mask, "overlay": overlay, "metadata": metadata}


def create_ui(
    *,
    settings: Settings | None = None,
    api_client: InferenceApiClient | None = None,
) -> gr.Blocks:
    """Build the Gradio interface without loading any segmentation model."""

    app_settings = settings or default_settings
    configure_image_decompression_limit(app_settings.max_image_pixels)
    client = api_client or InferenceApiClient(
        app_settings.flask_api_url,
        timeout_seconds=app_settings.request_timeout_seconds,
        api_key=app_settings.internal_api_key,
    )

    def load_models():
        try:
            models = client.get_models()
            choices = [
                (MODEL_LABELS_ZH.get(model["id"], model["label"]), model["id"])
                for model in models
            ]
            selected = choices[0][1] if choices else None
            weights = client.get_weights(selected) if selected else []
            selected_weight = weights[0] if weights else None
            try:
                health = client.get_health()
            except ApiClientError:
                health = None
            return (
                gr.Dropdown(choices=choices, value=selected),
                gr.Dropdown(choices=weights, value=selected_weight),
                format_service_status(health),
            )
        except (ApiClientError, KeyError) as exc:
            raise gr.Error(str(exc)) from exc

    def load_weights(model_id: str):
        try:
            weights = client.get_weights(model_id)
            if not weights:
                gr.Warning("找不到相容的模型權重")
                return gr.Dropdown(choices=[], value=None)
            return gr.Dropdown(choices=weights, value=weights[0])
        except ApiClientError as exc:
            raise gr.Error(str(exc)) from exc

    def run_inference(
        image: Image.Image | None,
        model_id: str | None,
        weight_name: str | None,
        threshold: float,
    ):
        try:
            if image is None:
                raise ApiClientError("請先上傳圖片")
            result = client.infer(
                image,
                model_id or "",
                weight_name or "",
                threshold,
            )
            return image.convert("RGB"), result["mask"], result["overlay"], result[
                "metadata"
            ]
        except ApiClientError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(
        title="古蹟劣化偵測",
        analytics_enabled=False,
        elem_id="precision-lab",
    ) as demo:
        with gr.Row(elem_classes=["lab-header"], equal_height=False):
            with gr.Column(
                scale=4,
                min_width=520,
                elem_classes=["brand-column"],
            ):
                gr.HTML(
                    """
                    <header class="brand-block">
                        <h1>古蹟劣化偵測</h1>
                    </header>
                    """
                )
            with gr.Column(
                scale=1,
                min_width=300,
                elem_classes=["status-column"],
            ):
                service_status = gr.HTML(
                    value=format_service_status(None),
                    elem_id="service-status",
                )

        with gr.Row(elem_classes=["lab-workbench"], equal_height=False):
            with gr.Column(
                scale=4,
                min_width=320,
                elem_classes=["control-panel"],
            ):
                gr.HTML(
                    """
                    <div class="section-heading">
                        <h2>推論設定</h2>
                    </div>
                    """
                )
                input_image = gr.Image(
                    type="pil",
                    sources=["upload"],
                    label="輸入影像",
                    height=290,
                    placeholder="拖曳或點選上傳 PNG、JPEG 圖片",
                    elem_classes=["source-stage"],
                )
                with gr.Row(elem_classes=["field-pair"]):
                    model_dropdown = gr.Dropdown(
                        choices=[],
                        label="模型",
                        allow_custom_value=True,
                        elem_id="model-dropdown",
                        elem_classes=["lab-dropdown"],
                    )
                    weight_dropdown = gr.Dropdown(
                        choices=[],
                        label="模型權重",
                        allow_custom_value=True,
                        elem_id="weight-dropdown",
                        elem_classes=["lab-dropdown"],
                    )
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="遮罩閾值",
                    elem_id="threshold-slider",
                )
                gr.HTML(
                    '<p class="threshold-note">0.00 較敏感 · 1.00 較嚴格</p>'
                )
                run_button = gr.Button(
                    "開始分割",
                    variant="primary",
                    elem_id="run-button",
                )

            with gr.Column(
                scale=7,
                min_width=480,
                elem_classes=["result-panel"],
            ):
                gr.HTML(
                    """
                    <div class="section-heading">
                        <h2>分割結果</h2>
                    </div>
                    """
                )
                overlay_output = gr.Image(
                    label="分割疊合圖",
                    height=470,
                    interactive=False,
                    elem_id="overlay-output",
                    elem_classes=["output-stage", "primary-stage"],
                )
                with gr.Row(elem_classes=["secondary-results"]):
                    original_output = gr.Image(
                        label="原始影像",
                        height=230,
                        interactive=False,
                        elem_classes=["output-stage"],
                    )
                    mask_output = gr.Image(
                        label="二值遮罩",
                        image_mode="L",
                        height=230,
                        interactive=False,
                        elem_classes=["output-stage"],
                    )
                information_output = gr.Textbox(
                    value=(
                        "等待推論\n"
                        "完成分割後，這裡會顯示本次推論資訊。"
                    ),
                    label="推論資訊",
                    lines=5,
                    interactive=False,
                    elem_classes=["telemetry-panel"],
                )

        demo.load(
            fn=load_models,
            outputs=[model_dropdown, weight_dropdown, service_status],
            queue=False,
            api_visibility="private",
        )
        model_dropdown.change(
            fn=load_weights,
            inputs=model_dropdown,
            outputs=weight_dropdown,
            queue=False,
            show_progress="minimal",
            api_visibility="private",
        )
        run_button.click(
            fn=run_inference,
            inputs=[
                input_image,
                model_dropdown,
                weight_dropdown,
                threshold_slider,
            ],
            outputs=[
                original_output,
                mask_output,
                overlay_output,
                information_output,
            ],
            concurrency_limit=1,
            concurrency_id="segmentation_inference",
            show_progress="minimal",
            scroll_to_output=True,
            api_visibility="private",
        )

    return demo.queue(
        default_concurrency_limit=1,
        max_size=app_settings.gradio_queue_max_size,
        api_open=False,
    )


def private_demo_auth(app_settings: Settings) -> tuple[str, str] | None:
    """Return Gradio credentials only after validating private mode."""

    validate_private_demo_settings(app_settings)
    if not app_settings.private_demo_mode:
        return None
    assert app_settings.gradio_auth_username is not None
    assert app_settings.gradio_auth_password is not None
    return (
        app_settings.gradio_auth_username,
        app_settings.gradio_auth_password,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info(
        "Starting Gradio UI host=%s port=%s api=%s",
        default_settings.gradio_host,
        default_settings.gradio_port,
        default_settings.flask_api_url,
    )
    demo = create_ui()
    demo.launch(
        server_name=default_settings.gradio_host,
        server_port=default_settings.gradio_port,
        share=False,
        theme=precision_lab_theme(),
        css=PRECISION_LAB_CSS,
        auth=private_demo_auth(default_settings),
        auth_message=None,
        max_file_size=f"{default_settings.max_upload_mb}mb",
        footer_links=[],
    )


if __name__ == "__main__":
    main()
