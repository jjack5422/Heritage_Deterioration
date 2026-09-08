"""Environment-backed configuration for the segmentation services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_SAM2_ADAPTER_WEIGHT_ROOT = WORKSPACE_ROOT / (
    "sam2_adapter/runs/"
    "2026-08-22_merged-crack_0820-splits_bg1-fg2_"
    "sam2-adapter-hiera-large_seed42/5fold/foreground/fold0/"
    "artifacts/checkpoints"
)
DEFAULT_SAM3_ADAPTER_WEIGHT_ROOT = WORKSPACE_ROOT / (
    "sam3_adapter/runs/2026-08-28_sam3-adapter-512_seed42/"
    "5fold/sam3_adapter/fold0/artifacts/checkpoints"
)
DEFAULT_DA_SAM3_WEIGHT_ROOT = WORKSPACE_ROOT / (
    "dual_adapter_sam3/runs/"
    "2026-09-04_visual-da-sam3-full-decoder-joint-512_seed42/"
    "5fold/visual_da_sam3/fold0/artifacts/checkpoints"
)
DEFAULT_RESUNET50_WEIGHT_ROOT = WORKSPACE_ROOT / (
    "unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_"
    "resunet50_seed42/5fold/foreground/fold0/artifacts/checkpoints"
)
DEFAULT_CONVNEXT_UNET_WEIGHT_ROOT = WORKSPACE_ROOT / (
    "unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_"
    "convnext-large_seed42/5fold/foreground/fold0/artifacts/checkpoints"
)


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "true" if default else "false").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _optional_string(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _configured_path(name: str, default: Path) -> Path:
    """Resolve relative model paths from the workspace instead of the shell CWD."""

    path = Path(os.getenv(name, str(default))).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    model_root: Path
    flask_host: str
    flask_port: int
    gradio_host: str
    gradio_port: int
    flask_api_url: str
    max_upload_mb: int
    request_timeout_seconds: float
    sam2_base_checkpoint: Path | None = None
    sam3_base_checkpoint: Path | None = None
    sam2_adapter_weight_root: Path | None = None
    sam3_adapter_weight_root: Path | None = None
    da_sam3_weight_root: Path | None = None
    resunet50_weight_root: Path | None = None
    convnext_unet_weight_root: Path | None = None
    inference_tile_size: int = 512
    inference_stride: int = 384
    inference_batch_size: int = 1
    sam3_input_size: int = 512
    private_demo_mode: bool = False
    gradio_auth_username: str | None = None
    gradio_auth_password: str | None = None
    internal_api_key: str | None = None
    inference_rate_limit_requests: int = 30
    inference_rate_limit_window_seconds: int = 600
    gradio_queue_max_size: int = 2
    max_image_pixels: int = 2048 * 2048
    max_image_side: int = 2048

    def model_weight_roots(self) -> Mapping[str, Path]:
        """Return only explicitly configured model-specific checkpoint roots."""

        candidates = {
            "sam2_adapter": self.sam2_adapter_weight_root,
            "sam3_adapter": self.sam3_adapter_weight_root,
            "da_sam3": self.da_sam3_weight_root,
            "resunet50": self.resunet50_weight_root,
            "convnext_unet": self.convnext_unet_weight_root,
        }
        return {
            model_id: path
            for model_id, path in candidates.items()
            if path is not None
        }


def validate_private_demo_settings(app_settings: Settings) -> None:
    """Reject an enabled private deployment unless every secret is usable."""

    if not app_settings.private_demo_mode:
        return
    if app_settings.flask_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("private demo requires Flask to bind to loopback")
    if app_settings.gradio_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("private demo requires Gradio to bind to loopback")
    if not app_settings.gradio_auth_username:
        raise ValueError("private demo requires GRADIO_AUTH_USERNAME")
    password = app_settings.gradio_auth_password or ""
    if not password:
        raise ValueError("private demo requires a non-empty GRADIO_AUTH_PASSWORD")
    api_key = app_settings.internal_api_key or ""
    if len(api_key) < 32 or api_key.lower().startswith(("change-me", "replace-with")):
        raise ValueError(
            "private demo requires INTERNAL_API_KEY with at least 32 characters"
        )


def load_settings() -> Settings:
    """Load settings from ``.env`` and the current process environment."""

    load_dotenv(PROJECT_ROOT / ".env")
    flask_host = os.getenv("FLASK_HOST", "127.0.0.1")
    flask_port = _positive_int("FLASK_PORT", 5000)
    api_url = os.getenv(
        "FLASK_API_URL", f"http://{flask_host}:{flask_port}"
    ).rstrip("/")
    tile_size = _positive_int("INFERENCE_TILE_SIZE", 512)
    stride = _positive_int("INFERENCE_STRIDE", 384)
    if stride > tile_size:
        raise ValueError("INFERENCE_STRIDE must not exceed INFERENCE_TILE_SIZE")
    sam3_input_size = _positive_int("SAM3_INPUT_SIZE", 512)
    if sam3_input_size != 512:
        raise ValueError("SAM3_INPUT_SIZE must be 512 for the selected web checkpoint")
    app_settings = Settings(
        model_root=Path(os.getenv("MODEL_ROOT", "/data/models")).expanduser(),
        flask_host=flask_host,
        flask_port=flask_port,
        gradio_host=os.getenv("GRADIO_HOST", "127.0.0.1"),
        gradio_port=_positive_int("GRADIO_PORT", 7860),
        flask_api_url=api_url,
        max_upload_mb=_positive_int("MAX_UPLOAD_MB", 10),
        request_timeout_seconds=_positive_float("REQUEST_TIMEOUT_SECONDS", 120.0),
        sam2_base_checkpoint=_configured_path(
            "SAM2_BASE_CHECKPOINT",
            WORKSPACE_ROOT / "segment-anything-2/checkpoints/sam2.1_hiera_large.pt",
        ),
        sam3_base_checkpoint=_configured_path(
            "SAM3_BASE_CHECKPOINT",
            WORKSPACE_ROOT / "segment-anything-3/checkpoints/sam3.pt",
        ),
        sam2_adapter_weight_root=_configured_path(
            "SAM2_ADAPTER_WEIGHT_ROOT", DEFAULT_SAM2_ADAPTER_WEIGHT_ROOT
        ),
        sam3_adapter_weight_root=_configured_path(
            "SAM3_ADAPTER_WEIGHT_ROOT", DEFAULT_SAM3_ADAPTER_WEIGHT_ROOT
        ),
        da_sam3_weight_root=_configured_path(
            "DA_SAM3_WEIGHT_ROOT", DEFAULT_DA_SAM3_WEIGHT_ROOT
        ),
        resunet50_weight_root=_configured_path(
            "RESUNET50_WEIGHT_ROOT", DEFAULT_RESUNET50_WEIGHT_ROOT
        ),
        convnext_unet_weight_root=_configured_path(
            "CONVNEXT_UNET_WEIGHT_ROOT", DEFAULT_CONVNEXT_UNET_WEIGHT_ROOT
        ),
        inference_tile_size=tile_size,
        inference_stride=stride,
        inference_batch_size=_positive_int("INFERENCE_BATCH_SIZE", 1),
        sam3_input_size=sam3_input_size,
        private_demo_mode=_boolean("PRIVATE_DEMO_MODE", False),
        gradio_auth_username=_optional_string("GRADIO_AUTH_USERNAME"),
        gradio_auth_password=_optional_string("GRADIO_AUTH_PASSWORD"),
        internal_api_key=_optional_string("INTERNAL_API_KEY"),
        inference_rate_limit_requests=_positive_int(
            "INFERENCE_RATE_LIMIT_REQUESTS", 30
        ),
        inference_rate_limit_window_seconds=_positive_int(
            "INFERENCE_RATE_LIMIT_WINDOW_SECONDS", 600
        ),
        gradio_queue_max_size=_positive_int("GRADIO_QUEUE_MAX_SIZE", 2),
        max_image_pixels=_positive_int("MAX_IMAGE_PIXELS", 2048 * 2048),
        max_image_side=_positive_int("MAX_IMAGE_SIDE", 2048),
    )
    validate_private_demo_settings(app_settings)
    return app_settings


settings = load_settings()
