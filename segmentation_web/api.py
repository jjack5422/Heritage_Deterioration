"""Flask inference API for registered segmentation models."""

from __future__ import annotations

import hmac
import logging
import sys
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from config import (
    Settings,
    WORKSPACE_ROOT,
    settings as default_settings,
    validate_private_demo_settings,
)
from imaging.image_processing import (
    ImageTooLargeError,
    InvalidImageError,
    configure_image_decompression_limit,
    pil_to_png_base64,
    validate_image,
)


# ``python api.py`` starts with ``segmentation_web`` as the import root.  The
# trained U-Net/SAM adapters live in sibling project directories, so add the
# workspace root before importing the inference manager below.
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


from inference import InferenceManager, device_information
from registry import (
    InvalidWeightError,
    UnknownModelError,
    get_models,
    get_weights,
    resolve_deterioration_class,
)
from request_security import SlidingWindowRateLimiter


LOGGER = logging.getLogger(__name__)


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "error": message}), status_code


def _threshold(raw_value: str | None) -> float:
    try:
        value = float(raw_value) if raw_value is not None else 0.5
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid threshold") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError("Invalid threshold")
    return value


def create_app(
    *,
    settings: Settings | None = None,
    inference_manager: InferenceManager | None = None,
) -> Flask:
    """Build the API application with injectable configuration for tests."""

    app_settings = settings or default_settings
    validate_private_demo_settings(app_settings)
    configure_image_decompression_limit(app_settings.max_image_pixels)
    manager = inference_manager or InferenceManager(runtime_settings=app_settings)
    limiter = SlidingWindowRateLimiter(
        request_limit=app_settings.inference_rate_limit_requests,
        window_seconds=app_settings.inference_rate_limit_window_seconds,
    )
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = app_settings.max_upload_mb * 1024 * 1024
    app.extensions["inference_manager"] = manager
    app.extensions["inference_rate_limiter"] = limiter

    @app.before_request
    def authenticate_private_api():
        if not app_settings.private_demo_mode or not request.path.startswith("/api/"):
            return None
        expected = app_settings.internal_api_key
        supplied = request.headers.get("X-API-Key", "")
        if expected is None or not hmac.compare_digest(supplied, expected):
            LOGGER.warning("Rejected unauthorized API request path=%s", request.path)
            return _error("Unauthorized", 401)
        return None

    @app.errorhandler(RequestEntityTooLarge)
    def handle_oversized_upload(_error_details: RequestEntityTooLarge):
        return _error("Upload too large", 413)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", **device_information()})

    @app.get("/api/models")
    def models():
        return jsonify(get_models())

    @app.get("/api/models/<model_id>/weights")
    def model_weights(model_id: str):
        try:
            weight_roots = app_settings.model_weight_roots()
            return jsonify(
                get_weights(
                    model_id,
                    model_root=(app_settings.model_root if not weight_roots else None),
                    weight_roots=weight_roots or None,
                )
            )
        except UnknownModelError:
            return _error("Unknown model", 400)

    @app.post("/api/infer")
    def infer():
        admitted, retry_after = limiter.admit()
        if not admitted:
            response = jsonify(
                {"status": "error", "error": "Rate limit exceeded"}
            )
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        image_upload = request.files.get("image")
        model_id = request.form.get("model", "").strip()
        weight_name = request.form.get("weight", "").strip()
        deterioration_class = (
            request.form.get("deterioration_class", "").strip() or None
        )
        if image_upload is None:
            return _error("Image is required", 400)
        if not model_id:
            return _error("Model is required", 400)
        if not weight_name:
            return _error("Weight is required", 400)

        try:
            threshold = _threshold(request.form.get("threshold"))
            deterioration_class = resolve_deterioration_class(
                model_id,
                deterioration_class,
            )
            image = validate_image(
                image_upload.stream,
                max_pixels=app_settings.max_image_pixels,
                max_side=app_settings.max_image_side,
            )
            LOGGER.info(
                "Inference request model=%s weight=%s class=%s size=%sx%s",
                model_id,
                weight_name,
                deterioration_class,
                image.width,
                image.height,
            )
            result = manager.predict(
                model_id,
                weight_name,
                image,
                threshold,
                deterioration_class=deterioration_class,
            )
        except ImageTooLargeError:
            return _error("Image dimensions too large", 413)
        except InvalidImageError:
            return _error("Invalid image", 400)
        except ValueError as exc:
            if str(exc) == "Invalid threshold":
                return _error("Invalid threshold", 400)
            if isinstance(exc, UnknownModelError):
                return _error("Unknown model", 400)
            if isinstance(exc, InvalidWeightError):
                return _error(str(exc), 400)
            LOGGER.warning("Invalid inference request: %s", exc)
            return _error(str(exc), 400)
        except RuntimeError as exc:
            LOGGER.warning("Model unavailable: %s", exc)
            return _error(str(exc), 503)
        except Exception:
            LOGGER.exception("Inference failed")
            return _error("Inference failed", 500)

        response: dict[str, Any] = {
            "status": "success",
            "model": result["model"],
            "weight": result["weight"],
            "deterioration_class": result["deterioration_class"],
            "threshold": threshold,
            "latency_ms": round(float(result["latency_ms"]), 3),
            "device": result["device"],
            "mask_png_base64": pil_to_png_base64(result["mask"]),
            "overlay_png_base64": pil_to_png_base64(result["overlay"]),
        }
        return jsonify(response)

    return app


app = create_app()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    device = device_information()
    LOGGER.info(
        "Starting Flask API host=%s port=%s device=%s gpu=%s",
        default_settings.flask_host,
        default_settings.flask_port,
        device["device"],
        device["gpu"],
    )
    app.run(
        host=default_settings.flask_host,
        port=default_settings.flask_port,
        debug=False,
    )


if __name__ == "__main__":
    main()
