"""Reusable image validation, mask conversion, and overlay helpers."""

from __future__ import annotations

import base64
import io
import warnings
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class InvalidImageError(ValueError):
    """Raised when uploaded image content is malformed or unsupported."""


class ImageTooLargeError(InvalidImageError):
    """Raised before inference when decoded image dimensions exceed limits."""


def configure_image_decompression_limit(max_pixels: int) -> None:
    """Make Pillow reject compressed images before an oversized decode."""

    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    Image.MAX_IMAGE_PIXELS = max_pixels
    warnings.filterwarnings(
        "error",
        category=Image.DecompressionBombWarning,
    )


def _validate_dimensions(
    size: tuple[int, int],
    *,
    max_pixels: int | None,
    max_side: int | None,
) -> None:
    width, height = size
    if max_side is not None and (width > max_side or height > max_side):
        raise ImageTooLargeError("Image dimensions too large")
    if max_pixels is not None and width * height > max_pixels:
        raise ImageTooLargeError("Image dimensions too large")


def validate_image(
    source: BinaryIO | Image.Image,
    *,
    max_pixels: int | None = None,
    max_side: int | None = None,
) -> Image.Image:
    """Validate an uploaded image and return a detached RGB copy."""

    if isinstance(source, Image.Image):
        try:
            _validate_dimensions(
                source.size,
                max_pixels=max_pixels,
                max_side=max_side,
            )
            source.load()
            return source.convert("RGB").copy()
        except ImageTooLargeError:
            raise
        except (OSError, ValueError) as exc:
            raise InvalidImageError("Invalid image") from exc

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        with Image.open(source) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidImageError(
                    "Unsupported image format. Use JPEG, PNG, or WEBP."
                )
            _validate_dimensions(
                opened.size,
                max_pixels=max_pixels,
                max_side=max_side,
            )
            opened.load()
            return opened.convert("RGB").copy()
    except InvalidImageError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageTooLargeError("Image dimensions too large") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("Invalid image") from exc


def binary_mask_to_pil(mask: np.ndarray | Image.Image) -> Image.Image:
    """Normalize a 2D mask to an 8-bit PIL image containing only 0 and 255."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Binary mask must be a two-dimensional array")
    binary = np.where(array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def make_overlay(
    image: Image.Image,
    mask: np.ndarray | Image.Image,
    *,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a colored segmentation mask on an RGB source image."""

    source = image.convert("RGB")
    mask_image = binary_mask_to_pil(mask)
    if mask_image.size != source.size:
        raise ValueError("Mask dimensions must match the source image")
    colored = Image.new("RGB", source.size, color)
    highlighted = Image.blend(source, colored, alpha)
    return Image.composite(highlighted, source, mask_image)


def pil_to_png_base64(image: Image.Image) -> str:
    """Encode a PIL image as Base64 PNG text."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_png_to_pil(encoded: str) -> Image.Image:
    """Decode Base64 PNG text into a detached RGB PIL image."""

    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != "PNG":
                raise InvalidImageError("Inference response was not a PNG image")
            opened.load()
            return opened.convert("RGB").copy()
    except InvalidImageError:
        raise
    except (ValueError, UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Invalid Base64 PNG response") from exc
