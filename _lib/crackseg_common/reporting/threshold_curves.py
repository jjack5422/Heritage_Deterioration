"""Render validation PR/ROC threshold curves as a directly viewable local PNG."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crackseg_common.thresholding import ThresholdPolicy


PRESET_COLOURS = {
    "sensitive": "#f59e0b",
    "balanced": "#2563eb",
    "clean": "#16a34a",
}


def curve_series(policy: ThresholdPolicy) -> dict[str, list[tuple[float, float, float]]]:
    """Return (x, y, threshold) points for validation PR and ROC curves."""

    rows = policy.validation.get("threshold_metrics", [])
    if not rows:
        raise ValueError("threshold policy 沒有 validation sweep，無法畫 PR/ROC curve")
    negative_pixels = int(policy.validation.get("negative_pixels", 0))
    pr, roc = [], []
    for row in sorted(rows, key=lambda item: float(item["threshold"])):
        values = {key: float(row[key]) for key in ("threshold", "precision", "recall")}
        fpr = row.get("fpr")
        if fpr is None:
            if negative_pixels <= 0:
                raise ValueError("threshold policy 缺少 negative_pixels/FPR")
            fpr = int(row["fp"]) / negative_pixels
        if not all(math.isfinite(value) for value in (*values.values(), float(fpr))):
            raise ValueError("threshold sweep 含非有限 PR/ROC 數值")
        pr.append((values["recall"], values["precision"], values["threshold"]))
        roc.append((float(fpr), values["recall"], values["threshold"]))
    return {"pr": pr, "roc": roc}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    for offset in np.arange(0, length, 18):
        stop = min(offset + 10, length)
        draw.line(
            (
                x1 + (x2 - x1) * offset / length,
                y1 + (y2 - y1) * offset / length,
                x1 + (x2 - x1) * stop / length,
                y1 + (y2 - y1) * stop / length,
            ),
            fill=fill,
            width=2,
        )


def _draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float, float]],
    preset_points: dict[str, tuple[float, float, float]],
    reference: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    left, top, right, bottom = box
    tick_font, label_font, title_font = _font(18), _font(22), _font(28)

    def xy(x_value: float, y_value: float) -> tuple[float, float]:
        return (
            left + min(1.0, max(0.0, x_value)) * (right - left),
            bottom - min(1.0, max(0.0, y_value)) * (bottom - top),
        )

    draw.text((left + 180, top - 52), title, fill="#111827", font=title_font)
    for index in range(6):
        value = index / 5
        x, y = xy(value, value)
        draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.text((x - 10, bottom + 10), f"{value:.1f}", fill="#4b5563", font=tick_font)
        draw.text((left - 45, y - 10), f"{value:.1f}", fill="#4b5563", font=tick_font)
    draw.line((left, top, left, bottom), fill="#111827", width=3)
    draw.line((left, bottom, right, bottom), fill="#111827", width=3)
    draw.text(((left + right) // 2 - 35, bottom + 42), x_label, fill="#111827", font=label_font)
    draw.text((left - 55, top - 34), y_label, fill="#111827", font=label_font)
    if reference:
        _dashed_line(draw, xy(*reference[0]), xy(*reference[1]), "#9ca3af")

    coordinates = [xy(x, y) for x, y, _ in points]
    if len(coordinates) > 1:
        draw.line(coordinates, fill="#374151", width=4, joint="curve")
    for x, y in coordinates:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#374151")
    for index, (name, (x_value, y_value, threshold)) in enumerate(preset_points.items()):
        colour = PRESET_COLOURS[name]
        x, y = xy(x_value, y_value)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colour, outline="white", width=2)
        offset_y = -30 if name == "sensitive" else 10
        draw.text((x + 10, y + offset_y), f"{name.title()} t={threshold:.2f}", fill=colour, font=tick_font)


def render_pr_roc_curve(policy: ThresholdPolicy, path: str | Path) -> Path:
    """Write side-by-side validation PR and ROC curves with preset operating points."""

    series = curve_series(policy)
    positive = int(policy.validation.get("positive_pixels", 0))
    negative = int(policy.validation.get("negative_pixels", 0))
    prevalence = positive / (positive + negative) if positive + negative else 0.0
    pr_presets = {
        name: (metric.recall, metric.precision, metric.threshold)
        for name, metric in policy.presets.items()
        if name in PRESET_COLOURS
    }
    roc_presets = {
        name: (metric.fpr, metric.recall, metric.threshold)
        for name, metric in policy.presets.items()
        if name in PRESET_COLOURS
    }
    canvas = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 24), f"{policy.expert.title()} Validation Threshold Curves", fill="#111827", font=_font(36))
    draw.text(
        (70, 70),
        f"Pixel-micro validation; foreground prevalence={prevalence:.4f}. PR is the primary view for imbalanced segmentation.",
        fill="#4b5563",
        font=_font(20),
    )
    _draw_chart(
        draw,
        (90, 155, 745, 660),
        "Precision–Recall (primary)",
        "Recall",
        "Precision",
        series["pr"],
        pr_presets,
        ((0.0, prevalence), (1.0, prevalence)),
    )
    _draw_chart(
        draw,
        (875, 155, 1530, 660),
        "ROC (secondary)",
        "False Positive Rate",
        "True Positive Rate",
        series["roc"],
        roc_presets,
        ((0.0, 0.0), (1.0, 1.0)),
    )
    draw.text((80, 738), "Preset operating points", fill="#111827", font=_font(26))
    for index, name in enumerate(("sensitive", "balanced", "clean")):
        metric = policy.presets[name]
        x = 80 + index * 505
        draw.rectangle((x, 785, x + 18, 803), fill=PRESET_COLOURS[name])
        draw.text((x + 28, 772), name.title(), fill="#111827", font=_font(22))
        draw.text(
            (x + 28, 812),
            f"t={metric.threshold:.2f}  P={metric.precision:.3f}  R={metric.recall:.3f}  F1={metric.f1:.3f}  FPR={metric.fpr:.4f}",
            fill="#374151",
            font=_font(18),
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    return destination


def write_tensorboard_curve(writer: Any, image_path: Path, step: int) -> None:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"))
    writer.add_image("threshold/pr_roc_curve", array, step, dataformats="HWC")
    writer.flush()
