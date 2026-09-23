"""Build class-stratified validation galleries for Dual-Adapter SAM3."""

from __future__ import annotations

import csv
import hashlib
import html
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


CLASS_LABELS = {
    "loss": "loss",
    "crack_craquelure": "craquelure",
    "craquelure": "craquelure",
}
PAGE_FILES = {
    "loss_best": "loss_best_20.html",
    "loss_worst": "loss_worst_20.html",
    "craquelure_best": "craquelure_best_20.html",
    "craquelure_worst": "craquelure_worst_20.html",
    "false_positives": "false_positives.html",
}
TENSORBOARD_GROUP_DIRS = {
    "loss_best": Path("loss") / "best",
    "loss_worst": Path("loss") / "worst",
    "craquelure_best": Path("craquelure") / "best",
    "craquelure_worst": Path("craquelure") / "worst",
    "false_positives": Path("false_positives"),
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _class_key(row: dict[str, Any]) -> str | None:
    label = CLASS_LABELS.get(str(row.get("target_class", "")))
    return label if label in {"loss", "craquelure"} else None


def rank_validation_rows(
    rows: Iterable[dict[str, Any]],
    *,
    top_k: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Split positive-GT rankings by class and isolate empty-GT false positives."""

    materialized = [dict(row) for row in rows]
    result: dict[str, list[dict[str, Any]]] = {
        "loss_best": [],
        "loss_worst": [],
        "craquelure_best": [],
        "craquelure_worst": [],
        "false_positives": [],
    }
    for class_name in ("loss", "craquelure"):
        positive = [
            row
            for row in materialized
            if _class_key(row) == class_name
            and _integer(row.get("gt_pixels")) > 0
            and _finite(row.get("f1")) is not None
        ]
        result[f"{class_name}_best"] = sorted(
            positive,
            key=lambda row: (-float(_finite(row.get("f1")) or 0.0), str(row.get("image", ""))),
        )[:top_k]
        result[f"{class_name}_worst"] = sorted(
            positive,
            key=lambda row: (float(_finite(row.get("f1")) or 0.0), str(row.get("image", ""))),
        )[:top_k]
    result["false_positives"] = sorted(
        (
            row
            for row in materialized
            if _class_key(row) is not None
            and _integer(row.get("gt_pixels")) == 0
            and _integer(row.get("pred_pixels")) > 0
        ),
        key=lambda row: (
            -_integer(row.get("pred_pixels")),
            -_integer(row.get("fp")),
            str(row.get("target_class", "")),
            str(row.get("image", "")),
        ),
    )[:top_k]
    return result


def _safe_source(run: Path, relative: Any) -> Path:
    source = (run / str(relative)).resolve()
    root = run.resolve()
    if root not in source.parents or not source.is_file():
        raise RuntimeError(f"unsafe or missing qualitative image: {relative}")
    return source


def _write_composite(run: Path, row: dict[str, Any], destination: Path) -> None:
    panels: list[Image.Image] = []
    try:
        for column in ("input_path", "gt_path", "prediction_path", "overlay_path"):
            panels.append(Image.open(_safe_source(run, row.get(column))).convert("RGB"))
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        left = 0
        for panel in panels:
            canvas.paste(panel, (left, 0))
            left += panel.width
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
    finally:
        for panel in panels:
            panel.close()


def _safe_name(row: dict[str, Any]) -> str:
    raw = f"{row.get('target_class', '')}_{Path(str(row.get('image', 'image'))).stem}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "image"
    digest = hashlib.sha1(str(row.get("image", "")).encode("utf-8")).hexdigest()[:8]
    return f"{slug[:100]}_{digest}"


def _prepare_assets(
    run: Path,
    reports: Path,
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    report_manifest_rows: list[dict[str, Any]] = []
    tensorboard_manifest_rows: list[dict[str, Any]] = []
    tensorboard_images = run / "tensorboard" / "images"
    for group, rows in groups.items():
        prepared[group] = []
        tensorboard_group = tensorboard_images / TENSORBOARD_GROUP_DIRS[group]
        tensorboard_group.mkdir(parents=True, exist_ok=True)
        for stale in tensorboard_group.glob("*.png"):
            stale.unlink()
        for rank, source_row in enumerate(rows, start=1):
            row = dict(source_row)
            filename = f"{rank:02d}_{_safe_name(row)}.png"
            tensorboard_relative = TENSORBOARD_GROUP_DIRS[group] / filename
            tensorboard_destination = tensorboard_images / tensorboard_relative
            _write_composite(run, row, tensorboard_destination)
            report_relative = Path("assets") / "ranked" / group / filename
            report_destination = reports / report_relative
            report_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tensorboard_destination, report_destination)
            row["rank"] = rank
            row["report_path"] = report_relative.as_posix()
            row["tensorboard_path"] = tensorboard_relative.as_posix()
            row["display_class"] = _class_key(row) or str(row.get("target_class", ""))
            prepared[group].append(row)
            record = {
                "group": group,
                "rank": rank,
                "target_class": row.get("target_class", ""),
                "image": row.get("image", ""),
                "f1": row.get("f1", ""),
                "gt_pixels": row.get("gt_pixels", ""),
                "pred_pixels": row.get("pred_pixels", ""),
                "fp": row.get("fp", ""),
            }
            report_manifest_rows.append({**record, "path": report_relative.as_posix()})
            tensorboard_manifest_rows.append({**record, "path": tensorboard_relative.as_posix()})
    fields = ("group", "rank", "target_class", "image", "f1", "gt_pixels", "pred_pixels", "fp", "path")
    with (reports / "ranking_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_manifest_rows)
    with (tensorboard_images / "class_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tensorboard_manifest_rows)
    return prepared


def _navigation() -> str:
    links = (
        ("Overview", "index.html"),
        ("loss Best", PAGE_FILES["loss_best"]),
        ("loss Worst", PAGE_FILES["loss_worst"]),
        ("craquelure Best", PAGE_FILES["craquelure_best"]),
        ("craquelure Worst", PAGE_FILES["craquelure_worst"]),
        ("False positives", PAGE_FILES["false_positives"]),
    )
    return "".join(f'<a href="{path}">{label}</a>' for label, path in links)


def _cards(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No matching validation examples.</p>'
    cards = []
    for row in rows:
        score = _finite(row.get("f1"))
        f1_text = f"{score:.4f}" if score is not None else "n/a"
        image_id = html.escape(str(row.get("image", "")))
        cards.append(
            f'<figure><img src="{html.escape(str(row["report_path"]))}" '
            f'alt="{html.escape(str(row["display_class"]))} rank {row["rank"]}">'
            f'<figcaption><strong>Rank {row["rank"]} · Target: {html.escape(str(row["display_class"]))}</strong><br>'
            f'F1: {f1_text} · GT pixels: {_integer(row.get("gt_pixels")):,} · '
            f'Prediction pixels: {_integer(row.get("pred_pixels")):,}<br><span>{image_id}</span></figcaption></figure>'
        )
    return "\n".join(cards)


def _section(title: str, rows: list[dict[str, Any]]) -> str:
    return f'<section><h2>{html.escape(title)}</h2><div class="grid">{_cards(rows)}</div></section>'


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:auto;padding:24px;background:#f6f7f9;color:#17202a}}
nav{{display:flex;flex-wrap:wrap;gap:8px 18px;margin:18px 0 28px}}nav a{{color:#075985}}
section{{margin:30px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
figure{{margin:0;background:white;padding:12px;border-radius:9px;box-shadow:0 1px 5px #ccd}}
img{{max-width:100%;height:auto}}.curve{{background:white;padding:8px}}figcaption{{margin-top:9px;line-height:1.5}}
figcaption span{{font-size:.86rem;color:#566573;overflow-wrap:anywhere}}.note,.empty{{background:#fff;padding:14px;border-left:4px solid #0ea5e9}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.summary div{{background:#fff;padding:14px;border-radius:8px}}
</style></head><body><h1>{html.escape(title)}</h1><nav>{_navigation()}</nav>{body}</body></html>"""


def _write_html(path: Path, title: str, sections: list[tuple[str, list[dict[str, Any]]]]) -> None:
    body = '<p class="note">Panel order: Input · GT · Prediction · Overlay. Pink = GT; cyan = prediction.</p>'
    body += "".join(_section(heading, rows) for heading, rows in sections)
    path.write_text(_page(title, body), encoding="utf-8")


def _validate_links(reports: Path) -> None:
    for report in reports.glob("*.html"):
        contents = report.read_text(encoding="utf-8")
        for relative in set(re.findall(r'src="([^"]+)"', contents)):
            if not (reports / relative).is_file():
                raise RuntimeError(f"broken image path in {report.name}: {relative}")


def build_class_report(run_dir: str | Path) -> Path:
    """Replace the generic galleries with auditable class-specific report pages."""

    run = Path(run_dir).resolve()
    validation_path = run / "metrics" / "per_image_validation.csv"
    if not validation_path.is_file():
        raise RuntimeError(f"validation metrics are missing: {validation_path}")
    with validation_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = rank_validation_rows(rows)
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    curve_source = run / "tensorboard" / "images" / "loss_curve.png"
    if not curve_source.is_file():
        raise RuntimeError(f"loss curve is missing: {curve_source}")
    assets = reports / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(curve_source, assets / "loss_curve.png")
    groups = _prepare_assets(run, reports, groups)

    pages = {
        "loss_best": ("loss Best 20", [("loss Best", groups["loss_best"])]),
        "loss_worst": ("loss Worst 20", [("loss Worst", groups["loss_worst"])]),
        "craquelure_best": ("craquelure Best 20", [("craquelure Best", groups["craquelure_best"])]),
        "craquelure_worst": ("craquelure Worst 20", [("craquelure Worst", groups["craquelure_worst"])]),
        "false_positives": ("空白 GT False Positives", [("Prediction pixels 由多至少", groups["false_positives"])]),
    }
    for key, (title, sections) in pages.items():
        _write_html(reports / PAGE_FILES[key], title, sections)
    _write_html(
        reports / "best_20.html",
        "Best validation examples by class",
        [("loss Best 20", groups["loss_best"]), ("craquelure Best 20", groups["craquelure_best"])],
    )
    _write_html(
        reports / "worst_20.html",
        "Worst validation examples by class",
        [("loss Worst 20", groups["loss_worst"]), ("craquelure Worst 20", groups["craquelure_worst"])],
    )
    counts = "".join(
        f'<div><strong>{html.escape(key.replace("_", " "))}</strong><br>{len(value)} images</div>'
        for key, value in groups.items()
    )
    overview = (
        '<p class="note">Best/Worst rankings include only samples with positive GT and are split by target class. '
        'Correct negatives are excluded. Empty-GT predictions are listed only under False positives.</p>'
        f'<section><h2>Loss curve</h2><img class="curve" src="assets/loss_curve.png" alt="Training and validation loss curve"></section>'
        f'<section><h2>Gallery summary</h2><div class="summary">{counts}</div></section>'
    )
    (reports / "index.html").write_text(_page("Dual-Adapter SAM3 validation report", overview), encoding="utf-8")
    _validate_links(reports)
    return reports / "index.html"


__all__ = ["build_class_report", "rank_validation_rows"]
