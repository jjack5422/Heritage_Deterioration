#!/usr/bin/env python3
"""Build a portable five-model OOF comparison report for merged craquelure."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence


METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("iou", "IoU"),
    ("micro_f1", "Micro-F1"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
)


class ModelSpec(NamedTuple):
    key: str
    name: str
    family: str
    run_dir: Path
    color: str


def default_model_specs(repo_root: Path) -> list[ModelSpec]:
    return [
        ModelSpec(
            "resunet50",
            "ResNet50 U-Net",
            "U-Net",
            repo_root
            / "unet/runs/2026-08-20_merged-craquelure_oof-retest_resunet50_seed42",
            "#0f766e",
        ),
        ModelSpec(
            "convnext_large",
            "ConvNeXt-Large U-Net",
            "U-Net",
            repo_root
            / "unet/runs/2026-08-20_merged-craquelure_oof-retest_convnext-large_seed42",
            "#2563eb",
        ),
        ModelSpec(
            "segformer_b5",
            "SegFormer-B5",
            "Transformer",
            repo_root
            / "segformer/runs/2026-08-20_merged-craquelure_oof-retest_segformer-b5_seed42",
            "#7c3aed",
        ),
        ModelSpec(
            "sam2_sac",
            "SAM2-SAC",
            "SAM2",
            repo_root
            / "sam2_sac/runs/2026-08-20_merged-craquelure_oof-retest_sam2-sac_seed42",
            "#dc2626",
        ),
        ModelSpec(
            "sam2_adapter",
            "SAM2-Adapter",
            "SAM2",
            repo_root
            / "sam2_adapter/runs/2026-08-20_merged-craquelure_oof-retest_sam2-adapter_seed42",
            "#d97706",
        ),
    ]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_image_f1(
    rows: Iterable[Mapping[str, str]], valid_pixels: Mapping[str, int]
) -> dict[str, float | int]:
    """Return image-equal and foreground-fraction-weighted foreground F1."""
    scores: list[float] = []
    weighted_total = 0.0
    total_weight = 0.0
    for row in rows:
        raw_f1 = str(row.get("f1", "")).strip()
        if not raw_f1:
            continue
        image = str(row["image"])
        valid = int(valid_pixels[image])
        if valid <= 0:
            raise ValueError(f"valid pixel count must be positive: {image}")
        score = float(raw_f1)
        gt_pixels = int(row["gt_pixels"])
        weight = gt_pixels / valid
        scores.append(score)
        weighted_total += weight * score
        total_weight += weight
    if not scores:
        raise ValueError("no positive-ground-truth images were available for F1 aggregation")
    return {
        "macro_f1": sum(scores) / len(scores),
        "weighted_f1": weighted_total / total_weight if total_weight else 0.0,
        "positive_images": len(scores),
        "weight_sum": total_weight,
    }


def _collect_image_names(model_specs: Sequence[ModelSpec]) -> set[str]:
    names: set[str] = set()
    for spec in model_specs:
        for fold in range(5):
            path = (
                spec.run_dir
                / "5fold"
                / "craquelure"
                / f"fold{fold}"
                / "metrics"
                / "per_image_outer_test.csv"
            )
            names.update(row["image"] for row in _read_csv(path))
    return names


def load_valid_pixels(dataset_root: Path, image_names: Iterable[str]) -> dict[str, int]:
    """Read exact valid-pixel counts from merged masks (labels 0 and 1 are scored)."""
    import numpy as np
    from PIL import Image

    counts: dict[str, int] = {}
    for name in sorted(set(image_names)):
        mask_path = dataset_root / "masks" / name
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        counts[name] = int(np.count_nonzero((mask == 0) | (mask == 1)))
    return counts


def _metric_record(tile_micro: Mapping[str, object]) -> dict[str, float]:
    return {
        "precision": float(tile_micro["mprecision"]),
        "recall": float(tile_micro["mrecall"]),
        "iou": float(tile_micro["miou"]),
        "micro_f1": float(tile_micro["mf1"]),
    }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_report_data(
    repo_root: Path,
    model_specs: Sequence[ModelSpec],
    valid_pixels: Mapping[str, int] | None = None,
) -> dict:
    if len(model_specs) != 5:
        raise ValueError(f"the comparison requires exactly five models, got {len(model_specs)}")
    for spec in model_specs:
        if not spec.run_dir.is_dir():
            raise FileNotFoundError(spec.run_dir)

    first_experiment = _read_json(model_specs[0].run_dir / "info" / "experiment.json")
    dataset_root = Path(first_experiment["dataset_root"])
    if valid_pixels is None:
        valid_pixels = load_valid_pixels(dataset_root, _collect_image_names(model_specs))

    models: list[dict] = []
    for spec in model_specs:
        oof = _read_json(spec.run_dir / "info" / "oof_summary.json")
        experiment = _read_json(spec.run_dir / "info" / "experiment.json")
        folds: list[dict] = []
        all_rows: list[dict[str, str]] = []
        summed_counts = {"tp": 0, "fp": 0, "fn": 0}

        for fold in range(5):
            fold_dir = spec.run_dir / "5fold" / "craquelure" / f"fold{fold}"
            outer = _read_json(fold_dir / "metrics" / "outer_test_metrics.json")
            rows = _read_csv(fold_dir / "metrics" / "per_image_outer_test.csv")
            epochs = _read_csv(fold_dir / "metrics" / "epochs.csv")
            image_f1 = aggregate_image_f1(rows, valid_pixels)
            tile = outer["tile_micro"]
            for key in summed_counts:
                summed_counts[key] += int(tile[key])
            metrics = _metric_record(tile)
            metrics.update(
                macro_f1=float(image_f1["macro_f1"]),
                weighted_f1=float(image_f1["weighted_f1"]),
            )
            checkpoints = outer.get("selected_checkpoints") or []
            selected_epoch = checkpoints[0].get("selected_epoch") if checkpoints else None
            loss_source = fold_dir / "tensorboard" / "images" / "loss_curve.png"
            if not loss_source.is_file():
                raise FileNotFoundError(
                    f"missing exported fold loss curve; run the TensorBoard image exporter: {loss_source}"
                )
            folds.append(
                {
                    "fold": fold,
                    "metrics": metrics,
                    "outer_loss": float(outer["loss"]),
                    "tile_count": len(rows),
                    "positive_images": int(image_f1["positive_images"]),
                    "epochs": len(epochs),
                    "selected_epoch": selected_epoch,
                    "loss_source": str(loss_source),
                }
            )
            all_rows.extend(rows)

        expected_counts = {key: int(oof["tile_micro"][key]) for key in summed_counts}
        if summed_counts != expected_counts:
            raise ValueError(
                f"fold confusion counts do not match OOF summary for {spec.name}: "
                f"{summed_counts} != {expected_counts}"
            )
        image_f1 = aggregate_image_f1(all_rows, valid_pixels)
        metrics = _metric_record(oof["tile_micro"])
        metrics.update(
            macro_f1=float(image_f1["macro_f1"]),
            weighted_f1=float(image_f1["weighted_f1"]),
        )
        models.append(
            {
                "key": spec.key,
                "name": spec.name,
                "family": spec.family,
                "color": spec.color,
                "run_dir": _relative_or_absolute(spec.run_dir, repo_root),
                "threshold": float(experiment.get("threshold", 0.5)),
                "tile_count": int(oof["outer_tile_count"]),
                "positive_images": int(image_f1["positive_images"]),
                "metrics": metrics,
                "folds": folds,
            }
        )

    return {
        "schema_version": 1,
        "title": "Merged Crack / Craquelure — Five-model OOF Report",
        "generated_date": "2026-08-20",
        "dataset_root": _relative_or_absolute(dataset_root, repo_root),
        "scope": "five-fold source-split out-of-fold retest; each image scored once",
        "metric_definitions": {
            "precision_recall_iou_micro_f1": "pooled foreground pixels across all OOF images",
            "macro_f1": "equal mean of per-image foreground F1 over positive-GT images",
            "weighted_f1": "per-image foreground F1 weighted by GT-foreground / valid-pixel fraction",
        },
        "models": models,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _point_shape(index: int, x: float, y: float, color: str) -> str:
    title = ""
    if index == 0:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>'
    if index == 1:
        return f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1" fill="{color}"/>'
    if index == 2:
        points = f"{x:.1f},{y - 5:.1f} {x + 5:.1f},{y:.1f} {x:.1f},{y + 5:.1f} {x - 5:.1f},{y:.1f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if index == 3:
        points = f"{x:.1f},{y - 5:.2f} {x + 5:.1f},{y + 4:.1f} {x - 5:.1f},{y + 4:.1f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    return (
        f'<path d="M {x - 4:.1f} {y - 4:.1f} L {x + 4:.1f} {y + 4:.1f} '
        f'M {x + 4:.1f} {y - 4:.1f} L {x - 4:.1f} {y + 4:.1f}" '
        f'stroke="{color}" stroke-width="2.6" stroke-linecap="round"/>'
    ) + title


def _fold_chart_svg(metric_key: str, label: str, models: Sequence[dict]) -> str:
    width, height = 920, 410
    left, top, plot_width, plot_height = 62, 34, 800, 260
    parts = [
        f'<svg class="fold-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(label)} across folds zero to four for five models">',
        f"<title>{html.escape(label)} — five-fold comparison</title>",
        f"<desc>Lines compare {html.escape(label)} from Fold 0 through Fold 4. Higher is better.</desc>",
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1.0 - tick) * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid-line"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{tick * 100:.0f}%</text>'
        )
    for fold in range(5):
        x = left + fold * plot_width / 4
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 27}" text-anchor="middle" class="axis-label">Fold {fold}</text>'
        )
    for model_index, model in enumerate(models):
        points = []
        for fold in model["folds"]:
            x = left + int(fold["fold"]) * plot_width / 4
            value = float(fold["metrics"][metric_key])
            y = top + (1.0 - value) * plot_height
            points.append((x, y, value, int(fold["fold"])))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{model["color"]}" '
            'stroke-width="2.7" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y, value, fold in points:
            parts.append(f'<g tabindex="0"><title>{html.escape(model["name"])} · Fold {fold}: {_pct(value)}</title>')
            parts.append(_point_shape(model_index, x, y, model["color"]))
            parts.append("</g>")
    for index, model in enumerate(models):
        row, col = divmod(index, 3)
        x, y = 72 + col * 275, 354 + row * 28
        parts.append(_point_shape(index, x, y - 4, model["color"]))
        parts.append(
            f'<text x="{x + 14}" y="{y}" class="legend-label">{html.escape(model["name"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _overview_metric_card(metric_key: str, label: str, models: Sequence[dict]) -> str:
    ranked = sorted(models, key=lambda model: model["metrics"][metric_key], reverse=True)
    rows = []
    for rank, model in enumerate(ranked, 1):
        value = float(model["metrics"][metric_key])
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-rank">{rank}</span><span class="bar-name">{html.escape(model["name"])}</span>'
            '<span class="bar-track" aria-hidden="true">'
            f'<span class="bar-fill" style="width:{value * 100:.3f}%;background:{model["color"]}"></span></span>'
            f'<span class="bar-value">{_pct(value)}</span></div>'
        )
    return (
        '<article class="metric-card">'
        f'<div class="metric-card-head"><h3>{html.escape(label)}</h3><span>higher is better</span></div>'
        + "".join(rows)
        + "</article>"
    )


def _summary_table(models: Sequence[dict]) -> str:
    best = {
        key: max(float(model["metrics"][key]) for model in models) for key, _ in METRICS
    }
    rows = []
    for model in sorted(models, key=lambda item: item["metrics"]["weighted_f1"], reverse=True):
        cells = []
        for key, _ in METRICS:
            value = float(model["metrics"][key])
            is_best = math.isclose(value, best[key], rel_tol=0, abs_tol=1e-12)
            cells.append(
                f'<td class="metric-cell{" best" if is_best else ""}" '
                f'style="--heat:{value:.4f}"><span>{_pct(value)}</span></td>'
            )
        rows.append(
            f'<tr><th scope="row"><span class="model-dot" style="background:{model["color"]}"></span>'
            f'{html.escape(model["name"])}<small>{html.escape(model["family"])}</small></th>'
            + "".join(cells)
            + "</tr>"
        )
    headers = "".join(f"<th scope=\"col\">{html.escape(label)}</th>" for _, label in METRICS)
    return (
        '<div class="table-wrap"><table class="summary-table"><thead><tr><th scope="col">Model</th>'
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _model_sections(models: Sequence[dict], loss_data_uris: Mapping[str, str]) -> str:
    sections = []
    for model in models:
        fold_rows = []
        for fold in model["folds"]:
            metric_cells = "".join(
                f'<td>{_pct(float(fold["metrics"][key]))}</td>' for key, _ in METRICS
            )
            epoch = fold["selected_epoch"]
            fold_rows.append(
                f'<tr><th scope="row">Fold {fold["fold"]}</th>{metric_cells}'
                f'<td>{float(fold["outer_loss"]):.4f}</td><td>{fold["tile_count"]}</td>'
                f'<td>{"—" if epoch is None else epoch}</td></tr>'
            )
        loss_figures = []
        for fold in model["folds"]:
            loss_key = f'{model["key"]}/fold{fold["fold"]}'
            loss_data_uri = loss_data_uris[loss_key]
            loss_figures.append(
                '<figure class="loss-figure">'
                f'<a href="{loss_data_uri}" target="_blank">'
                f'<img src="{loss_data_uri}" '
                f'alt="{html.escape(model["name"])} Fold {fold["fold"]} train and validation loss curve"></a>'
                f'<figcaption><strong>Fold {fold["fold"]}</strong><span>{fold["epochs"]} epochs · '
                f'selected {"epoch —" if fold["selected_epoch"] is None else f"epoch {fold["selected_epoch"]}"}</span></figcaption>'
                "</figure>"
            )
        headers = "".join(f"<th scope=\"col\">{label}</th>" for _, label in METRICS)
        sections.append(
            f'<section class="model-section" id="{model["key"]}">'
            f'<header class="model-header"><div><span class="eyebrow">{html.escape(model["family"])}</span>'
            f'<h2><span class="model-dot large" style="background:{model["color"]}"></span>{html.escape(model["name"])}</h2></div>'
            f'<div class="model-score"><span>OOF Weighted-F1</span><strong>{_pct(model["metrics"]["weighted_f1"])}</strong></div></header>'
            '<div class="table-wrap"><table class="fold-table"><thead><tr><th scope="col">Fold</th>'
            f'{headers}<th scope="col">Test loss</th><th scope="col">Images</th><th scope="col">Selected epoch</th>'
            f'</tr></thead><tbody>{"".join(fold_rows)}</tbody></table></div>'
            '<div class="loss-heading"><div><span class="eyebrow">Training diagnostics</span>'
            '<h3>Five independent loss curves</h3></div><p>Each fold is kept separate; no overlays or merged axes.</p></div>'
            f'<div class="loss-grid">{"".join(loss_figures)}</div></section>'
        )
    return "".join(sections)


def _render_html(data: dict, loss_data_uris: Mapping[str, str]) -> str:
    models = data["models"]
    metric_cards = "".join(_overview_metric_card(key, label, models) for key, label in METRICS)
    fold_charts = "".join(
        f'<article class="fold-chart-card"><h3>{html.escape(label)}</h3>{_fold_chart_svg(key, label, models)}</article>'
        for key, label in METRICS
    )
    model_nav = "".join(
        f'<a href="#{model["key"]}"><span style="background:{model["color"]}"></span>{html.escape(model["name"])}</a>'
        for model in models
    )
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Merged crack and craquelure five-model, five-fold out-of-fold evaluation report">
<title>五模型 OOF 評估報告 · Merged Crack / Craquelure</title>
<style>
:root {{ --ink:#18201f; --muted:#5f6b68; --paper:#f5f2e9; --card:#fffdf8; --line:#d9d4c8; --deep:#123c3a; --accent:#d97706; --shadow:0 14px 40px rgba(24,32,31,.09); }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,"Noto Sans TC","PingFang TC",system-ui,sans-serif; line-height:1.55; }}
a {{ color:inherit; }}
.topbar {{ position:sticky; top:0; z-index:20; display:flex; align-items:center; justify-content:space-between; gap:20px; padding:13px clamp(18px,4vw,56px); color:#ecf8f5; background:rgba(18,60,58,.96); backdrop-filter:blur(12px); box-shadow:0 6px 22px rgba(18,60,58,.2); }}
.brand {{ display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:.01em; text-decoration:none; }}
.brand-mark {{ width:14px; height:14px; border:3px solid #f5b642; border-radius:50%; box-shadow:8px 0 0 -3px #83d8c9; }}
.topbar nav {{ display:flex; gap:18px; font-size:.88rem; }}
.topbar nav a {{ color:#d9efea; text-decoration:none; }} .topbar nav a:hover {{ color:white; }}
.hero {{ position:relative; overflow:hidden; color:white; background:linear-gradient(130deg,#123c3a 0%,#185c55 58%,#0f766e 100%); padding:72px clamp(20px,6vw,92px) 56px; }}
.hero:after {{ content:""; position:absolute; width:520px; height:520px; right:-170px; top:-260px; border:1px solid rgba(255,255,255,.17); border-radius:45% 55% 60% 40%; transform:rotate(17deg); box-shadow:0 0 0 54px rgba(255,255,255,.035),0 0 0 108px rgba(255,255,255,.025); }}
.hero-inner {{ position:relative; z-index:1; max-width:1160px; margin:auto; }}
.kicker,.eyebrow {{ display:block; color:#c0e6df; font-size:.73rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
.hero h1 {{ max-width:850px; margin:12px 0 16px; font-family:Georgia,"Noto Serif TC",serif; font-size:clamp(2.35rem,5vw,5rem); line-height:1.03; letter-spacing:-.035em; }}
.hero p {{ max-width:780px; margin:0; color:#d8eeea; font-size:clamp(1rem,1.4vw,1.18rem); }}
.hero-stats {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); max-width:830px; margin-top:42px; border-top:1px solid rgba(255,255,255,.2); }}
.hero-stat {{ padding:19px 20px 0 0; }} .hero-stat strong {{ display:block; color:#ffcf6b; font-size:1.65rem; }} .hero-stat span {{ color:#cde7e2; font-size:.82rem; }}
.page {{ width:min(1320px,calc(100% - 36px)); margin:0 auto; }}
.section {{ padding:70px 0 0; }}
.section-head {{ display:flex; align-items:end; justify-content:space-between; gap:30px; margin-bottom:24px; }}
.section-head h2,.model-header h2 {{ margin:5px 0 0; font-family:Georgia,"Noto Serif TC",serif; font-size:clamp(1.8rem,3vw,2.8rem); line-height:1.12; }}
.section-head p {{ max-width:600px; margin:0; color:var(--muted); }}
.section .eyebrow,.model-section .eyebrow {{ color:#7a5a16; }}
.download-row {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 28px; }}
.download-row a {{ padding:8px 13px; border:1px solid var(--line); border-radius:999px; background:var(--card); color:var(--deep); font-size:.84rem; font-weight:700; text-decoration:none; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
.metric-card,.fold-chart-card,.model-section,.method-card {{ border:1px solid var(--line); border-radius:18px; background:var(--card); box-shadow:var(--shadow); }}
.metric-card {{ padding:20px 22px 18px; }}
.metric-card-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin-bottom:15px; }}
.metric-card h3,.fold-chart-card h3,.loss-heading h3 {{ margin:0; font-size:1.06rem; }} .metric-card-head span {{ color:var(--muted); font-size:.75rem; }}
.bar-row {{ display:grid; grid-template-columns:22px minmax(128px,1.1fr) minmax(120px,2fr) 64px; align-items:center; gap:9px; min-height:34px; font-size:.82rem; }}
.bar-rank {{ color:#8b8175; font-weight:800; }} .bar-name {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:650; }}
.bar-track {{ height:8px; overflow:hidden; border-radius:99px; background:#ebe6dc; }} .bar-fill {{ display:block; height:100%; border-radius:inherit; }}
.bar-value {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:800; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:15px; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; font-size:.86rem; font-variant-numeric:tabular-nums; }}
th,td {{ padding:13px 14px; border-bottom:1px solid #e7e2d8; text-align:right; white-space:nowrap; }}
thead th {{ color:#5d6664; background:#f1eee6; font-size:.74rem; letter-spacing:.025em; text-transform:uppercase; }}
tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
tbody th {{ text-align:left; }} tbody th small {{ display:block; color:var(--muted); font-weight:500; }}
.model-dot {{ display:inline-block; width:9px; height:9px; margin-right:9px; border-radius:50%; }} .model-dot.large {{ width:13px; height:13px; vertical-align:.18em; }}
.metric-cell {{ position:relative; background:color-mix(in srgb,var(--deep) calc(var(--heat)*13%),transparent); }} .metric-cell.best span {{ display:inline-block; padding:3px 7px; color:#fff; background:var(--deep); border-radius:6px; font-weight:800; }}
.fold-chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
.fold-chart-card {{ padding:20px 20px 10px; overflow:hidden; }} .fold-chart-card h3 {{ margin-left:9px; }}
.fold-svg {{ display:block; width:100%; min-width:460px; height:auto; }} .grid-line {{ stroke:#ddd7cc; stroke-width:1; stroke-dasharray:3 5; }} .axis-label {{ fill:#6b7471; font:12px system-ui,sans-serif; }} .legend-label {{ fill:#35403e; font:12px system-ui,sans-serif; }}
.model-nav {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }} .model-nav a {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border:1px solid var(--line); border-radius:999px; background:var(--card); font-size:.82rem; text-decoration:none; }} .model-nav span {{ width:8px; height:8px; border-radius:50%; }}
.model-section {{ scroll-margin-top:78px; margin-top:30px; padding:28px; }}
.model-header {{ display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:22px; }}
.model-score {{ min-width:170px; text-align:right; }} .model-score span {{ display:block; color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.06em; }} .model-score strong {{ color:var(--deep); font-family:Georgia,serif; font-size:2rem; }}
.fold-table th,.fold-table td {{ padding:11px 12px; }}
.loss-heading {{ display:flex; align-items:end; justify-content:space-between; gap:20px; margin:34px 0 15px; }} .loss-heading p {{ margin:0; color:var(--muted); font-size:.83rem; }}
.loss-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
.loss-figure {{ overflow:hidden; margin:0; border:1px solid #ded8cd; border-radius:13px; background:#fff; }} .loss-figure:last-child:nth-child(odd) {{ grid-column:1/-1; max-width:calc(50% - 8px); }}
.loss-figure a {{ display:block; background:#fff; }} .loss-figure img {{ display:block; width:100%; aspect-ratio:1.55; object-fit:contain; }}
.loss-figure figcaption {{ display:flex; justify-content:space-between; gap:12px; padding:10px 13px; border-top:1px solid #e6e1d7; font-size:.82rem; }} .loss-figure figcaption span {{ color:var(--muted); }}
.method-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }} .method-card {{ padding:22px; }} .method-card .number {{ display:block; color:#c36a08; font:700 1.7rem Georgia,serif; }} .method-card h3 {{ margin:6px 0; font-size:1rem; }} .method-card p {{ margin:0; color:var(--muted); font-size:.87rem; }}
.notice {{ margin-top:18px; padding:16px 19px; border-left:4px solid #0f766e; border-radius:0 10px 10px 0; background:#e8f3ef; color:#234844; }}
footer {{ margin-top:70px; padding:30px 20px 42px; color:#bcd5d0; background:var(--deep); text-align:center; font-size:.82rem; }}
@media (max-width:900px) {{ .metric-grid,.fold-chart-grid,.loss-grid,.method-grid {{ grid-template-columns:1fr; }} .loss-figure:last-child:nth-child(odd) {{ max-width:none; }} .hero-stats {{ grid-template-columns:repeat(2,1fr); }} .topbar nav {{ display:none; }} .section-head,.model-header,.loss-heading {{ align-items:flex-start; flex-direction:column; }} .model-score {{ text-align:left; }} }}
@media (max-width:560px) {{ .page {{ width:min(100% - 22px,1320px); }} .hero {{ padding-top:48px; }} .hero h1 {{ font-size:2.45rem; }} .metric-card,.model-section {{ padding:17px; }} .bar-row {{ grid-template-columns:18px minmax(112px,1fr) 58px; }} .bar-track {{ display:none; }} .loss-figure figcaption {{ align-items:flex-start; flex-direction:column; }} }}
@media print {{ .topbar {{ position:static; }} .hero {{ padding:38px; }} .page {{ width:100%; }} .section,.model-section {{ break-inside:avoid; }} .loss-figure {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header class="topbar"><a class="brand" href="#top"><span class="brand-mark"></span>CrackSeg · OOF Report</a><nav><a href="#overview">總覽</a><a href="#folds">五折</a><a href="#models">模型與 Loss</a><a href="#method">定義</a></nav></header>
<main id="top">
<section class="hero"><div class="hero-inner"><span class="kicker">Merged crack + craquelure · Five-fold OOF</span><h1>五模型完整評估報告</h1><p>同一份合併資料、同一組互斥五折與固定 threshold 0.5。每張影像只在其 outer fold 被評估一次，讓 U-Net、SegFormer 與 SAM2 系列可以直接比較。</p><div class="hero-stats"><div class="hero-stat"><strong>5</strong><span>model variants</span></div><div class="hero-stat"><strong>929</strong><span>unique OOF images</span></div><div class="hero-stat"><strong>25</strong><span>independent fold runs</span></div><div class="hero-stat"><strong>6</strong><span>reported metrics</span></div></div></div></section>
<div class="page">
<section class="section" id="overview"><div class="section-head"><div><span class="eyebrow">Executive comparison</span><h2>整體 OOF 指標</h2></div><p>前三項與 Micro-F1 由所有 outer-test 像素 pooled 計算；Macro-F1 與 Weighted-F1 從逐影像前景 F1 聚合。</p></div><div class="download-row"><a href="data/model_summary.csv" download>下載模型總表 CSV</a><a href="data/fold_metrics.csv" download>下載五折明細 CSV</a><a href="data/report_data.json" download>下載完整 JSON</a></div>{_summary_table(models)}<div class="metric-grid" style="margin-top:18px">{metric_cards}</div></section>
<section class="section" id="folds"><div class="section-head"><div><span class="eyebrow">Fold stability</span><h2>五折趨勢</h2></div><p>每張圖固定 Fold 0 → 4；顏色與點形同時區分模型，便於觀察跨域 fold 的落差。所有 Y 軸皆為 0–100%。</p></div><div class="fold-chart-grid">{fold_charts}</div></section>
<section class="section" id="models"><div class="section-head"><div><span class="eyebrow">Model dossiers</span><h2>每折數據與獨立 Loss</h2></div><p>每個模型保留五張 loss 圖，總計 25 張。點擊圖可開啟原始解析度。</p></div><nav class="model-nav" aria-label="Jump to model">{model_nav}</nav>{_model_sections(models, loss_data_uris)}</section>
<section class="section" id="method"><div class="section-head"><div><span class="eyebrow">Methodology</span><h2>指標定義與防洩漏</h2></div><p>以下定義同時套用整體 OOF 與各 fold，報告數值可由附帶 CSV / JSON 重算。</p></div><div class="method-grid"><article class="method-card"><span class="number">01</span><h3>Pixel-pooled</h3><p>Precision、Recall、IoU、Micro-F1 將評估範圍內 TP / FP / FN 相加後計算，反映整體前景像素表現。</p></article><article class="method-card"><span class="number">02</span><h3>Macro-F1</h3><p>先算每張 positive-GT 影像的 binary foreground F1，再對影像等權平均；純背景影像不納入。</p></article><article class="method-card"><span class="number">03</span><h3>Weighted-F1</h3><p>每張影像權重為 GT 前景像素 ÷ valid pixels，再對逐影像 foreground F1 加權；純背景影像權重為 0。</p></article></div><div class="notice"><strong>Selection firewall：</strong>outer test 未參與 checkpoint 或 threshold 選擇；checkpoint 沿用各來源 fold 僅依 inner validation 選出的結果，threshold 固定為 0.5。</div></section>
</div>
</main>
<footer>Generated 2026-08-20 · Dataset: {html.escape(data["dataset_root"])} · Static, offline-readable report</footer>
<script id="report-data" type="application/json">{embedded}</script>
</body>
</html>'''


def build_report(
    repo_root: Path,
    output_dir: Path,
    model_specs: Sequence[ModelSpec] | None = None,
    valid_pixels: Mapping[str, int] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    specs = list(model_specs or default_model_specs(repo_root))
    data = load_report_data(repo_root, specs, valid_pixels=valid_pixels)

    data_dir = output_dir / "data"
    loss_dir = output_dir / "assets" / "loss"
    data_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    fold_rows = []
    loss_data_uris: dict[str, str] = {}
    for model in data["models"]:
        model_rows.append(
            {
                "model": model["name"],
                "family": model["family"],
                "outer_images": model["tile_count"],
                "positive_images": model["positive_images"],
                **{key: model["metrics"][key] for key, _ in METRICS},
            }
        )
        target_model_dir = loss_dir / model["key"]
        target_model_dir.mkdir(parents=True, exist_ok=True)
        for fold in model["folds"]:
            target = target_model_dir / f"fold{fold['fold']}.png"
            shutil.copy2(Path(fold["loss_source"]), target)
            encoded = base64.b64encode(target.read_bytes()).decode("ascii")
            loss_data_uris[f'{model["key"]}/fold{fold["fold"]}'] = (
                f"data:image/png;base64,{encoded}"
            )
            fold_rows.append(
                {
                    "model": model["name"],
                    "fold": fold["fold"],
                    "outer_images": fold["tile_count"],
                    "positive_images": fold["positive_images"],
                    **{key: fold["metrics"][key] for key, _ in METRICS},
                    "outer_test_loss": fold["outer_loss"],
                    "training_epochs": fold["epochs"],
                    "selected_epoch": fold["selected_epoch"],
                }
            )
            fold["loss_asset"] = str(target.relative_to(output_dir))

    model_fields = [
        "model",
        "family",
        "outer_images",
        "positive_images",
        *(key for key, _ in METRICS),
    ]
    fold_fields = [
        "model",
        "fold",
        "outer_images",
        "positive_images",
        *(key for key, _ in METRICS),
        "outer_test_loss",
        "training_epochs",
        "selected_epoch",
    ]
    _write_csv_rows(data_dir / "model_summary.csv", model_fields, model_rows)
    _write_csv_rows(data_dir / "fold_metrics.csv", fold_fields, fold_rows)
    (data_dir / "report_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    index = output_dir / "index.html"
    index.write_text(_render_html(data, loss_data_uris), encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_comparisons/2026-08-20_merged-craquelure_5model_oof_report"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    if not output.is_absolute():
        output = args.repo_root / output
    index = build_report(args.repo_root, output)
    print(index)


if __name__ == "__main__":
    main()
