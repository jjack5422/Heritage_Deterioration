#!/usr/bin/env python3
"""Build validation/test four-panel reports for trained SAM3 expert runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sam2_adapter.h0_core import load_trainable_state_dict
from sam2_adapter.reporting import binary_metric_row
from sam2_adapter.runtime import _autocast, _batch_tensor
from sam3_adapter.expert_training_data import denormalize_image, prepare_expert_data_plan
from sam3_adapter.train import THRESHOLD, _build_model, _loader

EXPERTS = ("scratch_crack", "shrinkage_craquelure", "loss")
EXPERT_OUTPUT_NAMES = {
    "scratch_crack": "crack",
    "shrinkage_craquelure": "craquelure",
    "loss": "loss",
}
PARTITIONS = ("validation", "test")
PANEL_LABELS = ("Original", "Ground Truth", "Prediction", "Overlay")
DATASET115_ROOT = WORKSPACE_ROOT / "dataset115_filtered"
SOURCE_MANIFEST = DATASET115_ROOT / "metadata" / "manifest.csv"
PANEL_SIZE = 392
HEADER_HEIGHT = 32
LEGEND_HEIGHT = 48
COMPOSITE_SIZE = (PANEL_SIZE * 4, HEADER_HEIGHT + PANEL_SIZE + LEGEND_HEIGHT)
PREDICTION_COLOR = (37, 99, 235)
GROUND_TRUTH_CLASSES: tuple[tuple[str, tuple[str, ...], tuple[int, int, int]], ...] = (
    ("Crack / Scratch", ("D-01", "D-11"), (225, 29, 72)),
    ("Loss", ("D-02",), (245, 158, 11)),
    ("Craquelure / Shrinkage", ("D-03", "D-04"), (34, 197, 94)),
    ("Flaking / Peeling（起甲＋剝落）", ("D-05", "D-22"), (168, 85, 247)),
)
MANIFEST_FIELDS = (
    "partition",
    "expert",
    "model_expert",
    "checkpoint",
    "checkpoint_epoch",
    "image",
    "image_folder",
    "tile",
    "dataset",
    "source_group",
    "ground_truth_classes",
    "composite_path",
    "f1",
    "precision",
    "recall",
    "iou",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "gt_pixels",
    "pred_pixels",
    "error_reason",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Experiment root containing 1fold/<expert>/fold0")
    parser.add_argument("--output", type=Path, help="Destination; defaults to <run_root>/test_inference_report")
    parser.add_argument("--replace", action="store_true", help="Atomically replace an existing report")
    parser.add_argument("--experts", nargs="+", choices=EXPERTS, help="Experts to include; defaults to all")
    parser.add_argument("--checkpoint", choices=("best.pt", "last.pt"), default="best.pt")
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=["test"])
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/opentype/noto/"
        + ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc")
    )
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _source_mask_index() -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image = row["image"]
            code = row["mask_class_code"]
            if code in grouped[image]:
                raise RuntimeError(f"duplicate source mask: image={image}, code={code}")
            mask = (DATASET115_ROOT / row["mask"]).resolve()
            if not mask.is_file():
                raise FileNotFoundError(f"missing source mask: {mask}")
            grouped[image][code] = mask
    return dict(grouped)


def _load_ground_truth_masks(
    manifest_row: Mapping[str, Any],
    source_masks: Mapping[str, Mapping[str, Path]],
) -> dict[str, np.ndarray]:
    image_path = Path(str(manifest_row["image"])).resolve()
    try:
        image_key = image_path.relative_to(DATASET115_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"validation/test image is outside Dataset115: {image_path}") from error
    code_paths = source_masks.get(image_key)
    if code_paths is None:
        raise RuntimeError(f"source mask manifest has no image row: {image_key}")

    masks: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for label, codes, _ in GROUND_TRUTH_CLASSES:
        union: np.ndarray | None = None
        for code in codes:
            path = code_paths.get(code)
            if path is None:
                continue
            with Image.open(path) as handle:
                raw = np.asarray(handle.convert("L"), dtype=np.uint8)
            values = set(int(value) for value in np.unique(raw))
            if not values.issubset({0, 255}):
                raise RuntimeError(f"source mask is not binary: {path}, values={sorted(values)}")
            if shape is None:
                shape = raw.shape
            elif raw.shape != shape:
                raise RuntimeError(f"source mask shape mismatch for {image_key}: {raw.shape} != {shape}")
            current = raw == 255
            union = current if union is None else np.logical_or(union, current)
        if union is None:
            if shape is None:
                with Image.open(image_path) as handle:
                    shape = (handle.height, handle.width)
            union = np.zeros(shape, dtype=bool)
        masks[label] = union
    return masks


def _resize_rgb(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8, copy=False))
    return np.asarray(image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS), dtype=np.uint8)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(
        image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BOX),
        dtype=np.uint8,
    ) != 0


def _ground_truth_rgb(masks: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    resized = [_resize_mask(masks[label]) for label, _, _ in GROUND_TRUTH_CLASSES]
    stack = np.stack(resized, axis=0)
    counts = stack.sum(axis=0)
    ranks = np.cumsum(stack, axis=0) - 1
    yy, xx = np.indices(counts.shape)
    selector = np.zeros_like(counts)
    active = counts > 0
    selector[active] = (xx[active] + yy[active]) % counts[active]

    output = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    for index, (_, _, color) in enumerate(GROUND_TRUTH_CLASSES):
        selected = stack[index] & (ranks[index] == selector)
        output[selected] = color
    return output, active


def _prediction_rgb(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = _resize_mask(prediction)
    output = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    output[resized] = PREDICTION_COLOR
    return output, resized


def _overlay_rgb(
    input_rgb: np.ndarray,
    ground_truth_rgb: np.ndarray,
    ground_truth_active: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    base = input_rgb.astype(np.float32)
    prediction_blend = 0.52 * base + 0.48 * np.asarray(PREDICTION_COLOR, dtype=np.float32)
    ground_truth_blend = 0.52 * base + 0.48 * ground_truth_rgb.astype(np.float32)
    overlay = base.copy()
    overlay[prediction] = prediction_blend[prediction]
    overlay[ground_truth_active] = ground_truth_blend[ground_truth_active]

    overlap = prediction & ground_truth_active
    yy, xx = np.indices(overlap.shape)
    show_prediction = overlap & ((xx + yy) % 2 == 0)
    if overlap.any() and not show_prediction.any():
        row, column = np.argwhere(overlap)[0]
        show_prediction[row, column] = True
    overlay[show_prediction] = prediction_blend[show_prediction]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_legend(draw: ImageDraw.ImageDraw, *, top: int) -> None:
    font = _font(14)
    entries = [(label, color) for label, _, color in GROUND_TRUTH_CLASSES]
    entries.append(("Prediction", PREDICTION_COLOR))
    square = 14
    gap = 9
    entry_gap = 28
    widths = []
    for label, _ in entries:
        bounds = draw.textbbox((0, 0), label, font=font)
        widths.append(square + gap + bounds[2] - bounds[0])
    total_width = sum(widths) + entry_gap * (len(entries) - 1)
    x = (COMPOSITE_SIZE[0] - total_width) / 2
    square_top = top + (LEGEND_HEIGHT - square) / 2
    for (label, color), width in zip(entries, widths, strict=True):
        draw.rectangle((x, square_top, x + square, square_top + square), fill=color)
        bounds = draw.textbbox((0, 0), label, font=font)
        text_height = bounds[3] - bounds[1]
        draw.text(
            (x + square + gap, top + (LEGEND_HEIGHT - text_height) / 2 - bounds[1]),
            label,
            fill=(30, 41, 59),
            font=font,
        )
        x += width + entry_gap


def _save_composite(
    path: Path,
    *,
    input_rgb: np.ndarray,
    ground_truth_masks: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> tuple[str, ...]:
    input_panel = _resize_rgb(input_rgb)
    ground_truth_panel, ground_truth_active = _ground_truth_rgb(ground_truth_masks)
    prediction_panel, prediction_resized = _prediction_rgb(prediction)
    overlay_panel = _overlay_rgb(
        input_panel,
        ground_truth_panel,
        ground_truth_active,
        prediction_resized,
    )
    panels = (input_panel, ground_truth_panel, prediction_panel, overlay_panel)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"duplicate composite destination: {path}")
    composite = Image.new("RGB", COMPOSITE_SIZE, "white")
    draw = ImageDraw.Draw(composite)
    title_font = _font(17, bold=True)
    for index, (label, panel) in enumerate(zip(PANEL_LABELS, panels, strict=True)):
        left = index * PANEL_SIZE
        composite.paste(Image.fromarray(panel), (left, HEADER_HEIGHT))
        bounds = draw.textbbox((0, 0), label, font=title_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (left + (PANEL_SIZE - text_width) / 2, (HEADER_HEIGHT - text_height) / 2 - bounds[1]),
            label,
            fill=(30, 41, 59),
            font=title_font,
        )
        if index:
            draw.line((left, 0, left, HEADER_HEIGHT + PANEL_SIZE), fill=(226, 232, 240), width=1)
    legend_top = HEADER_HEIGHT + PANEL_SIZE
    draw.line((0, legend_top, COMPOSITE_SIZE[0], legend_top), fill=(226, 232, 240), width=1)
    _draw_legend(draw, top=legend_top)
    composite.save(path, format="PNG", optimize=True)
    return tuple(label for label, mask in ground_truth_masks.items() if bool(mask.any()))


def _load_reference_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["image"]: row for row in csv.DictReader(handle)}


def _assert_metric_counts(image: str, row: Mapping[str, Any], reference: Mapping[str, str]) -> None:
    for key in ("tp", "fp", "fn", "gt_pixels", "pred_pixels"):
        if int(row[key]) != int(reference[key]):
            raise RuntimeError(f"{image}: regenerated {key}={row[key]} differs from stored {reference[key]}")


def _partition_reference(fold_root: Path, partition: str, checkpoint_name: str) -> dict[str, dict[str, str]]:
    if checkpoint_name != "best.pt":
        return {}
    name = "per_image_validation.csv" if partition == "validation" else "outer_test_per_image.csv"
    return _load_reference_rows(fold_root / "metrics" / name)


def _inference_rows(
    run_root: Path,
    output_root: Path,
    expert: str,
    checkpoint_name: str,
    partitions: tuple[str, ...],
    source_masks: Mapping[str, Mapping[str, Path]],
) -> list[dict[str, Any]]:
    fold_root = run_root / "1fold" / expert / "fold0"
    config = _read_json(fold_root / "config" / "args.json")
    config["manifest"] = Path(config["manifest"])
    config["checkpoint"] = Path(config["checkpoint"])
    args = SimpleNamespace(**config)
    plan = prepare_expert_data_plan(args.manifest, expert)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the trained SAM3 inference path")
    device = torch.device("cuda")
    model = _build_model(args, device)
    checkpoint_path = fold_root / "artifacts" / "checkpoints" / checkpoint_name
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("expert") != expert
        or payload.get("dataset_sha256") != plan.adopted_dataset_sha256
        or payload.get("split_sha256") != plan.split_sha256
    ):
        raise RuntimeError(f"{expert}: checkpoint does not match the locked expert split")
    load_trainable_state_dict(model, payload["adaptation_state"])
    model.eval()

    rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for partition in partitions:
                partition_rows = tuple(getattr(plan, partition))
                loader = _loader(partition_rows, raw_ids=plan.raw_ids, train=False, args=args)
                locked_rows = {
                    f"{row['dataset']}__{row['tile']}": row
                    for row in partition_rows
                }
                reference_rows = _partition_reference(fold_root, partition, checkpoint_name)
                if reference_rows and len(reference_rows) != len(partition_rows):
                    raise RuntimeError(
                        f"{expert}/{partition}: stored metric rows do not match locked partition count"
                    )
                generated = 0
                for batch in loader:
                    images = _batch_tensor(batch, "image", device)
                    with _autocast(device, bool(args.amp)):
                        logits = model(images)
                    predictions = torch.sigmoid(logits[:, 0]) >= THRESHOLD
                    targets = _batch_tensor(batch, "target", device).long() == 1
                    for index in range(images.shape[0]):
                        image_id = str(batch["name"][index])
                        manifest_row = locked_rows.get(image_id)
                        if manifest_row is None:
                            raise RuntimeError(
                                f"{expert}: {image_id} missing from locked {partition} manifest"
                            )
                        image_folder = str(manifest_row["source_group"])
                        tile = Path(str(manifest_row["tile"])).stem
                        for label, value in (("image folder", image_folder), ("tile", tile)):
                            if not value or value in {".", ".."} or Path(value).name != value:
                                raise RuntimeError(f"{expert}: unsafe {label} name: {value!r}")
                        target = targets[index].cpu().numpy()
                        prediction = predictions[index].cpu().numpy()
                        input_rgb = (
                            denormalize_image(images[index])
                            .permute(1, 2, 0)
                            .mul(255)
                            .round()
                            .byte()
                            .numpy()
                        )
                        ground_truth_masks = _load_ground_truth_masks(manifest_row, source_masks)
                        relative_path = (
                            Path(partition)
                            / EXPERT_OUTPUT_NAMES[expert]
                            / image_folder
                            / f"{tile}.png"
                        )
                        classes_present = _save_composite(
                            output_root / relative_path,
                            input_rgb=input_rgb,
                            ground_truth_masks=ground_truth_masks,
                            prediction=prediction,
                        )
                        metric = binary_metric_row(target, prediction)
                        if reference_rows:
                            reference = reference_rows.get(image_id)
                            if reference is None:
                                raise RuntimeError(
                                    f"{expert}: {image_id} missing from stored {partition} metrics"
                                )
                            _assert_metric_counts(image_id, metric, reference)
                        rows.append(
                            {
                                "partition": partition,
                                "expert": EXPERT_OUTPUT_NAMES[expert],
                                "model_expert": expert,
                                "checkpoint": checkpoint_name,
                                "checkpoint_epoch": int(payload["epoch"]),
                                "image": image_id,
                                "image_folder": image_folder,
                                "tile": tile,
                                "dataset": str(batch["dataset"][index]),
                                "source_group": str(batch["source_group"][index]),
                                "ground_truth_classes": ";".join(classes_present),
                                "composite_path": relative_path.as_posix(),
                                **metric,
                            }
                        )
                        generated += 1
                    print(
                        f"{expert}/{partition}: generated {generated}/{len(partition_rows)}",
                        flush=True,
                    )
                if generated != len(partition_rows):
                    raise RuntimeError(
                        f"{expert}/{partition}: generated {generated}, expected {len(partition_rows)}"
                    )
    finally:
        del model
        torch.cuda.empty_cache()
    return rows


def _format_metric(value: Any) -> str:
    if value == "" or value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _cards(rows: Iterable[Mapping[str, Any]]) -> str:
    cards = []
    for row in rows:
        search = " ".join(
            (str(row["expert"]), str(row["partition"]), str(row["image_folder"]), str(row["tile"]))
        ).lower()
        cards.append(
            f'''<article class="card" data-expert="{html.escape(str(row["expert"]))}" data-partition="{html.escape(str(row["partition"]))}" data-search="{html.escape(search)}">
<a class="image-link" href="{html.escape(str(row["composite_path"]))}" target="_blank" rel="noopener"><img src="{html.escape(str(row["composite_path"]))}" alt="Four-panel inference result for {html.escape(str(row["tile"]))}" width="1568" height="472" loading="lazy" decoding="async"></a>
<div class="card-body"><div><span class="expert">{html.escape(str(row["expert"]))} · {html.escape(str(row["partition"]))}</span><h2>{html.escape(str(row["tile"]))}</h2><p>{html.escape(str(row["image_folder"]))}</p></div>
<dl><div><dt>F1</dt><dd>{_format_metric(row["f1"])}</dd></div><div><dt>Precision</dt><dd>{_format_metric(row["precision"])}</dd></div><div><dt>Recall</dt><dd>{_format_metric(row["recall"])}</dd></div><div><dt>IoU</dt><dd>{_format_metric(row["iou"])}</dd></div></dl></div></article>'''
        )
    return "\n".join(cards)


def _write_html(
    output_root: Path,
    rows: list[dict[str, Any]],
    report_experts: tuple[str, ...],
    partitions: tuple[str, ...],
    checkpoint_name: str,
) -> None:
    counts = {expert: sum(row["expert"] == expert for row in rows) for expert in report_experts}
    document = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM3 Multiclass Inference Report</title>
<style>:root{{--bg:#f8fafc;--surface:#fff;--ink:#1e293b;--muted:#475569;--line:#e2e8f0;--blue:#2563eb;--shadow:0 10px 28px rgba(15,23,42,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans TC",system-ui,sans-serif;line-height:1.5}}a{{color:inherit}}.top{{position:sticky;top:0;z-index:10;background:rgba(248,250,252,.96);border-bottom:1px solid var(--line)}}.top-inner,main{{width:min(1500px,calc(100% - 32px));margin:auto}}.top-inner{{padding:14px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}h1{{font-size:clamp(1.25rem,2.5vw,2rem);margin:0;margin-right:auto}}select,input{{min-height:42px;border:1px solid #cbd5e1;border-radius:8px;background:white;color:var(--ink);font:inherit;padding:8px 12px}}input{{min-width:min(340px,100%)}}.summary{{display:flex;gap:12px;flex-wrap:wrap;padding:24px 0 12px}}.stat{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px}}.stat strong{{font-size:1.25rem;margin-right:6px}}.legend{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(560px,100%),1fr));gap:18px;padding:12px 0 36px}}.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}}.card[hidden]{{display:none}}.card img{{display:block;width:100%;height:auto;background:#000}}.card-body{{display:flex;justify-content:space-between;gap:16px;padding:14px}}h2,p{{margin:0;overflow-wrap:anywhere}}h2{{font-size:1rem}}p,.expert,dt{{color:var(--muted);font-size:.85rem}}dl{{display:grid;grid-template-columns:repeat(2,auto);gap:6px 14px;margin:0}}dt,dd{{margin:0}}footer{{padding:20px;text-align:center;color:var(--muted);border-top:1px solid var(--line)}}</style></head><body>
<header class="top"><div class="top-inner"><h1>SAM3 Multiclass Inference Report</h1><label>Partition <select id="partition"><option value="all">All</option>{''.join(f'<option value="{part}">{part}</option>' for part in partitions)}</select></label><label>Expert <select id="expert"><option value="all">All</option>{''.join(f'<option value="{expert}">{expert}</option>' for expert in report_experts)}</select></label><input id="search" type="search" placeholder="Image folder or tile name"></div></header>
<main><section class="summary"><div class="stat"><strong>{len(rows)}</strong>images</div>{''.join(f'<div class="stat"><strong>{counts[expert]}</strong>{expert}</div>' for expert in report_experts)}</section><p class="legend">Original | Ground Truth | Prediction | Overlay. GT: Crack/Scratch, Loss, Craquelure/Shrinkage, Flaking/Peeling（起甲＋剝落）. Prediction is fixed blue. Threshold={THRESHOLD:.1f}.</p><section class="grid">{_cards(rows)}</section></main><footer>Generated from {checkpoint_name}; metrics remain expert-specific binary metrics.</footer>
<script>const e=document.querySelector('#expert'),p=document.querySelector('#partition'),q=document.querySelector('#search'),cards=[...document.querySelectorAll('.card')];function f(){{const s=q.value.trim().toLowerCase();for(const c of cards)c.hidden=!((e.value==='all'||c.dataset.expert===e.value)&&(p.value==='all'||c.dataset.partition===p.value)&&(!s||c.dataset.search.includes(s)))}}e.addEventListener('change',f);p.addEventListener('change',f);q.addEventListener('input',f);</script></body></html>'''
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    run_root: Path,
    rows: list[dict[str, Any]],
    report_experts: tuple[str, ...],
    partitions: tuple[str, ...],
    checkpoint_name: str,
) -> None:
    payload = {
        "schema_version": 2,
        "experiment": run_root.name,
        "threshold": THRESHOLD,
        "panel_order": list(PANEL_LABELS),
        "image_size": list(COMPOSITE_SIZE),
        "partitions": {partition: sum(row["partition"] == partition for row in rows) for partition in partitions},
        "total_images": len(rows),
        "experts": {expert: sum(row["expert"] == expert for row in rows) for expert in report_experts},
        "checkpoint": checkpoint_name,
        "checkpoint_epochs": {row["model_expert"]: int(row["checkpoint_epoch"]) for row in rows},
        "checkpoint_policy": (
            "validation-selected checkpoint; test excluded from selection"
            if checkpoint_name == "best.pt"
            else "final training checkpoint; reported separately from validation-selected best.pt"
        ),
        "ground_truth_classes": [
            {"label": label, "source_codes": list(codes), "color": list(color)}
            for label, codes, color in GROUND_TRUTH_CLASSES
        ],
        "ground_truth_overlap_rendering": "alternating class colors at overlapping pixels",
        "prediction_color": list(PREDICTION_COLOR),
        "metric_contract": "expert-specific binary target; additional GT classes are visualization-only",
        "manifest": "manifest.csv",
        "report": "index.html",
        "image_layout": "<partition>/<expert>/<source_image_name>/<tile_name>.png",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_report(
    run_root: Path,
    output: Path,
    *,
    experts: tuple[str, ...] = EXPERTS,
    partitions: tuple[str, ...] = ("test",),
    checkpoint_name: str = "best.pt",
    replace: bool = False,
) -> Path:
    run_root = run_root.resolve()
    output = output.resolve()
    if output.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    temporary = output.with_name(f".{output.name}.incomplete")
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        raise FileExistsError(f"stale report backup requires manual review: {backup}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    source_masks = _source_mask_index()
    try:
        rows = [
            row
            for expert in experts
            for row in _inference_rows(
                run_root,
                temporary,
                expert,
                checkpoint_name,
                partitions,
                source_masks,
            )
        ]
        report_experts = tuple(EXPERT_OUTPUT_NAMES[expert] for expert in experts)
        _write_manifest(temporary / "manifest.csv", rows)
        _write_summary(
            temporary / "summary.json",
            run_root,
            rows,
            report_experts,
            partitions,
            checkpoint_name,
        )
        _write_html(temporary, rows, report_experts, partitions, checkpoint_name)
        if output.exists():
            output.rename(backup)
        try:
            temporary.rename(output)
        except BaseException:
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "index.html"


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output = (args.output or run_root / "test_inference_report").resolve()
    experts = tuple(args.experts or EXPERTS)
    partitions = tuple(dict.fromkeys(args.partitions))
    print(
        build_report(
            run_root,
            output,
            experts=experts,
            partitions=partitions,
            checkpoint_name=args.checkpoint,
            replace=args.replace,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
