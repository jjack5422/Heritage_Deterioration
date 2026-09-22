"""Build static Best/Worst validation review pages for one run."""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

PANELS = (("input_path", "Input"), ("gt_path", "GT"), ("prediction_path", "Prediction"), ("overlay_path", "Overlay"))


def _score(row: dict[str, str]) -> float | None:
    try:
        return float(row["f1"])
    except (KeyError, TypeError, ValueError):
        return None


def _page(title: str, body: str) -> str:
    return f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;background:#f4f1eb;color:#24303a;margin:0}}main{{width:min(1500px,94vw);margin:auto;padding:30px 0}}nav a{{margin-right:16px}}article,.hero{{background:#fff;border:1px solid #ded8cd;border-radius:14px;padding:18px;margin:18px 0}}.panels{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}figure{{margin:0}}img{{width:100%;display:block}}figcaption{{text-align:center;padding:6px}}@media(max-width:800px){{.panels{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><nav><a href="index.html">Overview</a><a href="best_20.html">Best 20</a><a href="worst_20.html">Worst 20</a></nav>{body}</main></body></html>'''


def _cards(rows: list[dict[str, str]]) -> str:
    output = []
    for row in rows:
        panels = "".join(f'<figure><img src="{html.escape(row["report_" + key])}" alt="{label}"><figcaption>{label}</figcaption></figure>' for key, label in PANELS)
        output.append(f'<article><h2>{html.escape(row["image"])}</h2><p>F1 {html.escape(row["f1"])} · IoU {html.escape(row["iou"])} · Precision {html.escape(row["precision"])} · Recall {html.escape(row["recall"])}</p><div class="panels">{panels}</div></article>')
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run, reports = args.run_dir.resolve(), args.run_dir.resolve() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (run / "metrics" / "per_image_validation.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rankable = [row for row in rows if _score(row) is not None]
    if not rankable:
        raise RuntimeError("no rankable validation images")
    assets = reports / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "tensorboard" / "images" / "loss_curve.png", assets / "loss_curve.png")
    for index, row in enumerate(rows):
        destination = assets / "qualitative" / f"{index:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        for key, _ in PANELS:
            source, target = run / row[key], destination / f"{key.removesuffix('_path')}.png"
            shutil.copy2(source, target)
            row["report_" + key] = target.relative_to(reports).as_posix()
    ordered = sorted(rankable, key=lambda row: (_score(row), row["image"]))
    summary_file = run / "metrics" / "experiment_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    if not summary_file.is_file():
        summary = {"status": "reporting_smoke", "expert": rows[0].get("target_class", "unknown"), "selected_epoch": "unknown"}
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overview = f'<section class="hero"><h1>Training review</h1><p>Expert: {html.escape(str(summary.get("expert", "unknown")))} · selected epoch: {html.escape(str(summary.get("selected_epoch", "unknown")))}</p><img src="assets/loss_curve.png" alt="Loss curve"><p><a href="best_20.html">Best 20</a> · <a href="worst_20.html">Worst 20</a></p></section>'
    (reports / "index.html").write_text(_page("Training review", overview), encoding="utf-8")
    (reports / "best_20.html").write_text(_page("Best 20", _cards(list(reversed(ordered[-20:])))), encoding="utf-8")
    (reports / "worst_20.html").write_text(_page("Worst 20", _cards(ordered[:20])), encoding="utf-8")


if __name__ == "__main__":
    main()
