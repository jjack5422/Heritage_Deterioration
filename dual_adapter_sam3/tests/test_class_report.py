import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

from dual_adapter_sam3.class_report import build_class_report, rank_validation_rows


def _row(
    image: str,
    target_class: str,
    *,
    f1: float,
    gt_pixels: int,
    pred_pixels: int,
) -> dict[str, str | float | int]:
    identifier = Path(image).stem
    base = Path("artifacts") / "qualitative" / target_class / identifier
    return {
        "image": f"{target_class}/{image}",
        "target_class": target_class,
        "f1": f1,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "fp": pred_pixels if gt_pixels == 0 else 0,
        "input_path": (base / "input.png").as_posix(),
        "gt_path": (base / "gt.png").as_posix(),
        "prediction_path": (base / "prediction.png").as_posix(),
        "overlay_path": (base / "overlay.png").as_posix(),
    }


def test_rankings_are_split_by_class_and_exclude_empty_ground_truth() -> None:
    rows = [
        _row("loss_best.png", "loss", f1=0.9, gt_pixels=10, pred_pixels=10),
        _row("loss_worst.png", "loss", f1=0.2, gt_pixels=10, pred_pixels=8),
        _row("craq_best.png", "crack_craquelure", f1=0.8, gt_pixels=8, pred_pixels=7),
        _row("craq_worst.png", "crack_craquelure", f1=0.1, gt_pixels=8, pred_pixels=5),
        _row("correct_negative.png", "loss", f1=1.0, gt_pixels=0, pred_pixels=0),
        _row("small_fp.png", "loss", f1=0.0, gt_pixels=0, pred_pixels=3),
        _row("large_fp.png", "crack_craquelure", f1=0.0, gt_pixels=0, pred_pixels=20),
    ]

    groups = rank_validation_rows(rows)

    assert [row["image"] for row in groups["loss_best"]] == ["loss/loss_best.png", "loss/loss_worst.png"]
    assert [row["image"] for row in groups["loss_worst"]] == ["loss/loss_worst.png", "loss/loss_best.png"]
    assert [row["image"] for row in groups["craquelure_best"]] == [
        "crack_craquelure/craq_best.png",
        "crack_craquelure/craq_worst.png",
    ]
    assert [row["image"] for row in groups["craquelure_worst"]] == [
        "crack_craquelure/craq_worst.png",
        "crack_craquelure/craq_best.png",
    ]
    assert [row["image"] for row in groups["false_positives"]] == [
        "crack_craquelure/large_fp.png",
        "loss/small_fp.png",
    ]
    assert all("correct_negative" not in str(row["image"]) for group in groups.values() for row in group)


def test_builder_creates_class_pages_with_working_image_links(tmp_path: Path) -> None:
    run = tmp_path / "fold0"
    rows = [
        _row("loss.png", "loss", f1=0.75, gt_pixels=12, pred_pixels=10),
        _row("craq.png", "crack_craquelure", f1=0.5, gt_pixels=9, pred_pixels=7),
        _row("fp.png", "loss", f1=0.0, gt_pixels=0, pred_pixels=5),
    ]
    for row in rows:
        for column in ("input_path", "gt_path", "prediction_path", "overlay_path"):
            destination = run / str(row[column])
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(destination)
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    with (metrics / "per_image_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve = run / "tensorboard" / "images" / "loss_curve.png"
    curve.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(curve)

    build_class_report(run)

    expected = {
        "index.html",
        "best_20.html",
        "worst_20.html",
        "loss_best_20.html",
        "loss_worst_20.html",
        "craquelure_best_20.html",
        "craquelure_worst_20.html",
        "false_positives.html",
    }
    reports = run / "reports"
    assert expected <= {path.name for path in reports.glob("*.html")}
    assert "Correct negatives are excluded" in (reports / "index.html").read_text(encoding="utf-8")
    assert "Target: loss" in (reports / "false_positives.html").read_text(encoding="utf-8")
    tensorboard_images = run / "tensorboard" / "images"
    expected_image_groups = {
        "loss/best",
        "loss/worst",
        "craquelure/best",
        "craquelure/worst",
        "false_positives",
    }
    assert expected_image_groups <= {
        path.relative_to(tensorboard_images).as_posix()
        for path in tensorboard_images.rglob("*")
        if path.is_dir()
    }
    manifest = tensorboard_images / "class_manifest.csv"
    assert manifest.is_file()
    manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert {row["group"] for row in manifest_rows} == {
        "loss_best",
        "loss_worst",
        "craquelure_best",
        "craquelure_worst",
        "false_positives",
    }
    assert all((tensorboard_images / row["path"]).is_file() for row in manifest_rows)
    for report in reports.glob("*.html"):
        contents = report.read_text(encoding="utf-8")
        for relative in re.findall(r'src="([^"]+)"', contents):
            assert (reports / relative).is_file(), (report.name, relative)
