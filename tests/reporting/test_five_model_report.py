from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "reporting" / "build_five_model_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_merged_five_model_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_image_f1_uses_equal_and_foreground_fraction_weights() -> None:
    report = _load_module()
    rows = [
        {"image": "a.png", "f1": "0.2", "gt_pixels": "10"},
        {"image": "b.png", "f1": "0.8", "gt_pixels": "40"},
        {"image": "empty.png", "f1": "", "gt_pixels": "0"},
    ]

    result = report.aggregate_image_f1(
        rows,
        valid_pixels={"a.png": 100, "b.png": 200, "empty.png": 100},
    )

    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["weighted_f1"] == pytest.approx((0.1 * 0.2 + 0.2 * 0.8) / 0.3)
    assert result["positive_images"] == 2


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_report_is_self_contained_and_keeps_25_loss_images_separate(tmp_path: Path) -> None:
    report = _load_module()
    repo = tmp_path / "repo"
    data_root = repo / "dataset"
    (data_root / "masks").mkdir(parents=True)

    # The loader accepts an injected valid-pixel map, so this fixture does not need real PNG masks.
    specs = []
    for model_index in range(5):
        run = repo / f"model-{model_index}" / "runs" / "exp"
        specs.append(
            report.ModelSpec(
                key=f"model_{model_index}",
                name=f"Model {model_index}",
                family="Test",
                run_dir=run,
                color="#123456",
            )
        )
        (run / "info").mkdir(parents=True)
        (run / "info" / "experiment.json").write_text(
            json.dumps({"dataset_root": str(data_root), "threshold": 0.5}),
            encoding="utf-8",
        )
        (run / "info" / "oof_summary.json").write_text(
            json.dumps(
                {
                    "outer_tile_count": 10,
                    "tile_micro": {
                        "tp": 80,
                        "fp": 20,
                        "fn": 20,
                        "mprecision": 0.8,
                        "mrecall": 0.8,
                        "mf1": 0.8,
                        "miou": 2 / 3,
                    },
                }
            ),
            encoding="utf-8",
        )
        for fold in range(5):
            fold_dir = run / "5fold" / "craquelure" / f"fold{fold}"
            metrics_dir = fold_dir / "metrics"
            images_dir = fold_dir / "tensorboard" / "images"
            metrics_dir.mkdir(parents=True)
            images_dir.mkdir(parents=True)
            (metrics_dir / "outer_test_metrics.json").write_text(
                json.dumps(
                    {
                        "loss": 0.1 + fold / 100,
                        "valid_pixels": 200,
                        "tile_micro": {
                            "tp": 16,
                            "fp": 4,
                            "fn": 4,
                            "mprecision": 0.8,
                            "mrecall": 0.8,
                            "mf1": 0.8,
                            "miou": 2 / 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            image = f"image-{fold}.png"
            _write_csv(
                metrics_dir / "per_image_outer_test.csv",
                ["image", "f1", "gt_pixels"],
                [{"image": image, "f1": 0.8, "gt_pixels": 10}],
            )
            _write_csv(
                metrics_dir / "epochs.csv",
                ["epoch", "train_loss", "val_loss"],
                [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}],
            )
            (images_dir / "loss_curve.png").write_bytes(b"png")

    output_dir = repo / "model_comparisons" / "report"
    index = report.build_report(
        repo_root=repo,
        output_dir=output_dir,
        model_specs=specs,
        valid_pixels={f"image-{fold}.png": 100 for fold in range(5)},
    )

    html = index.read_text(encoding="utf-8")
    assert html.count('class="loss-figure"') == 25
    assert html.count('src="data:image/png;base64,') == 25
    assert html.count('href="data:image/png;base64,') == 25
    assert 'loading="lazy"' not in html
    assert "https://" not in html
    assert "Precision" in html
    assert "Recall" in html
    assert "IoU" in html
    assert "Micro-F1" in html
    assert "Macro-F1" in html
    assert "Weighted-F1" in html
    assert "outer test 未參與 checkpoint 或 threshold 選擇" in html
    assert len(list((output_dir / "assets" / "loss").glob("*/*.png"))) == 25
    assert (output_dir / "data" / "model_summary.csv").is_file()
    assert (output_dir / "data" / "fold_metrics.csv").is_file()
    assert (output_dir / "data" / "report_data.json").is_file()
