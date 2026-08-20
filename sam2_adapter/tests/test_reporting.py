"""Smoke test the required TensorBoard, PNG, CSV, and HTML reporting contract."""

from __future__ import annotations

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from sam2_adapter.reporting import (
    EpochReporter,
    RunLayout,
    finalize_reporting,
    save_qualitative_example,
)


def test_reporting_exports_required_static_artifacts(tmp_path) -> None:
    layout = RunLayout.create(tmp_path / "fold0")
    writer = SummaryWriter(str(layout.tensorboard))
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    reporter.record(
        1,
        train_loss=0.4,
        validation_loss=0.3,
        f1=0.5,
        precision=0.6,
        recall=0.4,
        iou=0.33,
        learning_rate=5e-4,
    )
    row = save_qualitative_example(
        layout,
        image_id="tile.png",
        input_rgb=np.full((8, 8, 3), 100, dtype=np.uint8),
        target=np.pad(np.ones((2, 2), dtype=np.uint8), ((3, 3), (3, 3))),
        prediction=np.pad(np.ones((2, 2), dtype=np.uint8), ((3, 3), (3, 3))),
        target_class="damage_union",
    )
    row.update({"image": "tile.png", "split": "validation", "target_class": "damage_union"})
    finalize_reporting(
        layout,
        writer=writer,
        reporter=reporter,
        validation_rows=[row],
        selected_epoch=1,
        outer_test_metrics={"scope": "outer_test_not_used_for_selection", "tile_micro": {"f1": 0.5}},
    )

    assert (layout.tensorboard / "images" / "loss_curve.png").is_file()
    assert (layout.tensorboard / "images" / "manifest.csv").is_file()
    assert (layout.tensorboard / "images" / "best").is_dir()
    assert (layout.tensorboard / "images" / "worst").is_dir()
    assert (layout.reports / "index.html").is_file()
    assert (layout.reports / "best_20.html").is_file()
    assert (layout.reports / "worst_20.html").is_file()
    assert (layout.metrics / "experiment_summary.json").is_file()
