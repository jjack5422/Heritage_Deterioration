from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reporting.backfill import (  # noqa: E402
    canonical_backfill_dir,
    discover_legacy_runs,
    parser,
    replay_history,
)
from crackseg_common.reporting.outputs import RunLayout  # noqa: E402


def test_discover_legacy_runs_returns_experts_and_folds_in_stable_order(tmp_path: Path) -> None:
    for relative in ("craquelure/fold1", "crack/fold2", "crack/fold0"):
        run = tmp_path / relative
        run.mkdir(parents=True)
        for name in (
            "best.pt",
            "last.pt",
            "history.json",
            "outer_test_metrics.json",
            "split_plan.json",
            "train.log",
        ):
            (run / name).touch()

    assert [path.relative_to(tmp_path).as_posix() for path in discover_legacy_runs(tmp_path)] == [
        "crack/fold0",
        "crack/fold2",
        "craquelure/fold1",
    ]


def test_canonical_backfill_dir_uses_original_run_identity(tmp_path: Path) -> None:
    destination = canonical_backfill_dir(
        output_root=tmp_path,
        experiment_id="crack_craquelure_5fold_20260813",
        expert="crack",
        outer_fold=3,
    )

    assert destination == (
        tmp_path / "crack_craquelure_5fold_20260813/5fold/crack/fold3"
    )


def test_backfill_defaults_to_the_unet_project_runs_folder(tmp_path: Path) -> None:
    options = parser().parse_args(["--legacy-root", str(tmp_path)])

    assert options.output_root == PROJECT_ROOT / "runs"
    assert options.experiment_id is None


def test_replay_history_writes_canonical_csv_and_all_tensorboard_tags(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "train": {"loss": 0.8},
            "val": {
                "panel_macro_loss": 0.7,
                "tile_micro": {
                    "mf1": 0.2,
                    "mprecision": 0.3,
                    "mrecall": 0.4,
                    "miou": 0.1,
                },
            },
            "lr": {"decoder": 0.0003, "encoder": 0.00003},
        },
        {
            "epoch": 2,
            "train": {"loss": 0.6},
            "val": {
                "panel_macro_loss": 0.5,
                "tile_micro": {
                    "mf1": 0.4,
                    "mprecision": 0.5,
                    "mrecall": 0.6,
                    "miou": 0.25,
                },
            },
            "lr": {"decoder": 0.00015, "encoder": 0.000015},
        },
    ]
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    layout = RunLayout.create(tmp_path / "run")

    replay_history(history_path, layout)

    with (layout.metrics / "epochs.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[-1]["val_loss"] == "0.5"
    accumulator = EventAccumulator(str(layout.tensorboard))
    accumulator.Reload()
    expected_tags = {
        "loss/train",
        "loss/validation",
        "metrics/f1",
        "metrics/precision",
        "metrics/recall",
        "metrics/iou",
        "optimizer/lr",
    }
    assert set(accumulator.Tags()["scalars"]) == expected_tags
    assert {event.step for event in accumulator.Scalars("loss/validation")} == {1, 2}
