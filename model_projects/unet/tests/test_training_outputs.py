from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = SRC.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC))

from crackseg_common.reporting.outputs import (  # noqa: E402
    EpochReporter,
    RunLayout,
    create_run_layout,
    default_run_dir,
    log_message,
    write_run_metadata,
)
from crackseg_common.reporting.qualitative import (  # noqa: E402
    _multiclass_score,
    ranked_validation_rows,
    score_image,
    write_checkpoint_comparison,
    write_multiclass_validation_artifacts,
    write_validation_artifacts,
)


class RecordingWriter:
    def __init__(self) -> None:
        self.events: list[tuple[str, float, int]] = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.events.append((tag, value, step))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


def validation_metrics() -> dict:
    return {
        "panel_macro_loss": 0.4,
        "expert_panel_macro": {"iou": 0.5},
        "tile_micro": {"mf1": 0.6, "mprecision": 0.7, "mrecall": 0.8, "miou": 0.5},
    }


def test_run_layout_contains_only_named_training_artifact_folders(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "run")

    assert layout.config.is_dir()
    assert layout.logs.is_dir()
    assert layout.tensorboard.is_dir()
    assert layout.metrics.is_dir()
    assert layout.checkpoints.is_dir()
    assert layout.qualitative.is_dir()
    assert layout.reports.is_dir()


def test_default_run_dir_groups_one_experiment_by_cv_expert_and_fold() -> None:
    path = default_run_dir(
        project_root=PROJECT_ROOT,
        experiment_id="crack_craquelure_5fold_20260813",
        expert="crack",
        outer_fold=0,
    )

    assert path == (
        PROJECT_ROOT
        / "runs/crack_craquelure_5fold_20260813/5fold/crack/fold0"
    )


class MetadataPlan:
    root = Path("/data/monument")
    manifest_hash = "manifest-sha256"
    source_class_names = ("background", "crack")
    class_names = ("background", "crack")
    expert_name = "crack"
    expert_id = 1
    outer_fold = 0
    inner_fold = 1

    def record(self) -> dict:
        return {"manifest_hash": self.manifest_hash, "outer_fold": self.outer_fold}


def test_default_layout_writes_shared_experiment_info(tmp_path: Path) -> None:
    args = Namespace(
        output_dir=None,
        experiment_id="crack_craquelure_5fold_20260813",
        encoder="resnet50",
        seed=42,
    )

    layout = create_run_layout(args, MetadataPlan(), project_root=tmp_path)

    assert layout.root == (
        tmp_path / "runs/crack_craquelure_5fold_20260813/5fold/crack/fold0"
    )
    experiment = (tmp_path / "runs/crack_craquelure_5fold_20260813/info/experiment.json")
    assert experiment.is_file()
    assert '"experts": [\n    "crack"\n  ]' in experiment.read_text(encoding="utf-8")


def test_run_metadata_records_args_dataset_environment_and_checkpoint_selection(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "run")
    args = Namespace(dataset_root="/data/monument", encoder="resnet50", seed=42)

    write_run_metadata(layout, args, MetadataPlan())

    assert (layout.config / "args.json").is_file()
    dataset = (layout.config / "dataset.json").read_text(encoding="utf-8")
    run = (layout.config / "run.json").read_text(encoding="utf-8")
    environment = (layout.config / "environment.json").read_text(encoding="utf-8")
    assert "manifest-sha256" in dataset
    assert '"selection_metric": "val_panel_macro_loss"' in run
    assert '"torch"' in environment
    assert '"packages"' in environment
    assert '"inference_rule": "foreground_probability > threshold"' in run
    assert '"threshold": 0.5' in run


def test_joint_run_metadata_records_argmax_task_without_binary_threshold(tmp_path: Path) -> None:
    class JointMetadataPlan(MetadataPlan):
        class_names = ("background", "crack", "craquelure")
        expert_name = "crack_craquelure"
        expert_id = -1

        def record(self) -> dict:
            return {
                "task": {"name": self.expert_name, "source_class_ids": [1, 4]},
                "manifest_hash": self.manifest_hash,
            }

    layout = RunLayout.create(tmp_path / "run")
    args = Namespace(dataset_root="/data/monument", encoder="resnet50", seed=42)

    write_run_metadata(layout, args, JointMetadataPlan())

    run = json.loads((layout.config / "run.json").read_text())
    assert run["task"]["name"] == "crack_craquelure"
    assert run["inference_rule"] == "softmax_argmax"
    assert run["threshold"] is None


def test_log_message_adds_timezone_timestamp(tmp_path: Path, capsys) -> None:
    layout = RunLayout.create(tmp_path / "run")

    log_message(layout, "epoch=1")

    line = (layout.logs / "train.log").read_text(encoding="utf-8").strip()
    assert line.startswith("[")
    assert "+" in line.split("]", 1)[0]
    assert line.endswith("epoch=1")
    assert "epoch=1" in capsys.readouterr().out


def test_epoch_reporter_writes_canonical_table_and_tensorboard_tags(tmp_path: Path) -> None:
    writer = RecordingWriter()
    reporter = EpochReporter(tmp_path / "metrics" / "epochs.csv", writer)

    reporter.record(
        epoch=3,
        train_metrics={"loss": 0.3},
        validation=validation_metrics(),
        learning_rates={"decoder": 0.001, "encoder": 0.0001},
    )
    reporter.close()

    with (tmp_path / "metrics" / "epochs.csv").open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row == {
        "epoch": "3",
        "train_loss": "0.3",
        "val_loss": "0.4",
        "f1": "0.6",
        "precision": "0.7",
        "recall": "0.8",
        "iou": "0.5",
        "learning_rate": "0.001",
    }
    assert {tag for tag, _, _ in writer.events} == {
        "loss/train",
        "loss/validation",
        "metrics/f1",
        "metrics/precision",
        "metrics/recall",
        "metrics/iou",
        "optimizer/lr",
    }
    assert writer.flushed and writer.closed


def test_epoch_reporter_skips_null_metric_tensorboard_event_but_keeps_empty_csv_cell(tmp_path: Path) -> None:
    writer = RecordingWriter()
    reporter = EpochReporter(tmp_path / "epochs.csv", writer)
    metrics = validation_metrics()
    metrics["tile_micro"]["mf1"] = None

    reporter.record(1, {"loss": 0.9}, metrics, {"decoder": 0.002})

    with (tmp_path / "epochs.csv").open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["f1"] == ""
    assert "metrics/f1" not in {tag for tag, _, _ in writer.events}


def test_epoch_reporter_adds_joint_cost_and_cross_confusion_tensorboard_tags(
    tmp_path: Path,
) -> None:
    writer = RecordingWriter()
    reporter = EpochReporter(tmp_path / "epochs.csv", writer)
    metrics = validation_metrics()
    metrics["loss_components"] = {"cost": 0.12}
    metrics["cross_confusion"] = {"craquelure_to_crack": {"rate": 0.08}}
    metrics["tile_micro"]["per_class"] = {
        "crack": {"f1": 0.55},
        "craquelure": {"f1": 0.72},
    }

    reporter.record(1, {"loss": 0.9, "cost": 0.2}, metrics, {"decoder": 0.002})

    tags = {tag for tag, _, _ in writer.events}
    assert {
        "loss/train_cost",
        "loss/validation_cost",
        "metrics/crack_f1",
        "metrics/craquelure_f1",
        "metrics/craquelure_to_crack_rate",
    } <= tags


class ValidationTiles(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        mask = torch.zeros(8, 8, dtype=torch.long)
        mask[:, 2:4] = 1
        return {
            "image": torch.zeros(3, 8, 8),
            "mask": mask,
            "name": f"panel{index}_R1_C01__y00000_x00000.png",
        }


class ExactValidationModel(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((image.shape[0], 2, *image.shape[-2:]), device=image.device)
        logits[:, 1, :, 2:4] = 8
        return logits


def test_validation_artifacts_write_metrics_and_four_images_for_every_tile(tmp_path: Path) -> None:
    rows = write_validation_artifacts(
        model=ExactValidationModel(),
        loader=torch.utils.data.DataLoader(ValidationTiles(), batch_size=2),
        device=torch.device("cpu"),
        output_root=tmp_path,
        expert_name="crack",
        ignore_value=255,
    )

    with (tmp_path / "metrics" / "per_image_validation.csv").open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert len(rows) == len(saved) == 2
    assert {row["f1"] for row in saved} == {"1.0"}
    for row in saved:
        assert row["split"] == "validation"
        assert row["target_class"] == "crack"
        for column in ("input_path", "gt_path", "prediction_path", "overlay_path"):
            assert (tmp_path / row[column]).is_file()


def test_no_foreground_image_has_blank_f1_and_explicit_false_positive_rate() -> None:
    target = np.zeros((2, 2), dtype=np.uint8)
    prediction = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    metrics = score_image(prediction, target, ignore_value=255)

    assert metrics["f1"] == ""
    assert metrics["gt_pixels"] == 0
    assert metrics["pred_pixels"] == 1
    assert metrics["valid_pixels"] == 4
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["error_reason"] == "false_positive"


def test_multiclass_no_foreground_image_reports_false_positive() -> None:
    target = np.zeros((2, 2), dtype=np.uint8)
    prediction = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    metrics = _multiclass_score(prediction, target, ignore_value=255)

    assert metrics["f1"] == ""
    assert metrics["gt_pixels"] == 0
    assert metrics["pred_pixels"] == 1
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["error_reason"] == "false_positive"


def test_multiclass_missing_all_foreground_reports_false_negative() -> None:
    target = np.array([[1, 0], [0, 2]], dtype=np.uint8)
    prediction = np.zeros((2, 2), dtype=np.uint8)

    metrics = _multiclass_score(prediction, target, ignore_value=255)

    assert metrics["f1"] == 0.0
    assert metrics["gt_pixels"] == 2
    assert metrics["pred_pixels"] == 0
    assert metrics["error_reason"] == "false_negative"


def test_validation_ranking_accepts_numeric_csv_strings_and_ignores_blank_f1() -> None:
    rows = [
        {"image": "middle", "f1": "0.5"},
        {"image": "best", "f1": "0.9"},
        {"image": "background-only", "f1": ""},
        {"image": "worst", "f1": "0.1"},
    ]

    best, worst = ranked_validation_rows(rows, top_k=2)

    assert [row["image"] for row in best] == ["best", "middle"]
    assert [row["image"] for row in worst] == ["worst", "middle"]


class JointValidationTiles(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        mask = torch.zeros(8, 8, dtype=torch.long)
        mask[:, 1:3] = 1
        mask[:, 5:7] = 2
        image = torch.zeros(3, 8, 8)
        image[0] = mask == 1
        image[1] = mask == 2
        return {
            "image": image,
            "mask": mask,
            "name": f"joint{index}_R1_C01__y00000_x00000.png",
        }


class ExactJointValidationModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        background = ((images[:, 0] + images[:, 1]) == 0).float() * 8
        return torch.stack((background, images[:, 0] * 8, images[:, 1] * 8), dim=1)


def test_multiclass_validation_artifacts_report_both_classes_and_cross_confusion(
    tmp_path: Path,
) -> None:
    rows = write_multiclass_validation_artifacts(
        model=ExactJointValidationModel(),
        loader=torch.utils.data.DataLoader(JointValidationTiles(), batch_size=2),
        device=torch.device("cpu"),
        output_root=tmp_path,
        ignore_value=255,
    )

    assert len(rows) == 2
    assert {row["f1"] for row in rows} == {1.0}
    assert {row["crack_f1"] for row in rows} == {1.0}
    assert {row["craquelure_f1"] for row in rows} == {1.0}
    assert {row["craquelure_to_crack"] for row in rows} == {0}
    assert len(list((tmp_path / "artifacts" / "qualitative").glob("*/overlay.png"))) == 2


def test_checkpoint_comparison_records_best_last_and_directional_delta(tmp_path: Path) -> None:
    best = validation_metrics()
    best["tile_micro"]["per_class"] = {"crack": {}, "craquelure": {}}
    best["tile_micro"]["confusion_matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    best["cross_confusion"] = {"craquelure_to_crack": {"count": 8, "rate": 0.08}}
    last = validation_metrics()
    last["tile_micro"]["per_class"] = {"crack": {}, "craquelure": {}}
    last["tile_micro"]["confusion_matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    last["panel_macro_loss"] = 0.45
    last["cross_confusion"] = {"craquelure_to_crack": {"count": 5, "rate": 0.05}}

    result = write_checkpoint_comparison(
        tmp_path / "checkpoint_comparison.json",
        best_epoch=23,
        best_metrics=best,
        last_epoch=80,
        last_metrics=last,
    )

    assert result["best"]["epoch"] == 23
    assert result["last"]["epoch"] == 80
    assert result["delta_last_minus_best"]["panel_macro_loss"] == pytest.approx(0.05)
    assert result["delta_last_minus_best"]["craquelure_to_crack_rate"] == pytest.approx(-0.03)
    assert (tmp_path / "checkpoint_comparison.csv").is_file()
