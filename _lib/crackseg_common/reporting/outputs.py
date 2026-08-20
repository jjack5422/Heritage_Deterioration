"""Structured, reproducible artifacts for one segmentation training run."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from crackseg_common.thresholding import INFERENCE_RULE


EPOCH_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "f1",
    "precision",
    "recall",
    "iou",
    "learning_rate",
)
REPORTING_SCRIPTS = Path.home() / ".codex" / "skills" / "training-output-reporting" / "scripts"
RECORDED_PACKAGES = (
    "albumentations",
    "numpy",
    "pillow",
    "segmentation-models-pytorch",
    "tensorboard",
    "torch",
    "torchvision",
)


@dataclass(frozen=True)
class RunLayout:
    """Named locations for all artifacts produced by one training run."""

    root: Path
    config: Path
    logs: Path
    tensorboard: Path
    metrics: Path
    checkpoints: Path
    qualitative: Path
    reports: Path

    @classmethod
    def create(cls, root: str | Path) -> RunLayout:
        root = Path(root)
        layout = cls(
            root=root,
            config=root / "config",
            logs=root / "logs",
            tensorboard=root / "tensorboard",
            metrics=root / "metrics",
            checkpoints=root / "artifacts" / "checkpoints",
            qualitative=root / "artifacts" / "qualitative",
            reports=root / "reports",
        )
        for directory in (
            layout.config,
            layout.logs,
            layout.tensorboard,
            layout.metrics,
            layout.checkpoints,
            layout.qualitative,
            layout.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return layout


def default_run_dir(
    *,
    project_root: str | Path,
    experiment_id: str,
    expert: str,
    outer_fold: int,
) -> Path:
    """Return the canonical fold location inside one five-fold experiment."""

    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError(f"experiment_id must be one folder name: {experiment_id!r}")
    return Path(project_root) / "runs" / experiment_id / "5fold" / expert / f"fold{outer_fold}"


def default_experiment_id(
    *, expert: str, encoder: str, seed: int, now: datetime | None = None
) -> str:
    """Create a sortable identifier for an ungrouped command-line run."""

    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H%M%S_%f")
    return f"{timestamp}_{expert}_{encoder}_seed{seed}"


def write_experiment_metadata(
    experiment_root: Path,
    experiment_id: str,
    layout: RunLayout,
    plan: Any,
) -> None:
    """Update the compact index shared by all folds in one experiment."""

    path = experiment_root / "info" / "experiment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    folds = existing.get("folds", {})
    folds.setdefault(plan.expert_name, [])
    folds[plan.expert_name] = sorted({*folds[plan.expert_name], plan.outer_fold})
    runs = set(existing.get("runs", []))
    runs.add(layout.root.relative_to(experiment_root).as_posix())
    now = datetime.now().astimezone().isoformat()
    metadata = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "layout": "5fold",
        "fold_count": 5,
        "experts": sorted(folds),
        "folds": folds,
        "runs": sorted(runs),
        "dataset_root": str(plan.root),
        "manifest_hash": plan.manifest_hash,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_run_layout(args: Any, plan: Any, *, project_root: str | Path) -> RunLayout:
    """Create a run folder and write its reproducibility records."""

    experiment_id = getattr(args, "experiment_id", None) or default_experiment_id(
        expert=plan.expert_name, encoder=args.encoder, seed=args.seed
    )
    output = Path(args.output_dir) if args.output_dir else default_run_dir(
        project_root=project_root,
        experiment_id=experiment_id,
        expert=plan.expert_name,
        outer_fold=plan.outer_fold,
    )
    layout = RunLayout.create(output)
    write_run_metadata(layout, args, plan)
    if not args.output_dir:
        write_experiment_metadata(
            Path(project_root) / "runs" / experiment_id,
            experiment_id,
            layout,
            plan,
        )
    return layout


def write_run_metadata(layout: RunLayout, args: Any, plan: Any) -> None:
    """Persist the inputs required to reproduce one expert-training run."""

    def write(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    git_revision = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            git_revision = result.stdout.strip()
    except OSError:
        pass
    write(layout.config / "args.json", vars(args))
    write(layout.config / "split_plan.json", plan.record())
    write(
        layout.config / "dataset.json",
        {
            "root": str(plan.root),
            "manifest_hash": plan.manifest_hash,
            "source_class_names": list(plan.source_class_names),
            "class_names": list(plan.class_names),
            "split": plan.record(),
        },
    )
    write(
        layout.config / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "git_revision": git_revision or None,
            "packages": package_versions(),
        },
    )
    joint = len(plan.class_names) == 3
    write(
        layout.config / "run.json",
        {
            "expert": plan.expert_name,
            "source_class_id": plan.expert_id,
            "task": plan.record().get("task"),
            "outer_fold": plan.outer_fold,
            "inner_fold": plan.inner_fold,
            "selection_metric": "val_panel_macro_loss",
            "selection_mode": "min",
            "inference_rule": "softmax_argmax" if joint else INFERENCE_RULE,
            "threshold": None if joint else 0.5,
            "threshold_source": None if joint else "uncalibrated_default",
            "threshold_policy": None,
            "command": [sys.executable, *sys.argv],
        },
    )
    (layout.logs / "train.log").write_text(
        f"run={layout.root}\nexpert={plan.expert_name}\nmanifest_hash={plan.manifest_hash}\n",
        encoding="utf-8",
    )


def package_versions() -> dict[str, str | None]:
    """Return versions for packages that materially affect a training run."""

    packages: dict[str, str | None] = {}
    for package in RECORDED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def log_message(layout: RunLayout, message: str) -> None:
    """Mirror a training message to the console and the durable run log."""

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    entry = f"[{timestamp}] {message}"
    print(entry, flush=True)
    with (layout.logs / "train.log").open("a", encoding="utf-8") as file:
        file.write(entry + "\n")


def run_reporting_script(layout: RunLayout, script_name: str, arguments: list[str]) -> None:
    """Run one bundled reporting script and retain its output in the run log."""

    script = REPORTING_SCRIPTS / script_name
    if not script.is_file():
        raise RuntimeError(f"training-output-reporting script is unavailable: {script}")
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    with (layout.logs / "train.log").open("a", encoding="utf-8") as file:
        file.write(result.stdout)
        file.write(result.stderr)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{script_name} failed: {detail}")


def export_tensorboard_scalars(layout: RunLayout) -> None:
    """Export event data to durable long-form and canonical epoch CSV files."""

    run_reporting_script(
        layout,
        "export_tensorboard_scalars.py",
        [
            "--logdir",
            str(layout.tensorboard),
            "--scalars-out",
            str(layout.metrics / "tensorboard_scalars.csv"),
            "--epochs-out",
            str(layout.metrics / "epochs.csv"),
        ],
    )


def export_tensorboard_images(layout: RunLayout) -> None:
    """Export ranked TensorBoard composites to local PNG files."""

    run_reporting_script(
        layout,
        "export_tensorboard_images.py",
        [
            "--logdir",
            str(layout.tensorboard),
            "--output-dir",
            str(layout.tensorboard / "images"),
        ],
    )


def export_tensorboard_artifacts(layout: RunLayout) -> None:
    """Export all durable, directly viewable TensorBoard artifacts."""

    export_tensorboard_scalars(layout)
    export_tensorboard_images(layout)


def build_training_dashboard(layout: RunLayout) -> None:
    """Generate the portable HTML report from durable run artifacts."""

    run_reporting_script(
        layout,
        "build_training_report.py",
        ["--run-dir", str(layout.root)],
    )


def create_tensorboard_writer(log_dir: Path) -> Any:
    """Create the optional-at-import TensorBoard writer required for training."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard is required for U-Net training reports. "
            "Install it in crackseg_env with: crackseg_env/bin/python -m pip install tensorboard"
        ) from error
    return SummaryWriter(log_dir=str(log_dir))


def finite_or_blank(value: Any) -> float | str:
    if value is None:
        return ""
    number = float(value)
    return number if math.isfinite(number) else ""


class EpochReporter:
    """Keep TensorBoard and the canonical per-epoch CSV in lockstep."""

    def __init__(self, epoch_csv: Path, writer: Any) -> None:
        self.epoch_csv = epoch_csv
        self.writer = writer
        self.epoch_csv.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.epoch_csv.open("a", encoding="utf-8", newline="")
        self.csv = csv.DictWriter(self.file, fieldnames=EPOCH_COLUMNS)
        if self.epoch_csv.stat().st_size == 0:
            self.csv.writeheader()
            self.file.flush()

    def record(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        validation: dict[str, Any],
        learning_rates: dict[str, float],
    ) -> dict[str, float | str | int]:
        micro = validation["tile_micro"]
        row: dict[str, float | str | int] = {
            "epoch": epoch,
            "train_loss": finite_or_blank(train_metrics["loss"]),
            "val_loss": finite_or_blank(validation["panel_macro_loss"]),
            "f1": finite_or_blank(micro["mf1"]),
            "precision": finite_or_blank(micro["mprecision"]),
            "recall": finite_or_blank(micro["mrecall"]),
            "iou": finite_or_blank(micro["miou"]),
            "learning_rate": finite_or_blank(learning_rates["decoder"]),
        }
        scalar_tags = {
            "loss/train": row["train_loss"],
            "loss/validation": row["val_loss"],
            "metrics/f1": row["f1"],
            "metrics/precision": row["precision"],
            "metrics/recall": row["recall"],
            "metrics/iou": row["iou"],
            "optimizer/lr": row["learning_rate"],
        }
        for tag, value in scalar_tags.items():
            if value != "":
                self.writer.add_scalar(tag, value, epoch)
        if "cost" in train_metrics:
            self.writer.add_scalar("loss/train_cost", train_metrics["cost"], epoch)
        if "loss_components" in validation:
            self.writer.add_scalar(
                "loss/validation_cost", validation["loss_components"]["cost"], epoch
            )
        if "cross_confusion" in validation:
            self.writer.add_scalar(
                "metrics/craquelure_to_crack_rate",
                validation["cross_confusion"]["craquelure_to_crack"]["rate"],
                epoch,
            )
            for name in ("crack", "craquelure"):
                self.writer.add_scalar(
                    f"metrics/{name}_f1",
                    micro["per_class"][name]["f1"],
                    epoch,
                )
        self.csv.writerow(row)
        self.file.flush()
        self.writer.flush()
        return row

    def close(self) -> None:
        self.file.close()
        self.writer.close()
