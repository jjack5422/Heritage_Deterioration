"""Backfill canonical reports for legacy crack/craquelure training runs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from crackseg_common.data_plan import DataError, DataPlan, prepare_dataset  # noqa: E402
from crackseg_common.reporting.outputs import (  # noqa: E402
    EpochReporter,
    RunLayout,
    build_training_dashboard,
    create_tensorboard_writer,
    export_tensorboard_images,
    export_tensorboard_scalars,
    write_experiment_metadata,
    write_run_metadata,
)
from crackseg_common.reporting.qualitative import (  # noqa: E402
    write_tensorboard_validation_images,
    write_validation_artifacts,
)
from crackseg_common.training_runtime import make_loader  # noqa: E402
from unet_model import build_resunet  # noqa: E402


LEGACY_FILES = (
    "best.pt",
    "last.pt",
    "history.json",
    "outer_test_metrics.json",
    "split_plan.json",
    "train.log",
)


def read_json(path: Path) -> Any:
    """Read JSON while retaining the source path in any parse error."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def discover_legacy_runs(legacy_root: Path) -> list[Path]:
    """Return complete expert/fold legacy directories in stable order."""

    discovered = []
    for candidate in legacy_root.glob("*/fold*"):
        if candidate.is_dir() and all((candidate / name).is_file() for name in LEGACY_FILES):
            discovered.append(candidate)
    return sorted(discovered, key=lambda path: path.relative_to(legacy_root).as_posix())


def canonical_backfill_dir(
    *,
    output_root: Path,
    experiment_id: str,
    expert: str,
    outer_fold: int,
) -> Path:
    """Build the canonical experiment/five-fold destination."""

    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError(f"experiment_id must be one folder name: {experiment_id!r}")
    return output_root / experiment_id / "5fold" / expert / f"fold{outer_fold}"


def validated_history(history_path: Path) -> list[dict[str, Any]]:
    history = read_json(history_path)
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"history must be a non-empty array: {history_path}")
    expected_epochs = list(range(1, len(history) + 1))
    actual_epochs = [item.get("epoch") if isinstance(item, dict) else None for item in history]
    if actual_epochs != expected_epochs:
        raise RuntimeError(
            f"history epochs are not sequential in {history_path}: {actual_epochs[:5]}"
        )
    return history


def replay_history(history_path: Path, layout: RunLayout) -> None:
    """Replay legacy epoch history into TensorBoard and the canonical tables."""

    if any(layout.tensorboard.iterdir()):
        raise RuntimeError(f"TensorBoard destination is not empty: {layout.tensorboard}")
    if (layout.metrics / "epochs.csv").exists():
        raise RuntimeError(f"epoch CSV already exists: {layout.metrics / 'epochs.csv'}")
    history = validated_history(history_path)
    shutil.copy2(history_path, layout.metrics / "history.json")
    reporter = EpochReporter(
        layout.metrics / "epochs.csv",
        create_tensorboard_writer(layout.tensorboard),
    )
    try:
        for item in history:
            reporter.record(
                int(item["epoch"]),
                item["train"],
                item["val"],
                item["lr"],
            )
    finally:
        reporter.close()
    export_tensorboard_scalars(layout)


def infer_started_at(legacy_root: Path) -> datetime:
    log_path = legacy_root / "TRAINING_LOG.md"
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    match = re.search(r"^- Started:\s*(\S+)", text, flags=re.MULTILINE)
    if match:
        return datetime.fromisoformat(match.group(1))
    return datetime.fromtimestamp(legacy_root.stat().st_mtime).astimezone()


def infer_training_revision(legacy_root: Path) -> str | None:
    log_path = legacy_root / "TRAINING_LOG.md"
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    match = re.search(r"^- Commit:\s*`([0-9a-f]{7,40})`", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def resolve_dataset_root(checkpoint_args: Namespace, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    recorded = Path(checkpoint_args.dataset_root)
    return recorded.resolve() if recorded.is_absolute() else (REPOSITORY_ROOT / recorded).resolve()


def validate_legacy_identity(
    legacy_run: Path,
    checkpoint: dict[str, Any],
    plan: DataPlan,
) -> None:
    """Reject a backfill if its checkpoint, stored split and dataset disagree."""

    expected_expert = {"name": plan.expert_name, "source_class_id": plan.expert_id}
    checks = {
        "checkpoint expert": checkpoint.get("expert") == expected_expert,
        "checkpoint manifest": checkpoint.get("manifest_hash") == plan.manifest_hash,
        "checkpoint outer fold": checkpoint.get("outer_fold") == plan.outer_fold,
        "checkpoint inner fold": checkpoint.get("inner_fold") == plan.inner_fold,
        "checkpoint selection metric": checkpoint.get("selection_metric")
        == "val_panel_macro_loss",
    }
    stored_split = read_json(legacy_run / "split_plan.json")
    checks.update(
        {
            "stored manifest": stored_split.get("manifest_hash") == plan.manifest_hash,
            "stored outer fold": stored_split.get("outer_fold") == plan.outer_fold,
            "stored inner fold": stored_split.get("inner_fold") == plan.inner_fold,
            "stored validation tiles": tuple(stored_split.get("val", ())) == plan.val,
        }
    )
    failed = [name for name, matches in checks.items() if not matches]
    if failed:
        raise DataError(f"legacy run identity mismatch ({', '.join(failed)}): {legacy_run}")


def reconstructed_training_command(args: Namespace) -> list[str]:
    """Record the legacy CLI values while making reconstruction explicit."""

    command = [
        "crackseg_env/bin/python",
        "unet/src/train.py",
    ]
    for name, value in vars(args).items():
        if name == "class_names" or value is None or value is False:
            continue
        flag = f"--{name.replace('_', '-')}"
        if value is True:
            command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def record_backfill_metadata(
    *,
    layout: RunLayout,
    legacy_run: Path,
    args: Namespace,
    plan: DataPlan,
    training_revision: str | None,
    started_at: datetime,
) -> None:
    write_run_metadata(layout, args, plan)
    run_path = layout.config / "run.json"
    run = read_json(run_path)
    run.update(
        {
            "command": reconstructed_training_command(args),
            "command_reconstructed": True,
            "original_started_at": started_at.isoformat(),
            "original_training_git_revision": training_revision,
        }
    )
    write_json(run_path, run)
    environment_path = layout.config / "environment.json"
    environment = read_json(environment_path)
    environment.update(
        {
            "metadata_capture": "backfill",
            "original_training_git_revision": training_revision,
            "original_package_versions_available": False,
        }
    )
    write_json(environment_path, environment)
    write_json(
        layout.config / "backfill.json",
        {
            "status": "in_progress",
            "legacy_run": str(legacy_run.resolve()),
            "tensorboard_source": "replayed from legacy history.json",
            "validation_source": "recomputed from validation split with best.pt",
            "created_at": datetime.now().astimezone().isoformat(),
        },
    )


def copy_legacy_artifacts(legacy_run: Path, layout: RunLayout) -> None:
    """Copy rather than move raw legacy artifacts into the canonical run."""

    shutil.copy2(legacy_run / "best.pt", layout.checkpoints / "best.pt")
    shutil.copy2(legacy_run / "last.pt", layout.checkpoints / "last.pt")
    shutil.copy2(
        legacy_run / "outer_test_metrics.json",
        layout.metrics / "outer_test_metrics.json",
    )
    shutil.copy2(legacy_run / "train.log", layout.logs / "train.log")


def append_log(layout: RunLayout, message: str) -> None:
    with (layout.logs / "train.log").open("a", encoding="utf-8") as file:
        file.write(f"\n[report-backfill] {message}\n")


def backfill_run(
    *,
    legacy_run: Path,
    output_root: Path,
    experiment_id: str,
    dataset_root: Path | None,
    started_at: datetime,
    training_revision: str | None,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Path:
    """Create one self-contained report without rerunning model training."""

    checkpoint = torch.load(legacy_run / "best.pt", map_location="cpu", weights_only=False)
    args = Namespace(**checkpoint["args"])
    expert = str(checkpoint["expert"]["name"])
    outer_fold = int(checkpoint["outer_fold"])
    inner_fold = int(checkpoint["inner_fold"])
    resolved_dataset = resolve_dataset_root(args, dataset_root)
    plan = prepare_dataset(resolved_dataset, expert, outer_fold, inner_fold)
    validate_legacy_identity(legacy_run, checkpoint, plan)
    destination = canonical_backfill_dir(
        output_root=output_root.resolve(),
        experiment_id=experiment_id,
        expert=expert,
        outer_fold=outer_fold,
    )
    if destination.exists():
        raise FileExistsError(f"canonical backfill destination already exists: {destination}")

    layout = RunLayout.create(destination)
    write_experiment_metadata(
        output_root.resolve() / experiment_id,
        experiment_id,
        layout,
        plan,
    )
    record_backfill_metadata(
        layout=layout,
        legacy_run=legacy_run,
        args=args,
        plan=plan,
        training_revision=training_revision,
        started_at=started_at,
    )
    copy_legacy_artifacts(legacy_run, layout)
    append_log(layout, "copied legacy checkpoints, outer-test metrics and log")
    replay_history(legacy_run / "history.json", layout)
    append_log(layout, "replayed epoch history into TensorBoard and CSV")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for backfill but is unavailable")
    model = build_resunet(args.encoder, None, num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    loader_args = Namespace(**vars(args))
    loader_args.batch_size = batch_size
    loader_args.num_workers = num_workers
    validation_loader = make_loader(plan, plan.val, loader_args, device, training=False)
    rows = write_validation_artifacts(
        model=model,
        loader=validation_loader,
        device=device,
        output_root=layout.root,
        expert_name=plan.expert_name,
        ignore_value=plan.ignore_value,
    )
    image_writer = create_tensorboard_writer(layout.tensorboard)
    try:
        write_tensorboard_validation_images(
            image_writer,
            layout.root,
            rows,
            int(checkpoint["epoch"]),
        )
    finally:
        image_writer.close()
    export_tensorboard_images(layout)
    append_log(layout, f"wrote validation artifacts for {len(plan.val)} tiles from best.pt")
    build_training_dashboard(layout)
    backfill_metadata = read_json(layout.config / "backfill.json")
    backfill_metadata.update(
        {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "validation_tile_count": len(plan.val),
        }
    )
    write_json(layout.config / "backfill.json", backfill_metadata)
    append_log(layout, "dashboard complete")
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return destination


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--legacy-root", type=Path, required=True)
    result.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
    )
    result.add_argument("--experiment-id")
    result.add_argument("--dataset-root", type=Path)
    result.add_argument("--device", default="auto")
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--expert", action="append", dest="experts")
    result.add_argument("--fold", action="append", type=int, dest="folds")
    return result


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    legacy_root = options.legacy_root.resolve()
    runs = discover_legacy_runs(legacy_root)
    if options.experts:
        runs = [run for run in runs if run.parent.name in options.experts]
    if options.folds:
        requested_folds = set(options.folds)
        runs = [run for run in runs if int(run.name.removeprefix("fold")) in requested_folds]
    if not runs:
        raise SystemExit(f"no complete legacy runs found under {legacy_root}")
    started_at = infer_started_at(legacy_root)
    training_revision = infer_training_revision(legacy_root)
    experiment_id = options.experiment_id or legacy_root.name
    completed = []
    for legacy_run in runs:
        print(f"backfill start: {legacy_run}", flush=True)
        destination = backfill_run(
            legacy_run=legacy_run,
            output_root=options.output_root,
            experiment_id=experiment_id,
            dataset_root=options.dataset_root,
            started_at=started_at,
            training_revision=training_revision,
            device_name=options.device,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
        )
        completed.append(destination)
        print(f"backfill complete: {destination}", flush=True)
    print(f"completed {len(completed)} canonical run reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
