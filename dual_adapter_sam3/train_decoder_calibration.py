"""Fine-tune the final PixelDecoder block and semantic head from DA-SAM3 Stage 2."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from sam2_adapter.reporting import append_log

from .checkpoints import load_adaptation_checkpoint
from .concepts import CHANNEL_ORDER, concept_contract_record, load_concept_registry
from .data import MonumentDeteriorationDataset
from .decoder_calibration import (
    CALIBRATION_EXPERT,
    configure_decoder_calibration,
    load_decoder_calibration_checkpoint,
    save_decoder_calibration_checkpoint,
)
from .losses import sam2_craquelure_hybrid_segmentation_loss
from .metrics import MultilabelConfusion
from .model import DualAdapterSam3, build_dual_adapter_model
from .reporting import EpochReporter, RunLayout, finalize_reporting, write_json
from .sam3_integration import DEFAULT_CHECKPOINT, file_sha256
from .splits import fold_membership, load_split_contract, source_group_index
from .train import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    _loader,
    _objective,
    _qualitative_rows,
    _to_device,
)
from .training import assert_finite_gradients, set_reproducible_seed


DEFAULT_SOURCE_EXPERIMENT = "2026-09-01_dual-adapter-sam3-multilabel-512_seed42"
DEFAULT_EXPERIMENT = "2026-09-03_da-sam3-decoder-calibration-512_seed42"
CRAQUELURE_LOSS_PROFILES = ("original", "sam2_bce_dice")


def calibration_run_root(experiment_root: Path, fold: int) -> Path:
    if fold not in range(5):
        raise ValueError(f"fold must be in [0, 5), got {fold}")
    return experiment_root / "5fold" / CALIBRATION_EXPERT / f"fold{fold}"


def source_checkpoint_path(source_experiment_root: Path, fold: int) -> Path:
    if fold not in range(5):
        raise ValueError(f"fold must be in [0, 5), got {fold}")
    return (
        source_experiment_root
        / "5fold"
        / "da_sam3"
        / f"fold{fold}"
        / "artifacts"
        / "checkpoints"
        / "stage2_best.pt"
    )


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_or_validate_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing shared experiment metadata is incompatible: {path}")
        return
    write_json(path, payload)


def _calibration_segmentation_loss(
    output: Any,
    batch: dict[str, Any],
    *,
    craquelure_loss: str,
) -> torch.Tensor:
    if craquelure_loss == "original":
        return _objective(output, batch).segmentation
    if craquelure_loss == "sam2_bce_dice":
        return sam2_craquelure_hybrid_segmentation_loss(
            output.logits,
            output.presence_logits,
            batch["targets"],
            batch["valid_masks"],
            batch["class_present"],
        )
    raise ValueError(f"unsupported craquelure loss profile: {craquelure_loss!r}")


def _run_calibration_epoch(
    model: DualAdapterSam3,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    craquelure_loss: str,
) -> float:
    # Keep the frozen official model deterministic. requires_grad, not train(),
    # controls updates for Conv2d and GroupNorm in the selected calibration scope.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    first_batch = True
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
            loss = _calibration_segmentation_loss(
                output,
                batch,
                craquelure_loss=craquelure_loss,
            )
        loss.backward()
        if first_batch:
            assert_finite_gradients(model)
            first_batch = False
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        batch_size = int(batch["image"].shape[0])
        total += float(loss.detach()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("decoder-calibration training loader is empty")
    return total / count


@torch.no_grad()
def _run_calibration_validation(
    model: DualAdapterSam3,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    craquelure_loss: str,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    confusion = MultilabelConfusion.empty(device=device)
    segmentation_total = 0.0
    count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
            loss = _calibration_segmentation_loss(
                output,
                batch,
                craquelure_loss=craquelure_loss,
            )
        batch_size = int(batch["image"].shape[0])
        segmentation_total += float(loss) * batch_size
        count += batch_size
        confusion.update(output.logits, batch["targets"], batch["valid_masks"])
    if count == 0:
        raise RuntimeError("decoder-calibration validation loader is empty")
    return segmentation_total / count, confusion.compute()


def train_fold(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official SAM3 decoder calibration")
    if args.epochs < 1:
        raise ValueError("decoder calibration requires at least one epoch")
    set_reproducible_seed(42)
    device = torch.device("cuda")
    registry = load_concept_registry(args.concepts)
    split = load_split_contract(args.splits, dataset_root=args.dataset)
    membership = fold_membership(split, args.fold)
    group_index = source_group_index(split)

    train_dataset = MonumentDeteriorationDataset(
        args.dataset,
        membership.train,
        group_index,
        train_augmentation=True,
        seed=42,
    )
    validation_dataset = MonumentDeteriorationDataset(
        args.dataset,
        membership.validation,
        group_index,
        train_augmentation=False,
        seed=42,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        train=True,
        workers=args.workers,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        train=False,
        workers=args.workers,
    )

    experiment_root = PROJECT_ROOT / "runs" / args.experiment_id
    run_root = calibration_run_root(experiment_root, args.fold)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run directory already contains artifacts: {run_root}")
    source_root = PROJECT_ROOT / "runs" / args.source_experiment_id
    source_path = source_checkpoint_path(source_root, args.fold)
    if not source_path.is_file():
        raise FileNotFoundError(f"source Stage2 checkpoint is missing: {source_path}")
    source_sha256 = file_sha256(source_path)

    model = build_dual_adapter_model(
        registry,
        model_variant="da_sam3",
        checkpoint=args.checkpoint,
        device=device,
    ).to(device)
    source_checkpoint = load_adaptation_checkpoint(
        source_path,
        model,
        expected_prompt_hash=registry.sha256,
        expected_split_hash=split.sha256,
    )
    trainable_names = configure_decoder_calibration(model)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    layout = RunLayout.create(run_root)
    info = experiment_root / "info"
    experiment_contract = {
        "experiment_type": "decoder_calibration",
        "expert": CALIBRATION_EXPERT,
        "base_model_variant": "da_sam3",
        "source_experiment_id": args.source_experiment_id,
        "trainable_scope": "final PixelDecoder Conv2d + final GroupNorm + semantic_seg_head",
        "craquelure_loss": args.craquelure_loss,
        "loss_contract": (
            "original DA-SAM3 Dice + weighted focal(pos_weight=2,gamma=2) + presence"
            if args.craquelure_loss == "original"
            else (
                "craquelure: unweighted BCE + 0.65 squared-denominator soft Dice; "
                "loss: original DA-SAM3 Dice + weighted focal(pos_weight=2,gamma=2); "
                "both retain presence_weight=0.1 and are averaged equally"
            )
        ),
        "checkpoint_selection": (
            f"minimum validation segmentation loss under {args.craquelure_loss} profile"
        ),
        "threshold": 0.5,
        "outer_test_policy": (
            "excluded from checkpoint selection; prior baseline outer-test results were "
            "observed before this diagnostic architecture ablation"
        ),
    }
    _write_or_validate_json(info / "experiment_contract.json", experiment_contract)
    _write_or_validate_json(info / "prompt_contract.json", concept_contract_record(registry))
    _write_or_validate_json(info / "split_contract.json", split.payload)
    _write_or_validate_json(
        info / "dataset_contract.json",
        json.loads((Path(args.dataset) / "manifest.json").read_text(encoding="utf-8")),
    )
    _write_or_validate_json(
        info / "model_contract.json",
        {"experiment": experiment_contract, "official_model": model.model_contract},
    )
    _write_or_validate_json(
        info / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "git_revision": _git_revision(),
            "seed": 42,
        },
    )
    write_json(layout.config / "resolved_config.json", vars(args))
    write_json(
        layout.config / "source_checkpoint.json",
        {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "stage": source_checkpoint["stage"],
            "epoch": int(source_checkpoint["epoch"]),
            "validation_segmentation_loss": float(
                source_checkpoint["validation_segmentation_loss"]
            ),
        },
    )
    write_json(
        layout.config / "trainable_scope.json",
        {
            "names": list(trainable_names),
            "tensor_count": len(trainable_names),
            "parameter_count": trainable_parameters,
        },
    )

    writer = SummaryWriter(log_dir=layout.tensorboard)
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")
    best_path = layout.checkpoints / "best.pt"
    last_path = layout.checkpoints / "last.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_calibration_epoch(
            model,
            train_loader,
            optimizer,
            device,
            craquelure_loss=args.craquelure_loss,
        )
        validation_loss, metrics = _run_calibration_validation(
            model,
            validation_loader,
            device,
            craquelure_loss=args.craquelure_loss,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        macro = metrics["macro"]
        reporter.record(
            epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            f1=macro["f1"],
            precision=macro["precision"],
            recall=macro["recall"],
            iou=macro["iou"],
            accuracy=macro["accuracy"],
            learning_rate=learning_rate,
        )
        for concept in CHANNEL_ORDER:
            writer.add_scalar(
                f"metrics/{concept}/f1", metrics["per_class"][concept]["f1"], epoch
            )
            writer.add_scalar(
                f"metrics/{concept}/iou", metrics["per_class"][concept]["iou"], epoch
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_decoder_calibration_checkpoint(
                best_path,
                model,
                epoch=epoch,
                validation_segmentation_loss=validation_loss,
                registry_hash=registry.sha256,
                split_hash=split.sha256,
                source_checkpoint_sha256=source_sha256,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        save_decoder_calibration_checkpoint(
            last_path,
            model,
            epoch=epoch,
            validation_segmentation_loss=validation_loss,
            registry_hash=registry.sha256,
            split_hash=split.sha256,
            source_checkpoint_sha256=source_sha256,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        scheduler.step()
        status = {
            "stage": "decoder_calibration",
            "craquelure_loss": args.craquelure_loss,
            "fold": args.fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_segmentation_loss": validation_loss,
            "macro_f1": macro["f1"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        line = json.dumps(status, ensure_ascii=False)
        append_log(layout, line)
        print(line, flush=True)

    selected = load_decoder_calibration_checkpoint(
        best_path,
        model,
        expected_prompt_hash=registry.sha256,
        expected_split_hash=split.sha256,
        expected_source_checkpoint_sha256=source_sha256,
    )
    validation_rows = _qualitative_rows(model, validation_loader, device, layout)
    write_json(
        layout.metrics / "selection.json",
        {
            "criterion": (
                f"minimum validation segmentation loss under {args.craquelure_loss} profile"
            ),
            "source_stage2_checkpoint": str(source_path.resolve()),
            "source_stage2_checkpoint_sha256": source_sha256,
            "selected_checkpoint": str(best_path.resolve()),
            "selected_epoch": int(selected["epoch"]),
            "selected_validation_segmentation_loss": float(
                selected["validation_segmentation_loss"]
            ),
            "outer_test_excluded_from_checkpoint_selection": True,
        },
    )
    finalize_reporting(
        layout,
        writer=writer,
        reporter=reporter,
        validation_rows=validation_rows,
        selected_epoch=int(selected["epoch"]),
        outer_test_metrics={
            "status": "deferred_until_all_five_validation_checkpoints_are_locked",
            "selected_epoch": int(selected["epoch"]),
            "selection_source": "minimum validation segmentation loss",
            "outer_test_excluded_from_checkpoint_selection": True,
            "outer_test_previously_observed_for_architecture_design": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_root": str(run_root),
                "selected_checkpoint": str(best_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source-experiment-id", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--concepts", type=Path, default=PROJECT_ROOT / "configs" / "concepts.yaml"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "configs" / "splits.json"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--craquelure-loss",
        choices=CRAQUELURE_LOSS_PROFILES,
        default="original",
        help="pixel objective for the crack_craquelure channel",
    )
    return parser


def main() -> None:
    train_fold(build_parser().parse_args())


if __name__ == "__main__":
    main()
