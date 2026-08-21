"""Train merged-crack SAM2-SAC at 512px with nested five-fold CV.

Usage (from the repository root)::

    crackseg_env/bin/python -m sam2_sac.train_h0

Raw label 1 is the merged crack foreground. Raw label 0 is background and all
other defect labels are excluded from loss and metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from crackseg_common.data_plan import binary_weighting_record
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sam2_sac.data import H0DataPlan, H0TileDataset, denormalize_image, prepare_data_plan
from sam2_sac.h0_core import (
    NativeSAM2MaskDecoder,
    binary_bce_dice_loss,
    freeze_except_layer_norm,
    load_trainable_state_dict,
    make_stage_target,
    model_parameter_counts,
    trainable_state_dict,
)
from sam2_sac.metrics import binary_summary
from sam2_sac.reporting import (
    EpochReporter,
    RunLayout,
    append_log,
    finalize_reporting,
    save_qualitative_example,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET = WORKSPACE_ROOT / "datasets" / "dataset_clean_v2_merged_craquelure"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_EXPERIMENT = "2026-08-21_sam2-sac-hiera-large_merged-crack_512_80ep_seed42"
Stage = Literal["foreground"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("foreground",), default="foreground")
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="run every requested epoch and retain last.pt for comparison",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dice-weight", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.image_size != 512:
        parser.error("H0 is locked to the requested 512x512 experiment; use --image-size 512")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    if any(fold not in range(5) for fold in args.folds):
        parser.error("this dataset has five folds numbered 0 through 4")
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds must not contain duplicates")
    return args


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _make_loader(
    plan: H0DataPlan,
    names: tuple[str, ...],
    *,
    train: bool,
    args: argparse.Namespace,
    batch_size: int | None = None,
) -> DataLoader[dict[str, Tensor | str]]:
    generator = torch.Generator()
    generator.manual_seed(args.seed + 10_000 * plan.outer_fold + (1 if train else 0))
    workers = min(args.num_workers, os.cpu_count() or 1)
    return DataLoader(
        H0TileDataset(plan, names, train_augmentation=train),
        batch_size=batch_size or args.batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers else None,
        generator=generator,
    )


def _binary_counts(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_value: int,
) -> tuple[int, int, int]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    valid = target != ignore_value
    prediction = logits >= 0.0
    actual = target == 1
    return (
        int((prediction & actual & valid).sum().item()),
        int((prediction & ~actual & valid).sum().item()),
        int((~prediction & actual & valid).sum().item()),
    )


def _merge_counts(
    current: tuple[int, int, int] | None,
    update: tuple[int, int, int],
) -> tuple[int, int, int]:
    if current is None:
        return update
    return tuple(first + second for first, second in zip(current, update, strict=True))


def _evaluate_stage(
    model: NativeSAM2MaskDecoder,
    loader: DataLoader[dict[str, Tensor | str]],
    *,
    stage: Stage,
    plan: H0DataPlan,
    device: torch.device,
    amp: bool,
    dice_weight: float,
    positive_weight: float,
    layout: RunLayout | None = None,
) -> dict[str, Any]:
    """Evaluate a selected stage, optionally saving all validation source PNGs."""

    model.eval()
    losses: list[float] = []
    counts_by_group: dict[str, tuple[int, int, int]] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = _batch_tensor(batch, "image", device)
            source = _batch_tensor(batch, "mask", device).long()
            target = make_stage_target(source, stage=stage, ignore_value=plan.ignore_value)
            with _autocast(device, amp):
                logits = model(images)
                loss = binary_bce_dice_loss(
                    logits,
                    target,
                    ignore_value=plan.ignore_value,
                    dice_weight=dice_weight,
                    positive_weight=positive_weight,
                )
            losses.append(float(loss.detach().cpu()))
            groups = [str(value) for value in batch["source_group"]]  # type: ignore[index]
            for index, group in enumerate(groups):
                current = _binary_counts(
                    logits[index : index + 1], target[index : index + 1], ignore_value=plan.ignore_value
                )
                counts_by_group[group] = _merge_counts(counts_by_group.get(group), current)
                if layout is None:
                    continue
                valid = target[index] != plan.ignore_value
                target_binary = ((target[index] == 1) & valid).detach().cpu().numpy()
                prediction_binary = ((logits[index, 0] >= 0.0) & valid).detach().cpu().numpy()
                rgb = (
                    denormalize_image(images[index])
                    .permute(1, 2, 0)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .numpy()
                )
                rows.append(
                    save_qualitative_example(
                        layout,
                        image_id=str(batch["name"][index]),  # type: ignore[index]
                        input_rgb=rgb,
                        target=target_binary,
                        prediction=prediction_binary,
                        target_class="foreground",
                    )
                )
    summary = binary_summary(counts_by_group)
    return {
        "loss": sum(losses) / len(losses),
        **summary,
        "per_image_rows": rows,
    }


def _train_epoch(
    model: NativeSAM2MaskDecoder,
    loader: DataLoader[dict[str, Tensor | str]],
    optimizer: AdamW,
    *,
    stage: Stage,
    plan: H0DataPlan,
    device: torch.device,
    amp: bool,
    dice_weight: float,
    positive_weight: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        images = _batch_tensor(batch, "image", device)
        source = _batch_tensor(batch, "mask", device).long()
        target = make_stage_target(source, stage=stage, ignore_value=plan.ignore_value)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = binary_bce_dice_loss(
                logits,
                target,
                ignore_value=plan.ignore_value,
                dice_weight=dice_weight,
                positive_weight=positive_weight,
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _batch_tensor(batch: dict[str, Tensor | str], name: str, device: torch.device) -> Tensor:
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] should be a Tensor")
    return value.to(device, non_blocking=True)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _checkpoint_payload(
    model: NativeSAM2MaskDecoder,
    *,
    stage: Stage,
    epoch: int,
    best_validation_loss: float,
    args: argparse.Namespace,
    plan: H0DataPlan,
    checkpoint_sha256: str,
    optimizer: AdamW,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "base_checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": checkpoint_sha256,
        "sam2_config": args.sam2_config,
        "image_size": args.image_size,
        "manifest_hash": plan.manifest_hash,
        "adaptation": "LayerNorm affine parameters only",
        "adaptation_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
    }


def _save_checkpoint(
    path: Path,
    model: NativeSAM2MaskDecoder,
    *,
    stage: Stage,
    epoch: int,
    best_validation_loss: float,
    args: argparse.Namespace,
    plan: H0DataPlan,
    checkpoint_sha256: str,
    optimizer: AdamW,
) -> None:
    torch.save(
        _checkpoint_payload(
            model,
            stage=stage,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            args=args,
            plan=plan,
            checkpoint_sha256=checkpoint_sha256,
            optimizer=optimizer,
        ),
        path,
    )


def _load_adaptation_checkpoint(
    model: NativeSAM2MaskDecoder,
    path: Path,
    *,
    expected_stage: Stage,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    if payload.get("stage") != expected_stage:
        raise ValueError(f"expected a {expected_stage!r} checkpoint, got {payload.get('stage')!r}: {path}")
    if payload.get("base_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("base SAM2 checkpoint hash does not match the adaptation checkpoint")
    state = payload.get("adaptation_state")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is missing adaptation_state: {path}")
    load_trainable_state_dict(model, state)
    return payload


def _stage_run_dir(experiment_id: str, stage: Stage, fold: int) -> Path:
    return PROJECT_ROOT / "runs" / experiment_id / "5fold" / stage / f"fold{fold}"


def _write_run_metadata(
    layout: RunLayout,
    *,
    args: argparse.Namespace,
    plan: H0DataPlan,
    stage: Stage,
    checkpoint_sha256: str,
    parameter_counts: dict[str, int],
    class_weighting: dict[str, object],
) -> None:
    revision = _git_revision()
    write_json(layout.config / "args.json", vars(args))
    write_json(layout.config / "dataset.json", plan.record())
    write_json(
        layout.config / "model.json",
        {
            "model": "SAM2.1 Hiera Large",
            "sam2_config": args.sam2_config,
            "base_checkpoint": str(args.checkpoint.resolve()),
            "base_checkpoint_sha256": checkpoint_sha256,
            "image_size": args.image_size,
            "architecture_modified": False,
            "native_components": ["image_encoder", "prompt_encoder", "mask_decoder"],
            "prompt_encoder": {
                "points": None,
                "boxes": None,
                "masks": None,
                "dense_embedding": "SAM2 learned no_mask_embed",
            },
            "mask_decoder_path": "official SAM2ImagePredictor direct decoder path; dynamic 512 feature shapes",
            "adaptation": "LayerNorm affine parameters only; every other parameter frozen",
            "parameter_counts": parameter_counts,
            "morphology": "none (H0)",
        },
    )
    write_json(
        layout.config / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "git_revision": revision,
            "packages": _package_versions(),
        },
    )
    write_json(
        layout.config / "run.json",
        {
            "experiment_id": args.experiment_id,
            "stage": stage,
            "outer_fold": plan.outer_fold,
            "inner_fold": plan.inner_fold,
            "selection_metric": "val_loss",
            "selection_mode": "min",
            "selection_scope": "validation only; outer test excluded",
            "early_stopping": not args.no_early_stop,
            "max_epochs": args.epochs,
            "loss": "fixed 2:1 foreground-weighted BCEWithLogits + 0.65 * unweighted soft Dice",
            "class_weighting": class_weighting,
            "inference_rule": "sigmoid(logit) >= 0.5",
            "threshold": 0.5,
            "command": [sys.executable, *sys.argv],
        },
    )
    append_log(layout, f"created={datetime.now().astimezone().isoformat(timespec='seconds')}")
    append_log(layout, f"stage={stage} outer_fold={plan.outer_fold} inner_fold={plan.inner_fold}")


def _update_experiment_metadata(
    *,
    args: argparse.Namespace,
    plan: H0DataPlan,
    stage: Stage,
    layout: RunLayout,
) -> None:
    experiment_root = PROJECT_ROOT / "runs" / args.experiment_id
    info_path = experiment_root / "info" / "experiment.json"
    existing: dict[str, Any] = {}
    if info_path.is_file():
        existing = json.loads(info_path.read_text(encoding="utf-8"))
    by_stage = dict(existing.get("folds", {}))
    stage_folds = set(by_stage.get(stage, []))
    stage_folds.add(plan.outer_fold)
    by_stage[stage] = sorted(stage_folds)
    run_paths = set(existing.get("runs", []))
    run_paths.add(layout.root.relative_to(experiment_root).as_posix())
    write_json(
        info_path,
        {
            "schema_version": 1,
            "experiment_id": args.experiment_id,
            "layout": "5fold",
            "fold_count": 5,
            "experts": sorted(by_stage),
            "folds": by_stage,
            "runs": sorted(run_paths),
            "dataset_root": str(plan.root),
            "manifest_hash": plan.manifest_hash,
            "sac": {
                "image_size": 512,
                "morphology": "none",
                "task": "binary merged crack foreground",
            },
            "created_at": existing.get("created_at", datetime.now().astimezone().isoformat()),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pillow", "tensorboard", "torch", "torchvision", "hydra-core", "omegaconf")
    values: dict[str, str | None] = {}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_stage_has_supervision(plan: H0DataPlan, stage: Stage) -> None:
    partitions = {
        "train": plan.train_counts,
        "validation": plan.val_counts,
        "outer_test": plan.test_counts,
    }
    for partition, counts in partitions.items():
        background, foreground, excluded = counts
        del background
        del excluded
        if foreground == 0:
            raise ValueError(f"{stage} has no positive pixels in {partition}")


def _build_model(args: argparse.Namespace, device: torch.device) -> NativeSAM2MaskDecoder:
    return NativeSAM2MaskDecoder(
        checkpoint=args.checkpoint,
        config=args.sam2_config,
        image_size=args.image_size,
        device=device,
    )


def _freeze_active_native_layer_norm(model: NativeSAM2MaskDecoder) -> tuple[str, ...]:
    """Adapt only active encoder/prompt/decoder LayerNorms, never video-memory modules."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    components = (
        ("model.image_encoder", model.model.image_encoder),
        ("model.sam_prompt_encoder", model.model.sam_prompt_encoder),
        ("model.sam_mask_decoder", model.model.sam_mask_decoder),
    )
    names: list[str] = []
    for prefix, component in components:
        for name in freeze_except_layer_norm(component):
            names.append(f"{prefix}.{name}")
    return tuple(sorted(names))


def _train_stage(
    args: argparse.Namespace,
    *,
    plan: H0DataPlan,
    stage: Stage,
    device: torch.device,
    checkpoint_sha256: str,
) -> Path:
    _assert_stage_has_supervision(plan, stage)
    output = _stage_run_dir(args.experiment_id, stage, plan.outer_fold)
    if output.exists() and any(output.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"refusing to overwrite an existing run: {output}; choose a new --experiment-id")
    layout = RunLayout.create(output)
    model = _build_model(args, device)
    trainable_names = _freeze_active_native_layer_norm(model)
    counts = model_parameter_counts(model)
    if not counts["trainable"]:
        raise RuntimeError("no LayerNorm affine parameters were found to adapt")
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weighting = binary_weighting_record(np.asarray(plan.train_counts[:2]))
    positive_weight = float(class_weighting["bce_positive_weight"])
    _write_run_metadata(
        layout,
        args=args,
        plan=plan,
        stage=stage,
        checkpoint_sha256=checkpoint_sha256,
        parameter_counts=counts,
        class_weighting=class_weighting,
    )
    write_json(layout.config / "trainable_parameters.json", {"names": list(trainable_names), "count": counts["trainable"]})
    _update_experiment_metadata(args=args, plan=plan, stage=stage, layout=layout)
    writer = SummaryWriter(log_dir=str(layout.tensorboard))
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    train_loader = _make_loader(plan, plan.train, train=True, args=args)
    validation_loader = _make_loader(plan, plan.val, train=False, args=args)
    outer_loader = _make_loader(plan, plan.test, train=False, args=args)
    best_loss = float("inf")
    best_epoch = 0
    stalled_epochs = 0
    finalized = False
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch(
                model,
                train_loader,
                optimizer,
                stage=stage,
                plan=plan,
                device=device,
                amp=args.amp,
                dice_weight=args.dice_weight,
                positive_weight=positive_weight,
            )
            validation = _evaluate_stage(
                model,
                validation_loader,
                stage=stage,
                plan=plan,
                device=device,
                amp=args.amp,
                dice_weight=args.dice_weight,
                positive_weight=positive_weight,
            )
            micro = validation["tile_micro"]
            learning_rate = float(optimizer.param_groups[0]["lr"])
            reporter.record(
                epoch,
                train_loss=train_loss,
                validation_loss=float(validation["loss"]),
                f1=micro["mf1"],
                precision=micro["mprecision"],
                recall=micro["mrecall"],
                iou=micro["miou"],
                learning_rate=learning_rate,
            )
            _save_checkpoint(
                layout.checkpoints / "last.pt",
                model,
                stage=stage,
                epoch=epoch,
                best_validation_loss=best_loss,
                args=args,
                plan=plan,
                checkpoint_sha256=checkpoint_sha256,
                optimizer=optimizer,
            )
            message = (
                f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} "
                f"val_loss={validation['loss']:.6f} val_f1={_format_metric(micro['mf1'])} "
                f"lr={learning_rate:.8f}"
            )
            print(message, flush=True)
            append_log(layout, message)
            if float(validation["loss"]) < best_loss:
                best_loss = float(validation["loss"])
                best_epoch = epoch
                stalled_epochs = 0
                _save_checkpoint(
                    layout.checkpoints / "best.pt",
                    model,
                    stage=stage,
                    epoch=epoch,
                    best_validation_loss=best_loss,
                    args=args,
                    plan=plan,
                    checkpoint_sha256=checkpoint_sha256,
                    optimizer=optimizer,
                )
            else:
                stalled_epochs += 1
            scheduler.step()
            if not args.no_early_stop and stalled_epochs >= args.patience:
                append_log(layout, f"early_stop=epoch{epoch} patience={args.patience}")
                break
        if not best_epoch:
            raise RuntimeError("no finite validation checkpoint was selected")
        _load_adaptation_checkpoint(
            model,
            layout.checkpoints / "last.pt",
            expected_stage=stage,
            checkpoint_sha256=checkpoint_sha256,
        )
        validation_last = _evaluate_stage(
            model,
            validation_loader,
            stage=stage,
            plan=plan,
            device=device,
            amp=args.amp,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
        )
        outer_last = _evaluate_stage(
            model,
            outer_loader,
            stage=stage,
            plan=plan,
            device=device,
            amp=args.amp,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
        )
        write_json(
            layout.metrics / "last_validation_metrics.json",
            {key: value for key, value in validation_last.items() if key != "per_image_rows"},
        )
        write_json(
            layout.metrics / "last_outer_test_metrics.json",
            {
                "scope": "last_checkpoint_comparison_only; not used for selection",
                "stage": stage,
                "checkpoint": "artifacts/checkpoints/last.pt",
                "epoch": int(torch.load(layout.checkpoints / "last.pt", map_location="cpu", weights_only=False)["epoch"]),
                "loss": outer_last["loss"],
                "tile_micro": outer_last["tile_micro"],
                "expert_panel_macro": outer_last["expert_panel_macro"],
            },
        )
        selected = _load_adaptation_checkpoint(
            model,
            layout.checkpoints / "best.pt",
            expected_stage=stage,
            checkpoint_sha256=checkpoint_sha256,
        )
        validation_selected = _evaluate_stage(
            model,
            validation_loader,
            stage=stage,
            plan=plan,
            device=device,
            amp=args.amp,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
            layout=layout,
        )
        outer_test = _evaluate_stage(
            model,
            outer_loader,
            stage=stage,
            plan=plan,
            device=device,
            amp=args.amp,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
        )
        write_json(
            layout.metrics / "selected_validation_metrics.json",
            {key: value for key, value in validation_selected.items() if key != "per_image_rows"},
        )
        outer_record: dict[str, Any] = {
            "scope": "outer_test_not_used_for_selection",
            "stage": stage,
            "selected_checkpoint": "artifacts/checkpoints/best.pt",
            "selected_epoch": int(selected["epoch"]),
            "selection_metric": "validation fixed 2:1 foreground-weighted BCEWithLogits + 0.65 unweighted Dice loss",
            "threshold": 0.5,
            "inference_rule": "sigmoid(logit) >= 0.5",
            "loss": outer_test["loss"],
            "tile_micro": outer_test["tile_micro"],
            "expert_panel_macro": outer_test["expert_panel_macro"],
        }
        finalize_reporting(
            layout,
            writer=writer,
            reporter=reporter,
            validation_rows=validation_selected["per_image_rows"],
            selected_epoch=best_epoch,
            outer_test_metrics=outer_record,
        )
        finalized = True
        append_log(layout, f"completed selected_epoch={best_epoch} best_val_loss={best_loss:.6f}")
        return layout.checkpoints / "best.pt"
    finally:
        if not finalized:
            reporter.close()
            writer.close()
        del model
        torch.cuda.empty_cache()


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H0 training requires the configured CUDA device")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    checkpoint_sha256 = _sha256(args.checkpoint)
    print(
        f"H0 start: 512px native SAM2.1 Hiera-L, folds={list(args.folds)}, stage={args.stage}, "
        f"checkpoint_sha256={checkpoint_sha256}",
        flush=True,
    )
    for fold in args.folds:
        plan = prepare_data_plan(args.dataset, outer_fold=fold)
        _train_stage(
            args,
            plan=plan,
            stage="foreground",
            device=device,
            checkpoint_sha256=checkpoint_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
