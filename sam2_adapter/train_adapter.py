"""Train one binary merged-crack SAM2-Adapter with nested five-fold CV."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from crackseg_common.data_plan import binary_weighting_record
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sam2_adapter.adapter_core import make_binary_target
from sam2_adapter.adapter_model import SAM2AdapterMaskDecoder
from sam2_adapter.data import H0DataPlan, denormalize_image, prepare_data_plan
from sam2_adapter.h0_core import (
    binary_bce_dice_loss,
    load_trainable_state_dict,
    model_parameter_counts,
    trainable_state_dict,
)
from sam2_adapter.metrics import binary_summary
from sam2_adapter.reporting import (
    EpochReporter,
    RunLayout,
    append_log,
    finalize_reporting,
    save_qualitative_example,
    write_json,
)
from sam2_adapter.runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    _autocast,
    _batch_tensor,
    _git_revision,
    _make_loader,
    _package_versions,
    _seed_everything,
    _sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TASK_NAME = "foreground"
FIXED_THRESHOLD = 0.5
DEFAULT_EXPERIMENT = "2026-08-21_sam2-adapter-hiera-large_merged-crack_512_80ep_seed42"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dice-weight", type=float, default=0.65)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--scale-factor", type=int, default=32)
    parser.add_argument("--highpass-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.image_size != 512:
        parser.error("the controlled comparison is locked to 512x512")
    positive = (args.epochs, args.batch_size, args.accumulation_steps, args.scale_factor)
    if any(value < 1 for value in positive):
        parser.error("epochs, batch size, accumulation steps, and scale factor must be positive")
    if not 0.0 < args.highpass_rate < 1.0:
        parser.error("--highpass-rate must be in (0, 1)")
    if any(fold not in range(5) for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        parser.error("--folds must contain unique values from 0 through 4")
    return args


def _run_dir(experiment_id: str, fold: int) -> Path:
    return PROJECT_ROOT / "runs" / experiment_id / "5fold" / TASK_NAME / f"fold{fold}"


def _build_model(args: argparse.Namespace, device: torch.device) -> SAM2AdapterMaskDecoder:
    return SAM2AdapterMaskDecoder(
        checkpoint=args.checkpoint,
        config=args.sam2_config,
        image_size=args.image_size,
        device=device,
        scale_factor=args.scale_factor,
        highpass_rate=args.highpass_rate,
    )


def _counts(logits: Tensor, target: Tensor, *, ignore_value: int) -> tuple[int, int, int]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    valid = target != ignore_value
    prediction = torch.sigmoid(logits) >= FIXED_THRESHOLD
    actual = target == 1
    return (
        int((prediction & actual & valid).sum().item()),
        int((prediction & ~actual & valid).sum().item()),
        int((~prediction & actual & valid).sum().item()),
    )


def _merge(current: tuple[int, int, int] | None, update: tuple[int, int, int]) -> tuple[int, int, int]:
    if current is None:
        return update
    return tuple(a + b for a, b in zip(current, update, strict=True))


def _evaluate(
    model: SAM2AdapterMaskDecoder,
    loader: DataLoader[dict[str, Tensor | str]],
    *,
    plan: H0DataPlan,
    device: torch.device,
    amp: bool,
    dice_weight: float,
    positive_weight: float,
    layout: RunLayout | None = None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    panel_counts: dict[str, tuple[int, int, int]] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = _batch_tensor(batch, "image", device)
            source = _batch_tensor(batch, "mask", device).long()
            target = make_binary_target(source, ignore_value=plan.ignore_value)
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
                count = _counts(
                    logits[index : index + 1],
                    target[index : index + 1],
                    ignore_value=plan.ignore_value,
                )
                panel_counts[group] = _merge(panel_counts.get(group), count)
                valid = target[index] != plan.ignore_value
                if layout is not None:
                    target_binary = ((target[index] == 1) & valid).cpu().numpy()
                    prediction_binary = (
                        (torch.sigmoid(logits[index, 0]) >= FIXED_THRESHOLD) & valid
                    ).cpu().numpy()
                    rgb = (
                        denormalize_image(images[index]).permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
                    )
                    rows.append(
                        save_qualitative_example(
                            layout,
                            image_id=str(batch["name"][index]),  # type: ignore[index]
                            input_rgb=rgb,
                            target=target_binary,
                            prediction=prediction_binary,
                            target_class=TASK_NAME,
                        )
                    )
    result: dict[str, Any] = {
        "loss": sum(losses) / len(losses),
        **binary_summary(panel_counts),
        "per_image_rows": rows,
    }
    return result


def _train_epoch(
    model: SAM2AdapterMaskDecoder,
    loader: DataLoader[dict[str, Tensor | str]],
    optimizer: AdamW,
    *,
    plan: H0DataPlan,
    device: torch.device,
    amp: bool,
    dice_weight: float,
    positive_weight: float,
    accumulation_steps: int,
    gradient_clip: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for batch_index, batch in enumerate(loader, start=1):
        images = _batch_tensor(batch, "image", device)
        source = _batch_tensor(batch, "mask", device).long()
        target = make_binary_target(source, ignore_value=plan.ignore_value)
        with _autocast(device, amp):
            loss = binary_bce_dice_loss(
                model(images),
                target,
                ignore_value=plan.ignore_value,
                dice_weight=dice_weight,
                positive_weight=positive_weight,
            )
        (loss / accumulation_steps).backward()
        if batch_index % accumulation_steps == 0 or batch_index == len(loader):
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), gradient_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _checkpoint_payload(
    model: SAM2AdapterMaskDecoder,
    *,
    epoch: int,
    best_validation_loss: float,
    args: argparse.Namespace,
    plan: H0DataPlan,
    checkpoint_sha256: str,
    optimizer: AdamW,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task": TASK_NAME,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": checkpoint_sha256,
        "sam2_config": args.sam2_config,
        "image_size": args.image_size,
        "manifest_hash": plan.manifest_hash,
        "adapter": model.adapter_metadata,
        "adaptation_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
    }


def _save_checkpoint(path: Path, model: SAM2AdapterMaskDecoder, **values: Any) -> None:
    torch.save(_checkpoint_payload(model, **values), path)


def _load_checkpoint(
    model: SAM2AdapterMaskDecoder,
    path: Path,
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != TASK_NAME:
        raise ValueError(f"checkpoint task mismatch: {path}")
    if payload.get("base_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("base SAM2 checkpoint hash does not match")
    load_trainable_state_dict(model, payload["adaptation_state"])
    return payload


def _write_metadata(
    layout: RunLayout,
    *,
    args: argparse.Namespace,
    plan: H0DataPlan,
    model: SAM2AdapterMaskDecoder,
    checkpoint_sha256: str,
    class_weighting: dict[str, object],
) -> None:
    counts = model_parameter_counts(model)
    write_json(layout.config / "args.json", vars(args))
    write_json(layout.config / "dataset.json", plan.record())
    write_json(
        layout.config / "model.json",
        {
            "model": "SAM2.1 Hiera-L SAM2-Adapter",
            "base_checkpoint": str(args.checkpoint),
            "base_checkpoint_sha256": checkpoint_sha256,
            "sam2_config": args.sam2_config,
            "adapter": model.adapter_metadata,
            "parameter_counts": counts,
            "trainable_scope": "four-stage visual adapters + active native mask decoder path",
            "frozen_scope": "original Hiera image encoder, FPN neck, prompt encoder, video memory",
            "prompt": "none (points=None, boxes=None, masks=None)",
        },
    )
    write_json(layout.config / "trainable_parameters.json", {"names": list(model.trainable_names), "count": counts["trainable"]})
    write_json(
        layout.config / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "git_revision": _git_revision(),
            "packages": _package_versions(),
        },
    )
    write_json(
        layout.config / "run.json",
        {
            "experiment_id": args.experiment_id,
            "task": TASK_NAME,
            "outer_fold": plan.outer_fold,
            "inner_fold": plan.inner_fold,
            "selection_metric": "validation fixed 2:1 foreground-weighted BCEWithLogits + 0.65 unweighted soft Dice",
            "selection_scope": "validation only; outer test excluded",
            "class_weighting": class_weighting,
            "threshold": FIXED_THRESHOLD,
            "threshold_source": "pre-registered merged dataset evaluation contract",
            "max_epochs": args.epochs,
            "early_stopping": False,
            "command": [sys.executable, *sys.argv],
        },
    )
    append_log(layout, f"created={datetime.now().astimezone().isoformat(timespec='seconds')}")
    append_log(layout, f"task={TASK_NAME} outer_fold={plan.outer_fold} inner_fold={plan.inner_fold}")


def _update_experiment_info(args: argparse.Namespace, plan: H0DataPlan, layout: RunLayout) -> None:
    root = PROJECT_ROOT / "runs" / args.experiment_id
    path = root / "info" / "experiment.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    folds = dict(existing.get("folds", {}))
    folds[TASK_NAME] = sorted(set(folds.get(TASK_NAME, [])) | {plan.outer_fold})
    runs = sorted(set(existing.get("runs", [])) | {layout.root.relative_to(root).as_posix()})
    write_json(
        path,
        {
            "schema_version": 2,
            "experiment_id": args.experiment_id,
            "layout": "5fold",
            "fold_count": 5,
            "experts": [TASK_NAME],
            "folds": folds,
            "runs": runs,
            "dataset_root": str(plan.root),
            "manifest_hash": plan.manifest_hash,
            "task": "binary merged crack/craquelure foreground segmentation",
            "threshold": FIXED_THRESHOLD,
            "threshold_source": "pre-registered constant; no threshold search",
            "created_at": existing.get("created_at", datetime.now().astimezone().isoformat()),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _monitor(args: argparse.Namespace, message: str) -> None:
    path = PROJECT_ROOT / "runs" / args.experiment_id / "info" / "TRAINING_LOG.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {datetime.now().astimezone().isoformat(timespec='seconds')} — {message}\n")


def _train_fold(
    args: argparse.Namespace,
    *,
    plan: H0DataPlan,
    device: torch.device,
    checkpoint_sha256: str,
) -> Path:
    output = _run_dir(args.experiment_id, plan.outer_fold)
    if output.exists() and any(output.iterdir()):
        best = output / "artifacts" / "checkpoints" / "best.pt"
        if args.allow_existing and best.is_file() and "completed" in (output / "logs" / "train.log").read_text(encoding="utf-8"):
            return best
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    layout = RunLayout.create(output)
    model = _build_model(args, device)
    class_weighting = binary_weighting_record(plan.train_counts[:2])
    positive_weight = float(class_weighting["bce_positive_weight"])
    _write_metadata(
        layout,
        args=args,
        plan=plan,
        model=model,
        checkpoint_sha256=checkpoint_sha256,
        class_weighting=class_weighting,
    )
    _update_experiment_info(args, plan, layout)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = _make_loader(plan, plan.train, train=True, args=args)
    validation_loader = _make_loader(plan, plan.val, train=False, args=args)
    outer_loader = _make_loader(plan, plan.test, train=False, args=args)
    writer = SummaryWriter(log_dir=str(layout.tensorboard))
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    best_loss = float("inf")
    best_epoch = 0
    finalized = False
    torch.cuda.reset_peak_memory_stats(device)
    _monitor(args, f"START fold{plan.outer_fold} {TASK_NAME}")
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch(
                model,
                train_loader,
                optimizer,
                plan=plan,
                device=device,
                amp=args.amp,
                dice_weight=args.dice_weight,
                positive_weight=positive_weight,
                accumulation_steps=args.accumulation_steps,
                gradient_clip=args.gradient_clip,
            )
            validation = _evaluate(
                model,
                validation_loader,
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
            values = dict(
                epoch=epoch,
                best_validation_loss=best_loss,
                args=args,
                plan=plan,
                checkpoint_sha256=checkpoint_sha256,
                optimizer=optimizer,
            )
            _save_checkpoint(layout.checkpoints / "last.pt", model, **values)
            if float(validation["loss"]) < best_loss:
                best_loss = float(validation["loss"])
                best_epoch = epoch
                values["best_validation_loss"] = best_loss
                _save_checkpoint(layout.checkpoints / "best.pt", model, **values)
            message = (
                f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} "
                f"val_loss={validation['loss']:.6f} val_f1={micro['mf1'] or 0:.4f} "
                f"best_epoch={best_epoch} lr={learning_rate:.8f}"
            )
            print(f"fold{plan.outer_fold} {TASK_NAME} {message}", flush=True)
            append_log(layout, message)
            scheduler.step()
        selected = _load_checkpoint(
            model,
            layout.checkpoints / "best.pt",
            checkpoint_sha256=checkpoint_sha256,
        )
        validation_selected = _evaluate(
            model,
            validation_loader,
            plan=plan,
            device=device,
            amp=args.amp,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
            layout=layout,
        )
        outer = _evaluate(
            model,
            outer_loader,
            plan=plan,
            device=device,
            amp=args.amp,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
        )
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        outer_record = {
            "scope": "outer_test_not_used_for_checkpoint_selection",
            "task": TASK_NAME,
            "selected_checkpoint": "artifacts/checkpoints/best.pt",
            "selected_epoch": int(selected["epoch"]),
            "selection_metric": "validation fixed 2:1 foreground-weighted BCEWithLogits + 0.65 unweighted soft Dice",
            "threshold": FIXED_THRESHOLD,
            "threshold_source": "pre-registered merged dataset evaluation contract",
            "loss": outer["loss"],
            "tile_micro": outer["tile_micro"],
            "expert_panel_macro": outer["expert_panel_macro"],
            "peak_allocated_mib": peak_mib,
        }
        write_json(
            layout.metrics / "selected_validation_metrics.json",
            {key: value for key, value in validation_selected.items() if key != "per_image_rows"},
        )
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
        _monitor(args, f"COMPLETE fold{plan.outer_fold} {TASK_NAME}: best_epoch={best_epoch}")
        return layout.checkpoints / "best.pt"
    finally:
        if not finalized:
            reporter.close()
            writer.close()
        del model
        torch.cuda.empty_cache()


def _smoke_test(args: argparse.Namespace, plan: H0DataPlan, device: torch.device) -> None:
    model = _build_model(args, device)
    loader = _make_loader(plan, plan.train, train=True, args=args)
    batch = next(iter(loader))
    images = _batch_tensor(batch, "image", device)
    source = _batch_tensor(batch, "mask", device).long()
    target = make_binary_target(source, ignore_value=plan.ignore_value)
    class_weighting = binary_weighting_record(plan.train_counts[:2])
    positive_weight = float(class_weighting["bce_positive_weight"])
    torch.cuda.reset_peak_memory_stats(device)
    with _autocast(device, args.amp):
        loss = binary_bce_dice_loss(
            model(images),
            target,
            ignore_value=plan.ignore_value,
            dice_weight=args.dice_weight,
            positive_weight=positive_weight,
        )
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    if missing:
        raise RuntimeError(f"trainable parameters without gradients: {missing[:10]}")
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "batch_size": args.batch_size,
                "loss": float(loss.detach().cpu()),
                "parameter_counts": model_parameter_counts(model),
                "adapter": model.adapter_metadata,
                "class_weighting": class_weighting,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            },
            indent=2,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.dataset.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("dataset or SAM2 checkpoint is missing")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SAM2-Adapter training requires CUDA")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    checkpoint_sha256 = _sha256(args.checkpoint)
    first_plan = prepare_data_plan(args.dataset, outer_fold=args.folds[0])
    if args.smoke_test:
        _smoke_test(args, first_plan, device)
        return 0
    print(
        f"SAM2-Adapter start folds={args.folds} task={TASK_NAME} experiment={args.experiment_id}",
        flush=True,
    )
    _monitor(args, f"EXPERIMENT START folds={args.folds} task={TASK_NAME}")
    for fold in args.folds:
        plan = prepare_data_plan(args.dataset, outer_fold=fold)
        _train_fold(
            args,
            plan=plan,
            device=device,
            checkpoint_sha256=checkpoint_sha256,
        )
    _monitor(args, "EXPERIMENT COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
