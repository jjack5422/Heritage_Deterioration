"""Train one locked SAM2-Adapter deterioration expert at 1008px or 1024px."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sam2_adapter.adapter_model import SAM2AdapterMaskDecoder
from sam2_adapter.expert_training_data import (
    EXPERT_RAW_IDS,
    ExpertTileDataset,
    LockedExpertDataPlan,
    denormalize_image,
    prepare_locked_expert_data_plan,
)
from sam2_adapter.h0_core import load_trainable_state_dict, model_parameter_counts, trainable_state_dict
from sam2_adapter.metrics import binary_summary, pixel_accuracy
from sam2_adapter.reporting import (
    EpochReporter,
    RunLayout,
    append_log,
    binary_metric_row,
    finalize_reporting,
    save_qualitative_example,
    write_json,
)
from sam2_adapter.runtime import _autocast, _batch_tensor, _git_revision, _package_versions, _seed_everything, _sha256
from sam3_adapter.losses import per_image_weighted_bce_dice_loss, weighted_bce_dice_loss

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MANIFEST_DIR = WORKSPACE_ROOT / "outputs" / "deterioration_statistics" / "sam3_experts"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_EXPERIMENT = "2026-09-20_three-experts_sam2-adapter-1008_50ep_locked-splits_seed42"
EXPERT_POSITIVE_WEIGHTS = {
    "scratch_crack": 1.0,
    "shrinkage_craquelure": 2.0,
    "loss": 2.0,
}
EXPERT_DICE_REDUCTIONS = {
    "scratch_crack": "batch_global",
    "shrinkage_craquelure": "batch_global",
    "loss": "positive_image_mean",
}
EXPERT_BATCH_CONTRACTS = {
    "scratch_crack": (4, 1),
    "shrinkage_craquelure": (2, 2),
    "loss": (4, 1),
}
MODEL_INPUT_SIZE = 1008
SUPPORTED_MODEL_INPUT_SIZES = (1008, 1024)
BACKBONE_INPUT_SIZE = 1024
METRIC_SIZE = 512
IMAGE_PIXELS = METRIC_SIZE * METRIC_SIZE
THRESHOLD = 0.5
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=tuple(EXPERT_RAW_IDS), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--model-input-size", type=int, default=MODEL_INPUT_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dice-weight", type=float, default=0.65)
    parser.add_argument("--positive-weight", type=float)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--scale-factor", type=int, default=32)
    parser.add_argument("--highpass-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-data-only", action="store_true")
    args = parser.parse_args(argv)
    args.manifest = (args.manifest or DEFAULT_MANIFEST_DIR / f"{args.expert}.json").resolve()
    args.checkpoint = args.checkpoint.resolve()
    expected_batch, expected_accumulation = EXPERT_BATCH_CONTRACTS[args.expert]
    args.batch_size = expected_batch if args.batch_size is None else args.batch_size
    args.accumulation_steps = expected_accumulation if args.accumulation_steps is None else args.accumulation_steps
    args.positive_weight = (
        EXPERT_POSITIVE_WEIGHTS[args.expert] if args.positive_weight is None else args.positive_weight
    )
    args.dice_reduction = EXPERT_DICE_REDUCTIONS[args.expert]
    args.loss_function = "bce_with_logits_plus_soft_dice"
    args.empty_target_policy = (
        "exclude_from_dice_include_in_bce"
        if args.dice_reduction == "positive_image_mean"
        else "include_in_batch_global_dice"
    )
    if args.model_input_size not in SUPPORTED_MODEL_INPUT_SIZES:
        parser.error(
            "model input size must be one of "
            + ", ".join(f"{size}x{size}" for size in SUPPORTED_MODEL_INPUT_SIZES)
        )
    if args.epochs != 50:
        parser.error("the approved comparison is locked to 50 epochs")
    if (args.batch_size, args.accumulation_steps) != (expected_batch, expected_accumulation):
        parser.error(
            f"batch contract for {args.expert} is locked to batch_size={expected_batch}, "
            f"accumulation_steps={expected_accumulation}"
        )
    if args.positive_weight != EXPERT_POSITIVE_WEIGHTS[args.expert] or args.dice_weight != 0.65:
        parser.error(
            f"loss contract for {args.expert} is locked to "
            f"positive_weight={EXPERT_POSITIVE_WEIGHTS[args.expert]} and dice_weight=0.65"
        )
    if args.num_workers < 0 or args.scale_factor <= 0 or not 0.0 < args.highpass_rate < 1.0:
        parser.error("invalid worker or adapter configuration")
    return args


class SAM2ExpertModel(nn.Module):
    """Present a 512px interface at the selected SAM2 Hiera input size."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.adapter = SAM2AdapterMaskDecoder(
            checkpoint=args.checkpoint,
            config=args.sam2_config,
            image_size=BACKBONE_INPUT_SIZE,
            device=device,
            scale_factor=args.scale_factor,
            highpass_rate=args.highpass_rate,
        )
        self.model_input_size = args.model_input_size
        self.backbone_padding = (BACKBONE_INPUT_SIZE - self.model_input_size) // 2
        self.register_buffer("image_mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, METRIC_SIZE, METRIC_SIZE):
            raise ValueError(f"source images must be Bx3x512x512, got {tuple(images.shape)}")
        rgb = (images.float() * self.image_std + self.image_mean).clamp(0.0, 1.0)
        resized = F.interpolate(
            rgb,
            size=(self.model_input_size, self.model_input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        prepared = (resized - self.image_mean) / self.image_std
        if self.backbone_padding:
            prepared = F.pad(
                prepared,
                (self.backbone_padding,) * 4,
                mode="reflect",
            )
        logits = self.adapter(prepared)
        content_logits = (
            logits[
                ...,
                self.backbone_padding : -self.backbone_padding,
                self.backbone_padding : -self.backbone_padding,
            ]
            if self.backbone_padding
            else logits
        )
        return F.interpolate(
            content_logits.float(),
            size=(METRIC_SIZE, METRIC_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    @property
    def adapter_metadata(self) -> dict[str, Any]:
        return self.adapter.adapter_metadata


def _segmentation_loss(logits: Tensor, target: Tensor, args: argparse.Namespace) -> Tensor:
    loss_function = (
        per_image_weighted_bce_dice_loss
        if args.dice_reduction == "positive_image_mean"
        else weighted_bce_dice_loss
    )
    return loss_function(
        logits,
        target,
        ignore_value=255,
        positive_weight=args.positive_weight,
        dice_weight=args.dice_weight,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _loader(
    rows: Sequence[Mapping[str, str]],
    *,
    raw_ids: Sequence[int],
    train: bool,
    args: argparse.Namespace,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + (1 if train else 0))
    workers = min(args.num_workers, os.cpu_count() or 1)
    return DataLoader(
        ExpertTileDataset(rows, raw_ids=raw_ids, train_augmentation=train),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers else None,
        generator=generator,
    )


def _build_model(args: argparse.Namespace, device: torch.device) -> SAM2ExpertModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        return SAM2ExpertModel(args, device).to(device)


def _trainable_initialization_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(trainable_state_dict(model).items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _counts(logits: Tensor, target: Tensor) -> tuple[int, int, int]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    valid = target != 255
    prediction = (torch.sigmoid(logits) >= THRESHOLD) & valid
    actual = (target == 1) & valid
    return tuple(
        int(value.item())
        for value in ((prediction & actual).sum(), (prediction & ~actual).sum(), (~prediction & actual).sum())
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    plan: LockedExpertDataPlan,
    args: argparse.Namespace,
    device: torch.device,
    layout: RunLayout | None = None,
    *,
    collect_rows: bool = False,
    split: str = "validation",
) -> dict[str, Any]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    source_counts: dict[str, tuple[int, int, int]] = {}
    dataset_counts: dict[str, tuple[int, int, int]] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = _batch_tensor(batch, "image", device)
            target = _batch_tensor(batch, "target", device).long()
            with _autocast(device, args.amp):
                logits = model(images)
                loss = _segmentation_loss(logits, target, args)
            loss_total += float(loss.cpu()) * images.shape[0]
            sample_count += images.shape[0]
            for index, source_group in enumerate(batch["source_group"]):
                update = _counts(logits[index : index + 1], target[index : index + 1])
                dataset_key = str(batch["dataset"][index])
                source_key = f"{dataset_key}:{source_group}"
                previous = source_counts.get(source_key, (0, 0, 0))
                source_counts[source_key] = tuple(
                    left + right for left, right in zip(previous, update, strict=True)
                )
                previous = dataset_counts.get(dataset_key, (0, 0, 0))
                dataset_counts[dataset_key] = tuple(
                    left + right for left, right in zip(previous, update, strict=True)
                )
                target_array = (target[index] == 1).cpu().numpy()
                prediction_array = (torch.sigmoid(logits[index, 0]) >= THRESHOLD).cpu().numpy()
                if layout is not None:
                    rgb = denormalize_image(images[index]).permute(1, 2, 0).mul(255).round().byte().numpy()
                    row = save_qualitative_example(
                        layout,
                        image_id=str(batch["name"][index]),
                        input_rgb=rgb,
                        target=target_array,
                        prediction=prediction_array,
                        target_class=plan.expert,
                    )
                    row.update({"dataset": dataset_key, "source_group": str(source_group)})
                    rows.append(row)
                elif collect_rows:
                    row = binary_metric_row(target_array, prediction_array)
                    row.update(
                        {
                            "image": str(batch["name"][index]),
                            "split": split,
                            "target_class": plan.expert,
                            "dataset": dataset_key,
                            "source_group": str(source_group),
                        }
                    )
                    rows.append(row)
    if not sample_count:
        raise RuntimeError("evaluation loader is empty")
    summary = binary_summary(source_counts)
    return {
        "loss": loss_total / sample_count,
        **summary,
        "by_dataset": {
            dataset: binary_summary({dataset: counts})["tile_micro"]
            for dataset, counts in sorted(dataset_counts.items())
        },
        "by_source": {
            source: binary_summary({source: counts})["tile_micro"]
            for source, counts in sorted(source_counts.items())
        },
        "per_image_rows": rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty metrics CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader, 1):
        images = _batch_tensor(batch, "image", device)
        target = _batch_tensor(batch, "target", device).long()
        with _autocast(device, args.amp):
            loss = _segmentation_loss(model(images), target, args)
        (loss / args.accumulation_steps).backward()
        if batch_index % args.accumulation_steps == 0 or batch_index == len(loader):
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.gradient_clip,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_total += float(loss.detach().cpu()) * images.shape[0]
        sample_count += images.shape[0]
    if not sample_count:
        raise RuntimeError("training loader is empty")
    return loss_total / sample_count


def _run_root(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / "runs" / args.experiment_id / "1fold" / args.expert / "fold0"


def _write_metadata(
    layout: RunLayout,
    model: SAM2ExpertModel,
    plan: LockedExpertDataPlan,
    args: argparse.Namespace,
    checkpoint_hash: str,
) -> None:
    write_json(layout.config / "args.json", vars(args))
    write_json(layout.config / "dataset.json", plan.record())
    write_json(
        layout.config / "model.json",
        {
            "expert": args.expert,
            "backbone": "SAM2.1 Hiera-L",
            "model_input_size": args.model_input_size,
            "backbone_input_size": BACKBONE_INPUT_SIZE,
            "backbone_padding": (
                f"reflect; {model.backbone_padding}px each side after "
                f"{args.model_input_size}px resize"
                if model.backbone_padding
                else "none"
            ),
            "source_and_metric_size": METRIC_SIZE,
            "source_rgb_resize": "bicubic; align_corners=false; antialias=true; clamp=[0,1]",
            "target_mask_resize": "none",
            "logit_resize_to_metric_size": "bilinear; align_corners=false",
            "decoder": "native SAM2 mask decoder",
            "parameter_counts": model_parameter_counts(model),
            "base_checkpoint": str(args.checkpoint),
            "base_checkpoint_sha256": checkpoint_hash,
            "trainable_scope": "stage-aware visual adapters + active native SAM2 mask decoder path",
            "trainable_names": list(model.trainable_names),
            "trainable_initialization_sha256": _trainable_initialization_sha256(model),
            "adapter_metadata": model.adapter_metadata,
        },
    )
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
            "expert": args.expert,
            "fold": 0,
            "model_input_size": args.model_input_size,
            "backbone_input_size": BACKBONE_INPUT_SIZE,
            "backbone_padding": (
                f"reflect; {model.backbone_padding}px each side"
                if model.backbone_padding
                else "none"
            ),
            "selection_metric": "maximum validation pixel-micro F1 at threshold 0.5; lower validation loss breaks ties",
            "selection_scope": "locked validation only; locked test evaluated once after checkpoint selection",
            "threshold": THRESHOLD,
            "epochs": args.epochs,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "command": [sys.executable, *sys.argv],
        },
    )
    append_log(layout, f"created={datetime.now().astimezone().isoformat(timespec='seconds')}")


def _save(
    path: Path,
    model: nn.Module,
    epoch: int,
    best_f1: float,
    best_loss: float,
    checkpoint_hash: str,
    plan: LockedExpertDataPlan,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "expert": plan.expert,
            "foreground_raw_ids": plan.raw_ids,
            "epoch": epoch,
            "best_validation_f1": best_f1,
            "best_validation_loss_at_best_f1": best_loss,
            "selection_metric": "validation_pixel_micro_f1",
            "selection_threshold": THRESHOLD,
            "base_checkpoint_sha256": checkpoint_hash,
            "dataset_sha256": plan.adopted_dataset_sha256,
            "split_sha256": plan.split_sha256,
            "adaptation_state": trainable_state_dict(model),
        },
        path,
    )


def _checkpoint_improved(
    *,
    validation_f1: float,
    validation_loss: float,
    best_f1: float,
    best_loss: float,
) -> bool:
    return validation_f1 > best_f1 or (
        validation_f1 == best_f1 and validation_loss < best_loss
    )


def _update_experiment_info(args: argparse.Namespace, plan: LockedExpertDataPlan) -> None:
    info = PROJECT_ROOT / "runs" / args.experiment_id / "info"
    experiment_path = info / "experiment.json"
    recorded_experts: set[str] = set()
    created_at = datetime.now().astimezone().isoformat()
    if experiment_path.is_file():
        existing = json.loads(experiment_path.read_text(encoding="utf-8"))
        if existing.get("experiment_id") != args.experiment_id:
            raise ValueError(f"experiment metadata mismatch: {experiment_path}")
        recorded_experts.update(str(expert) for expert in existing.get("experts", ()))
        if existing.get("model_input_size") != args.model_input_size:
            raise ValueError(
                f"experiment model input size mismatch: {experiment_path}"
            )
        created_at = str(existing.get("created_at", created_at))
    recorded_experts.add(args.expert)
    write_json(
        experiment_path,
        {
            "schema_version": 2,
            "experiment_id": args.experiment_id,
            "experts": [expert for expert in EXPERT_RAW_IDS if expert in recorded_experts],
            "fold_count": 1,
            "model": "SAM2.1 Hiera-L Adapter",
            "model_input_size": args.model_input_size,
            "backbone_input_size": BACKBONE_INPUT_SIZE,
            "epochs": 50,
            "selection_boundary": "validation-only checkpoint selection; test evaluated once afterward",
            "created_at": created_at,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )
    write_json(info / f"{args.expert}_dataset_contract.json", plan.record())


def train(
    args: argparse.Namespace,
    plan: LockedExpertDataPlan,
    device: torch.device,
    checkpoint_hash: str,
) -> Path:
    output = _run_root(args)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    layout = RunLayout.create(output)
    model = _build_model(args, device)
    _write_metadata(layout, model, plan, args, checkpoint_hash)
    _update_experiment_info(args, plan)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    train_loader = _loader(plan.train, raw_ids=plan.raw_ids, train=True, args=args)
    validation_loader = _loader(plan.validation, raw_ids=plan.raw_ids, train=False, args=args)
    test_loader = _loader(plan.test, raw_ids=plan.raw_ids, train=False, args=args)
    print(
        f"{args.expert} training_started train={len(plan.train)} "
        f"validation={len(plan.validation)} test={len(plan.test)} excluded={len(plan.excluded)}",
        flush=True,
    )
    writer = SummaryWriter(log_dir=str(layout.tensorboard))
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    best_f1, best_loss, best_epoch, finalized = float("-inf"), float("inf"), 0, False
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch(model, train_loader, optimizer, args, device)
            validation = _evaluate(model, validation_loader, plan, args, device)
            micro = validation["tile_micro"]
            accuracy = pixel_accuracy(
                int(micro["tp"]), int(micro["fp"]), int(micro["fn"]), len(plan.validation) * IMAGE_PIXELS
            )
            reporter.record(
                epoch,
                train_loss=train_loss,
                validation_loss=validation["loss"],
                f1=micro["mf1"],
                precision=micro["mprecision"],
                recall=micro["mrecall"],
                iou=micro["miou"],
                accuracy=accuracy,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            current_f1 = float(micro["mf1"] or 0.0)
            current_loss = float(validation["loss"])
            if _checkpoint_improved(
                validation_f1=current_f1,
                validation_loss=current_loss,
                best_f1=best_f1,
                best_loss=best_loss,
            ):
                best_f1, best_loss, best_epoch = current_f1, current_loss, epoch
                _save(layout.checkpoints / "best.pt", model, epoch, best_f1, best_loss, checkpoint_hash, plan)
            _save(layout.checkpoints / "last.pt", model, epoch, best_f1, best_loss, checkpoint_hash, plan)
            message = (
                f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} "
                f"val_loss={current_loss:.6f} val_f1={current_f1:.4f} "
                f"best_f1={best_f1:.4f} best_epoch={best_epoch}"
            )
            print(f"{args.expert} {message}", flush=True)
            append_log(layout, message)
            scheduler.step()

        payload = torch.load(layout.checkpoints / "best.pt", map_location="cpu", weights_only=False)
        load_trainable_state_dict(model, payload["adaptation_state"])
        selected = _evaluate(model, validation_loader, plan, args, device, layout)
        selected_metrics = {key: value for key, value in selected.items() if key != "per_image_rows"}
        write_json(layout.metrics / "selected_validation_metrics.json", selected_metrics)

        outer = _evaluate(model, test_loader, plan, args, device, collect_rows=True, split="test")
        outer_rows = outer.pop("per_image_rows")
        _write_csv(layout.metrics / "outer_test_per_image.csv", outer_rows)
        source_rows = [
            {"source_group": source, **metrics}
            for source, metrics in outer["by_source"].items()
        ]
        _write_csv(layout.metrics / "outer_test_per_source.csv", source_rows)
        outer_record = {
            "scope": "outer_test_not_used_for_checkpoint_or_threshold_selection",
            "selected_checkpoint": "artifacts/checkpoints/best.pt",
            "selected_epoch": best_epoch,
            "threshold": THRESHOLD,
            **outer,
            "per_image_metrics": "metrics/outer_test_per_image.csv",
            "per_source_metrics": "metrics/outer_test_per_source.csv",
        }
        write_json(
            layout.metrics / "experiment_summary.json",
            {
                "status": "completed",
                "expert": args.expert,
                "fold": 0,
                "selected_epoch": best_epoch,
                "best_validation_f1": best_f1,
                "validation_loss_at_best_f1": best_loss,
                "split_sha256": plan.split_sha256,
                "adopted_dataset_sha256": plan.adopted_dataset_sha256,
                "outer_test_excluded_from_selection": True,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            },
        )
        finalize_reporting(
            layout,
            writer=writer,
            reporter=reporter,
            validation_rows=selected["per_image_rows"],
            selected_epoch=best_epoch,
            outer_test_metrics=outer_record,
        )
        finalized = True
        append_log(layout, f"completed selected_epoch={best_epoch}")
    finally:
        if not finalized:
            reporter.close()
            writer.close()
        del model
        torch.cuda.empty_cache()
    return output


def smoke_test(
    args: argparse.Namespace,
    plan: LockedExpertDataPlan,
    device: torch.device,
) -> None:
    model = _build_model(args, device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batch = next(iter(_loader(plan.train, raw_ids=plan.raw_ids, train=True, args=args)))
    images = _batch_tensor(batch, "image", device)
    target = _batch_tensor(batch, "target", device).long()
    torch.cuda.reset_peak_memory_stats(device)
    with _autocast(device, args.amp):
        logits = model(images)
        loss = _segmentation_loss(logits, target, args)
    if logits.shape != (args.batch_size, 1, METRIC_SIZE, METRIC_SIZE) or not torch.isfinite(loss):
        raise RuntimeError(f"invalid smoke output: logits={tuple(logits.shape)}, loss={loss}")
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"trainable parameters without gradients: {missing[:10]}")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        args.gradient_clip,
    )
    if not torch.isfinite(gradient_norm):
        raise RuntimeError(f"non-finite gradient norm: {gradient_norm}")
    optimizer.step()
    print(
        json.dumps(
            {
                "status": "passed",
                "expert": args.expert,
                "model_input_size": args.model_input_size,
                "loss": float(loss.detach().cpu()),
                "logits_shape": list(logits.shape),
                "parameter_counts": model_parameter_counts(model),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            },
            indent=2,
        ),
        flush=True,
    )
    del optimizer, model
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = prepare_locked_expert_data_plan(args.manifest, args.expert)
    if args.validate_data_only:
        print(json.dumps(plan.record(), ensure_ascii=False, indent=2))
        return 0
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"base checkpoint is missing: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2-Adapter expert training requires CUDA")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    if args.smoke_test:
        smoke_test(args, plan, device)
        return 0
    checkpoint_hash = _sha256(args.checkpoint)
    train(args, plan, device, checkpoint_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
