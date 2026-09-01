"""Train A or B with the locked common prompt-free probe protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sam2_adapter.adapter_core import make_binary_target
from sam2_adapter.data import H0DataPlan, denormalize_image, prepare_data_plan
from sam2_adapter.h0_core import load_trainable_state_dict, model_parameter_counts, trainable_state_dict
from sam2_adapter.metrics import binary_summary, pixel_accuracy
from sam2_adapter.reporting import EpochReporter, RunLayout, append_log, finalize_reporting, save_qualitative_example, write_json
from sam2_adapter.runtime import _autocast, _batch_tensor, _git_revision, _make_loader, _package_versions, _seed_everything, _sha256
from sam3_adapter.losses import weighted_bce_dice_loss
from sam3_adapter.probe_models import Sam2ProbeModel, Sam3ProbeModel
from sam3_adapter.sam3_adapter_model import Sam3AdapterModel


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET = WORKSPACE_ROOT / "datasets" / "dataset_clean_v2_merged_craquelure"
CHECKPOINTS = {
    "sam2_probe": WORKSPACE_ROOT / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_large.pt",
    "sam3_probe": WORKSPACE_ROOT / "segment-anything-3" / "checkpoints" / "sam3.pt",
    "sam3_adapter": WORKSPACE_ROOT / "segment-anything-3" / "checkpoints" / "sam3.pt",
}
THRESHOLD = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dice-weight", type=float, default=0.65)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", default="2026-08-26_sam2-sam3-native-probe-adapter_seed42")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-input-size", type=int, choices=(512, 1008), default=None,
                        help="SAM3-Adapter input size; default 1008. Only affects sam3_adapter.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint = (args.checkpoint or CHECKPOINTS[args.group]).resolve()
    if args.model_input_size is None:
        args.model_input_size = 1008
    if args.group != "sam3_adapter" and args.model_input_size != 1008:
        parser.error("--model-input-size=512 is only supported for sam3_adapter")
    if args.batch_size * args.accumulation_steps != 4:
        parser.error("effective batch size must equal 4")
    if args.positive_weight != 2.0 or args.dice_weight != 0.65:
        parser.error("loss contract is locked to positive_weight=2.0 and dice_weight=0.65")
    if any(fold not in range(5) for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        parser.error("folds must be unique values from 0 through 4")
    return args


def _build(args: argparse.Namespace, device: torch.device, *, fold: int) -> nn.Module:
    cls = {"sam2_probe": Sam2ProbeModel, "sam3_probe": Sam3ProbeModel, "sam3_adapter": Sam3AdapterModel}[args.group]
    if args.group == "sam3_adapter":
        # Adapter layers and task-specific decoder parameters absent from the
        # official checkpoint are newly initialized. Make that initialization
        # depend only on the approved base seed and fold, not on the order in
        # which folds happen to run in this process.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + fold)
            return cls(args.checkpoint, device=device, input_size=args.model_input_size)
    return cls(args.checkpoint, device=device, probe_seed=args.seed + fold)


def _trainable_initialization_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(trainable_state_dict(model).items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _counts(logits: Tensor, target: Tensor, ignore: int) -> tuple[int, int, int]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    valid = target != ignore
    pred = torch.sigmoid(logits) >= THRESHOLD
    actual = target == 1
    return tuple(int(value.item()) for value in ((pred & actual & valid).sum(), (pred & ~actual & valid).sum(), (~pred & actual & valid).sum()))


def _evaluate(model: nn.Module, loader: DataLoader, plan: H0DataPlan, args: argparse.Namespace, device: torch.device, layout: RunLayout | None = None) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    panels: dict[str, tuple[int, int, int]] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = _batch_tensor(batch, "image", device)
            target = make_binary_target(_batch_tensor(batch, "mask", device).long(), ignore_value=plan.ignore_value)
            with _autocast(device, args.amp):
                logits = model(images)
                loss = weighted_bce_dice_loss(logits, target, ignore_value=plan.ignore_value)
            losses.append(float(loss.cpu()))
            for index, group in enumerate(batch["source_group"]):
                update = _counts(logits[index:index + 1], target[index:index + 1], plan.ignore_value)
                previous = panels.get(str(group), (0, 0, 0))
                panels[str(group)] = tuple(a + b for a, b in zip(previous, update, strict=True))
                if layout is not None:
                    valid = target[index] != plan.ignore_value
                    rgb = denormalize_image(images[index]).permute(1, 2, 0).mul(255).round().byte().numpy()
                    rows.append(save_qualitative_example(
                        layout,
                        image_id=str(batch["name"][index]),
                        input_rgb=rgb,
                        target=((target[index] == 1) & valid).cpu().numpy(),
                        prediction=((torch.sigmoid(logits[index, 0]) >= THRESHOLD) & valid).cpu().numpy(),
                        target_class="foreground",
                    ))
    return {"loss": sum(losses) / len(losses), **binary_summary(panels), "per_image_rows": rows}


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: AdamW, plan: H0DataPlan, args: argparse.Namespace, device: torch.device) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for index, batch in enumerate(loader, 1):
        images = _batch_tensor(batch, "image", device)
        target = make_binary_target(_batch_tensor(batch, "mask", device).long(), ignore_value=plan.ignore_value)
        with _autocast(device, args.amp):
            loss = weighted_bce_dice_loss(model(images), target, ignore_value=plan.ignore_value)
        (loss / args.accumulation_steps).backward()
        if index % args.accumulation_steps == 0 or index == len(loader):
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.gradient_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _run_dir(args: argparse.Namespace, fold: int) -> Path:
    return PROJECT_ROOT / "runs" / args.experiment_id / "5fold" / args.group / f"fold{fold}"


def _metadata(layout: RunLayout, model: nn.Module, plan: H0DataPlan, args: argparse.Namespace, checkpoint_hash: str) -> None:
    counts = model_parameter_counts(model)
    write_json(layout.config / "args.json", vars(args))
    write_json(layout.config / "dataset.json", plan.record())
    write_json(layout.config / "model.json", {
        "group": args.group,
        "backbone": "SAM2.1 Hiera-L" if args.group == "sam2_probe" else "SAM3 ViT",
        "model_input_size": model.model_input_size,
        "source_and_metric_size": 512,
        "decoder": "SharedFpnProbe(width=128, three 256-channel levels)" if args.group != "sam3_adapter" else "author SAM-family mask decoder",
        "prompt": "none",
        "parameter_counts": counts,
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": checkpoint_hash,
        "trainable_scope": "probe only" if args.group != "sam3_adapter" else "official-author prompt_generator adapter + active SAM-family mask decoder path",
        "trainable_names": list(model.trainable_names),
        "trainable_initialization_sha256": _trainable_initialization_sha256(model),
        "probe_seed": getattr(model, "probe_seed", None),
    })
    write_json(layout.config / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "git_revision": _git_revision(), "packages": _package_versions(),
    })
    write_json(layout.config / "run.json", {
        "experiment_id": args.experiment_id, "group": args.group, "outer_fold": plan.outer_fold, "inner_fold": plan.inner_fold,
        "selection_metric": "validation weighted BCE(pos_weight=2.0) + 0.65 soft Dice",
        "selection_scope": "validation only; outer test excluded", "threshold": THRESHOLD,
        "epochs": args.epochs, "effective_batch_size": args.batch_size * args.accumulation_steps,
        "command": [sys.executable, *sys.argv],
    })
    append_log(layout, f"created={datetime.now().astimezone().isoformat(timespec='seconds')}")


def _save(path: Path, model: nn.Module, epoch: int, best_loss: float, checkpoint_hash: str) -> None:
    torch.save({"schema_version": 1, "epoch": epoch, "best_validation_loss": best_loss, "base_checkpoint_sha256": checkpoint_hash, "adaptation_state": trainable_state_dict(model)}, path)


def _train_fold(args: argparse.Namespace, plan: H0DataPlan, device: torch.device, checkpoint_hash: str) -> None:
    output = _run_dir(args, plan.outer_fold)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    layout = RunLayout.create(output)
    model = _build(args, device, fold=plan.outer_fold)
    _metadata(layout, model, plan, args, checkpoint_hash)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = _make_loader(plan, plan.train, train=True, args=args)
    val_loader = _make_loader(plan, plan.val, train=False, args=args)
    test_loader = _make_loader(plan, plan.test, train=False, args=args)
    writer = SummaryWriter(log_dir=str(layout.tensorboard))
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    best_loss, best_epoch, finalized = float("inf"), 0, False
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch(model, train_loader, optimizer, plan, args, device)
            validation = _evaluate(model, val_loader, plan, args, device)
            micro = validation["tile_micro"]
            accuracy = pixel_accuracy(
                int(micro["tp"]), int(micro["fp"]), int(micro["fn"]),
                plan.val_counts[0] + plan.val_counts[1],
            )
            reporter.record(epoch, train_loss=train_loss, validation_loss=validation["loss"], f1=micro["mf1"], precision=micro["mprecision"], recall=micro["mrecall"], iou=micro["miou"], accuracy=accuracy, learning_rate=optimizer.param_groups[0]["lr"])
            _save(layout.checkpoints / "last.pt", model, epoch, best_loss, checkpoint_hash)
            if validation["loss"] < best_loss:
                best_loss, best_epoch = float(validation["loss"]), epoch
                _save(layout.checkpoints / "best.pt", model, epoch, best_loss, checkpoint_hash)
            message = f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={validation['loss']:.6f} val_f1={micro['mf1'] or 0:.4f} best_epoch={best_epoch}"
            print(f"{args.group} fold{plan.outer_fold} {message}", flush=True)
            append_log(layout, message)
            scheduler.step()
        payload = torch.load(layout.checkpoints / "best.pt", map_location="cpu", weights_only=False)
        load_trainable_state_dict(model, payload["adaptation_state"])
        selected = _evaluate(model, val_loader, plan, args, device, layout)
        outer = _evaluate(model, test_loader, plan, args, device)
        outer_record = {"scope": "outer_test_not_used_for_checkpoint_selection", "selected_epoch": best_epoch, "threshold": THRESHOLD, "loss": outer["loss"], "tile_micro": outer["tile_micro"], "expert_panel_macro": outer["expert_panel_macro"]}
        write_json(layout.metrics / "selected_validation_metrics.json", {key: value for key, value in selected.items() if key != "per_image_rows"})
        write_json(layout.metrics / "experiment_summary.json", {"status": "completed", "group": args.group, "fold": plan.outer_fold, "selected_epoch": best_epoch, "best_validation_loss": best_loss, "outer_test_excluded_from_selection": True, "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2})
        finalize_reporting(layout, writer=writer, reporter=reporter, validation_rows=selected["per_image_rows"], selected_epoch=best_epoch, outer_test_metrics=outer_record)
        finalized = True
        append_log(layout, f"completed selected_epoch={best_epoch}")
    finally:
        if not finalized:
            reporter.close(); writer.close()
        del model
        torch.cuda.empty_cache()


def _smoke(args: argparse.Namespace, plan: H0DataPlan, device: torch.device) -> None:
    model = _build(args, device, fold=plan.outer_fold)
    loader = _make_loader(plan, plan.train, train=True, args=args)
    batch = next(iter(loader))
    images = _batch_tensor(batch, "image", device)
    target = make_binary_target(_batch_tensor(batch, "mask", device).long(), ignore_value=plan.ignore_value)
    torch.cuda.reset_peak_memory_stats(device)
    with _autocast(device, args.amp):
        loss = weighted_bce_dice_loss(model(images), target, ignore_value=plan.ignore_value)
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    if missing:
        raise RuntimeError(f"trainable parameters without gradients: {missing[:10]}")
    print(json.dumps({"status": "passed", "group": args.group, "loss": float(loss.detach().cpu()), "parameter_counts": model_parameter_counts(model), "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2}, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dataset = args.dataset.resolve()
    if not args.dataset.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("dataset or checkpoint is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("probe training requires CUDA")
    device = torch.device("cuda")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    plan = prepare_data_plan(args.dataset, outer_fold=args.folds[0])
    if args.smoke_test:
        _smoke(args, plan, device)
        return 0
    checkpoint_hash = _sha256(args.checkpoint)
    info = PROJECT_ROOT / "runs" / args.experiment_id / "info"
    write_json(info / "experiment.json", {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "groups": ["sam2_probe", "sam3_probe", "sam3_adapter"],
        "fold_count": 5,
        "dataset_root": str(args.dataset),
        "manifest_hash": plan.manifest_hash,
        "selection_boundary": "validation only; clean outer test evaluated once after selection",
        "created_at": datetime.now().astimezone().isoformat(),
    })
    write_json(info / "dataset_contract.json", plan.record())
    write_json(info / "comparison_contract.json", {
        "A_vs_B": "same prompt-free probe; native pretrained preprocessing/resolution",
        "A_model_input": 1024,
        "B_model_input": 1008,
        "source_and_metric_resolution": 512,
        "effective_batch_size": 4,
        "artificial_corruptions": False,
    })
    for fold in args.folds:
        _train_fold(args, prepare_data_plan(args.dataset, outer_fold=fold), device, checkpoint_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
