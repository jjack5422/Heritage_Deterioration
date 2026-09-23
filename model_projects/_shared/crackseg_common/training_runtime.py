"""Shared training lifecycle for crack/craquelure segmentation projects."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np
import torch
from crackseg_common.augment import train_transforms, val_transforms
from crackseg_common.dataset import TileSegDataset
from crackseg_common.losses import SegLoss
from crackseg_common.metrics import format_metrics
from crackseg_common.data_plan import (
    ExpertDataset,
    DataError,
    DataPlan,
    JointDataset,
    MergedForegroundDataset,
    class_weights,
    cost_weight_for_epoch,
    joint_class_weights,
    joint_sampling_weights,
    prepare_dataset,
)
from crackseg_common.evaluation import evaluate
from crackseg_common.reporting.outputs import (
    EpochReporter,
    RunLayout,
    build_training_dashboard,
    create_run_layout,
    create_tensorboard_writer,
    export_tensorboard_artifacts,
    log_message,
)
from crackseg_common.reporting.qualitative import (
    prepare_final_evaluation,
    prepare_fixed_binary_final_evaluation,
    prepare_joint_final_evaluation,
    write_tensorboard_validation_images,
)
from crackseg_common.training_engine import train_epoch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


@dataclass(frozen=True)
class TrainingProject:
    """Model-family hooks and artifact root for one independent project."""

    root: Path
    description: str
    add_model_arguments: Callable[[argparse.ArgumentParser], None]
    resolve_model_args: Callable[[argparse.Namespace], None]
    build_model: Callable[[argparse.Namespace, int], nn.Module]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    best: float,
    bad_epochs: int,
    args: argparse.Namespace,
    plan: DataPlan,
    val: dict[str, Any],
) -> None:
    legacy_args = {**vars(args), "class_names": ",".join(plan.class_names)}
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_val_panel_macro_loss": best,
            "epochs_without_improvement": bad_epochs,
            "selection_metric": "val_panel_macro_loss",
            "selection_mode": "min",
            "args": legacy_args,
            "class_names": list(plan.class_names),
            "source_class_names": list(plan.source_class_names),
            "expert": {"name": plan.expert_name, "source_class_id": plan.expert_id},
            "task": plan.record()["task"],
            "manifest_hash": plan.manifest_hash,
            "outer_fold": plan.outer_fold,
            "inner_fold": plan.inner_fold,
            "val": val,
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    plan: DataPlan,
) -> tuple[int, float, int]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if len(plan.class_names) == 3:
        if data.get("task") != plan.record()["task"]:
            raise DataError("checkpoint task 與目前 crack/craquelure task 不一致")
    else:
        expected_expert = {"name": plan.expert_name, "source_class_id": plan.expert_id}
        if data.get("expert") != expected_expert:
            raise DataError("checkpoint expert 與目前指定的劣化類別不一致")
    if data.get("manifest_hash") != plan.manifest_hash:
        raise DataError("checkpoint dataset manifest 不一致")
    if data.get("outer_fold") != plan.outer_fold or data.get("inner_fold") != plan.inner_fold:
        raise DataError("checkpoint split 不一致")
    if data.get("selection_metric") != "val_panel_macro_loss":
        raise DataError("checkpoint 使用舊版 selection metric，請建立新的 run")
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    if scaler is not None and data.get("scaler"):
        scaler.load_state_dict(data["scaler"])
    if scheduler is not None and data.get("scheduler"):
        scheduler.load_state_dict(data["scheduler"])
    return (
        int(data["epoch"]),
        float(data["best_val_panel_macro_loss"]),
        int(data.get("epochs_without_improvement", 0)),
    )


def json_safe(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(value) for value in data]
    if isinstance(data, (float, np.floating)) and not math.isfinite(float(data)):
        return None
    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def training_parser(project: TrainingProject) -> argparse.ArgumentParser:
    """Build the command-line parser for one model project."""

    result = argparse.ArgumentParser(description=project.description)
    result.add_argument("--dataset-root", required=True)
    result.add_argument(
        "--expert",
        required=True,
        help="劣化類別，或以 crack_craquelure 啟用三類聯合訓練",
    )
    result.add_argument("--outer-fold", type=int, default=0)
    result.add_argument(
        "--inner-fold",
        type=int,
        help="validation fold；預設使用 outer fold 的下一個 fold 循環輪替",
    )
    project.add_model_arguments(result)
    result.add_argument("--image-size", type=int, default=512)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--lr-factor", type=float, default=0.5)
    result.add_argument("--lr-patience", type=int, default=6)
    result.add_argument("--lr-threshold", type=float, default=0.005)
    result.add_argument("--lr-cooldown", type=int, default=2)
    result.add_argument("--min-lr", type=float, default=1e-6)
    result.add_argument("--min-encoder-lr", type=float, default=1e-7)
    result.add_argument("--early-stop-patience", type=int, default=None,
                        help="預設關閉；設定後才允許提早停止")
    result.add_argument("--ce-weight", type=float, default=0.4)
    result.add_argument("--dice-weight", type=float, default=0.4)
    result.add_argument("--cost-weight", type=float, default=0.2)
    result.add_argument("--craquelure-to-crack-cost", type=float, default=0.0)
    result.add_argument("--no-class-weights", action="store_true")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="auto")
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--output-dir")
    result.add_argument("--experiment-id", help="group all folds under runs/<id>/5fold")
    result.add_argument("--resume")
    result.add_argument("--skip-outer-test", action="store_true",
                        help="調參 run 不讀取 outer test")
    result.add_argument("--validate-only", action="store_true")
    return result


def make_loader(
    plan: DataPlan,
    names: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    *,
    training: bool,
) -> DataLoader:
    transforms = train_transforms(args.image_size) if training else val_transforms(args.image_size)
    source = TileSegDataset(plan.root, plan.items(names), transforms=transforms)
    joint = len(plan.class_names) == 3
    if joint:
        dataset = JointDataset(source, *plan.source_class_ids, plan.ignore_value)
    elif plan.expert_name == "foreground":
        dataset = MergedForegroundDataset(source, plan.expert_id, plan.ignore_value)
    else:
        dataset = ExpertDataset(source, plan.expert_id, plan.ignore_value)
    sampler = None
    if training and joint:
        sampler = WeightedRandomSampler(
            joint_sampling_weights(plan, names),
            num_samples=len(names),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed + plan.outer_fold),
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    groups = param_groups(model, lr, encoder_lr_mult)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def param_groups(
    model: nn.Module,
    base_lr: float,
    encoder_lr_mult: float = 0.1,
) -> list[dict[str, Any]]:
    """Split pretrained encoder and decoder parameters into LR groups."""

    encoder_parameters, decoder_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = encoder_parameters if name.startswith("encoder.") else decoder_parameters
        target.append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": decoder_parameters, "lr": base_lr, "name": "decoder"}
    ]
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": base_lr * encoder_lr_mult,
                "name": "encoder",
            }
        )
    return groups


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
    patience: int,
    threshold: float,
    cooldown: int,
    min_lr: float,
    min_encoder_lr: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    floors = {"decoder": min_lr, "encoder": min_encoder_lr}
    group_floors = [
        floors.get(group.get("name", "decoder"), min_lr)
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode="rel",
        cooldown=cooldown,
        min_lr=group_floors,
    )


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


@dataclass
class TrainingRun:
    """Own the mutable state and lifecycle of one expert training run."""

    args: argparse.Namespace
    plan: DataPlan
    layout: RunLayout
    device: torch.device
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    model: nn.Module
    criterion: SegLoss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    scaler: torch.amp.GradScaler | None

    @classmethod
    def create(
        cls,
        args: argparse.Namespace,
        plan: DataPlan,
        layout: RunLayout,
        project: TrainingProject,
    ) -> TrainingRun:
        seed_everything(args.seed)
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("要求 CUDA，但 CUDA 不可用")
        train_loader = make_loader(plan, plan.train, args, device, training=True)
        val_loader = make_loader(plan, plan.val, args, device, training=False)
        test_loader = make_loader(plan, plan.test, args, device, training=False)
        model = project.build_model(args, len(plan.class_names)).to(device)
        total, trainable = count_params(model)
        log_message(layout, f"{args.architecture} params={total / 1e6:.2f}M "
                    f"trainable={trainable / 1e6:.2f}M")
        weights = None
        if not args.no_class_weights:
            calculate = joint_class_weights if len(plan.class_names) == 3 else class_weights
            weights = calculate(plan.train_counts).to(device)
        criterion = SegLoss(
            len(plan.class_names),
            plan.ignore_value,
            weights,
            args.ce_weight,
            args.dice_weight,
            0.0,
            args.craquelure_to_crack_cost,
        ).to(device)
        optimizer = build_optimizer(
            model,
            lr=args.lr,
            encoder_lr_mult=args.encoder_lr_mult,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer,
            factor=args.lr_factor,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            cooldown=args.lr_cooldown,
            min_lr=args.min_lr,
            min_encoder_lr=args.min_encoder_lr,
        )
        log_message(layout, f"learning rates decoder={args.lr:.2e} encoder={args.lr * args.encoder_lr_mult:.2e}")
        log_message(
            layout,
            f"batch micro={args.batch_size} "
            f"accumulation={args.gradient_accumulation_steps} "
            f"effective={args.batch_size * args.gradient_accumulation_steps}",
        )
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None
        return cls(
            args,
            plan,
            layout,
            device,
            train_loader,
            val_loader,
            test_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
        )

    def execute(self) -> None:
        start_epoch, best, bad_epochs = 0, math.inf, 0
        resume_path = Path(self.args.resume) if self.args.resume else None
        if resume_path:
            start_epoch, best, bad_epochs = restore_checkpoint(
                resume_path,
                self.model,
                self.optimizer,
                self.scaler,
                self.scheduler,
                self.plan,
            )
        history = []
        best_path, last_path = self.layout.checkpoints / "best.pt", self.layout.checkpoints / "last.pt"
        if resume_path and resume_path.resolve() != best_path.resolve():
            copy2(resume_path, best_path)
        reporter = EpochReporter(self.layout.metrics / "epochs.csv", create_tensorboard_writer(self.layout.tensorboard))
        try:
            for epoch in range(start_epoch, self.args.epochs):
                maximum_cost_weight = (
                    getattr(self.args, "cost_weight", 0.0)
                    if getattr(self.args, "craquelure_to_crack_cost", 0.0) > 0
                    else 0.0
                )
                self.criterion.cost_weight = cost_weight_for_epoch(
                    epoch + 1, maximum_cost_weight
                )
                train_metrics = train_epoch(
                    self.model,
                    self.train_loader,
                    self.optimizer,
                    self.criterion,
                    self.device,
                    self.scaler,
                    getattr(self.args, "gradient_accumulation_steps", 1),
                )
                self.criterion.cost_weight = maximum_cost_weight
                val_metrics = evaluate(
                    self.model,
                    self.val_loader,
                    self.criterion,
                    self.device,
                    self.plan.expert_name,
                    self.plan.ignore_value,
                )
                monitor = float(val_metrics["panel_macro_loss"])
                improved = monitor < best
                best = min(best, monitor)
                bad_epochs = 0 if improved else bad_epochs + 1
                self.scheduler.step(monitor)
                learning_rates = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
                history.append(
                    {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics, "lr": learning_rates}
                )
                write_json(self.layout.metrics / "history.json", history)
                reporter.record(epoch + 1, train_metrics, val_metrics, learning_rates)
                if improved:
                    self._save(best_path, epoch, best, bad_epochs, val_metrics)
                self._save(last_path, epoch, best, bad_epochs, val_metrics)
                score = val_metrics["expert_panel_macro"]["iou"]
                log_message(
                    self.layout, f"epoch={epoch + 1} loss={train_metrics['loss']:.4f} "
                    f"val_panel_loss={monitor:.4f} val_{self.plan.expert_name}_panel_IoU={score:.4f} "
                    f"lr={learning_rates}"
                )
                log_message(self.layout, format_metrics(val_metrics["tile_micro"]))
                if (
                    self.args.early_stop_patience is not None
                    and bad_epochs >= self.args.early_stop_patience
                ):
                    log_message(self.layout, f"early stopping after {bad_epochs} epochs without lower panel-macro loss")
                    break
            if len(self.plan.class_names) == 3:
                best_epoch, test_metrics, rows = prepare_joint_final_evaluation(
                    best_path=best_path, last_path=last_path, model=self.model,
                    criterion=self.criterion, validation_loader=self.val_loader,
                    test_loader=self.test_loader, device=self.device,
                    output_root=self.layout.root, plan=self.plan,
                    evaluate_outer_test=not getattr(self.args, "skip_outer_test", False),
                )
                if test_metrics is not None:
                    write_json(self.layout.metrics / "outer_test_metrics.json", test_metrics)
                    log_message(self.layout, f"outer_test_macro_IoU={test_metrics['tile_micro']['miou']:.4f}")
            else:
                if self.plan.expert_name == "foreground":
                    best_epoch, policy, test_metrics, rows = (
                        prepare_fixed_binary_final_evaluation(
                            best_path=best_path,
                            model=self.model,
                            criterion=self.criterion,
                            validation_loader=self.val_loader,
                            test_loader=self.test_loader,
                            device=self.device,
                            output_root=self.layout.root,
                            plan=self.plan,
                        )
                    )
                else:
                    best_epoch, policy, test_metrics, rows = prepare_final_evaluation(
                        best_path=best_path, model=self.model, criterion=self.criterion,
                        validation_loader=self.val_loader, test_loader=self.test_loader,
                        device=self.device, output_root=self.layout.root, plan=self.plan,
                        writer=reporter.writer,
                    )
                write_json(self.layout.metrics / "outer_test_metrics.json", test_metrics)
                score = test_metrics["expert_panel_macro"]["iou"]
                score_text = "null" if score is None else f"{score:.4f}"
                log_message(self.layout, f"outer_test_{self.plan.expert_name}_panel_IoU={score_text} threshold={policy.threshold:.2f}")
            write_tensorboard_validation_images(reporter.writer, self.layout.root, rows, best_epoch)
        finally:
            reporter.close()
        export_tensorboard_artifacts(self.layout)
        build_training_dashboard(self.layout)

    def _save(
        self,
        path: Path,
        epoch: int,
        best: float,
        bad_epochs: int,
        val_metrics: dict[str, Any],
    ) -> None:
        save_checkpoint(
            path,
            epoch=epoch + 1,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.scheduler,
            best=best,
            bad_epochs=bad_epochs,
            args=self.args,
            plan=self.plan,
            val=val_metrics,
        )


def main_for_project(
    project: TrainingProject,
    argv: Sequence[str] | None = None,
) -> int:
    """Run one independent model project through the shared lifecycle."""

    cli = training_parser(project)
    args = cli.parse_args(argv)
    project.resolve_model_args(args)
    try:
        plan = prepare_dataset(args.dataset_root, args.expert, args.outer_fold, args.inner_fold)
    except DataError as exc:
        cli.error(str(exc))
    layout = create_run_layout(args, plan, project_root=project.root)
    log_message(
        layout, f"expert={plan.expert_name!r} source_id={plan.expert_id} "
        f"train={len(plan.train)} val={len(plan.val)} "
        f"test={len(plan.test)} outer={plan.outer_fold} inner={plan.inner_fold}"
    )
    if not args.validate_only:
        TrainingRun.create(args, plan, layout, project).execute()
    return 0
