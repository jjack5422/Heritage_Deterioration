"""Run Fold-first two-stage specialization with durable validation reporting."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .checkpoints import (
    adaptation_state as _adaptation_state,
    checkpoint_model_variant as _checkpoint_model_variant,
    expected_adaptation_keys as _expected_adaptation_keys,
    load_adaptation_checkpoint as _load_adaptation,
)
from .checkpoint_selection import (
    LOSS_F1_TOLERANCE,
    ValidationCandidate,
    select_constrained_joint,
    update_pareto_front,
)
from .concepts import CHANNEL_ORDER, concept_contract_record, load_concept_registry
from .data import MonumentDeteriorationDataset
from .losses import LossOutput, hybrid_multilabel_objective, multilabel_objective
from .metrics import BoundaryConfusion, MultilabelConfusion
from .model import (
    SUPPORTED_MODEL_VARIANTS,
    DualAdapterSam3,
    DualAdapterSam3Output,
    build_dual_adapter_model,
)
from .reporting import EpochReporter, RunLayout, finalize_reporting, save_concept_qualitative, write_json
from .sam3_integration import DEFAULT_CHECKPOINT
from .splits import fold_membership, load_split_contract, source_group_index
from .training import (
    RouterHealthGate,
    assert_finite_gradients,
    build_optimizer_and_scheduler,
    build_stage2_hard_pool,
    set_epoch_router_temperature,
    set_reproducible_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT.parent / "datasets" / "dataset_clean_v2_merged_craquelure"
DEFAULT_EXPERIMENT = "2026-09-01_dual-adapter-sam3-multilabel-512_seed42"


def _variant_run_root(experiment_root: Path, model_variant: str, fold: int) -> Path:
    if model_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(
            f"unsupported model variant {model_variant!r}; expected one of {SUPPORTED_MODEL_VARIANTS}"
        )
    if fold not in range(5):
        raise ValueError(f"fold must be in [0, 5), got {fold}")
    return experiment_root / "5fold" / model_variant / f"fold{fold}"


def _write_or_validate_model_contract(path: Path, contract: Mapping[str, Any]) -> None:
    requested = dict(contract)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        normalized_existing = dict(existing)
        if "model_variant" not in normalized_existing:
            normalized_existing["model_variant"] = "da_sam3"
        if normalized_existing != requested:
            raise RuntimeError(
                "existing model contract is incompatible with requested model variant: "
                f"existing={existing.get('model_variant')!r}, "
                f"requested={requested.get('model_variant')!r}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, requested)


def _worker_seed(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng.seed(torch.initial_seed() % (2**32))


def _loader(
    dataset: MonumentDeteriorationDataset,
    *,
    batch_size: int,
    train: bool,
    weights: Tensor | None = None,
    workers: int = 4,
) -> DataLoader:
    generator = torch.Generator().manual_seed(42)
    sampler = None
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
        drop_last=False,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _routing_terms(output: DualAdapterSam3Output) -> list[tuple[Tensor, Tensor, Tensor]]:
    return [
        (diagnostics.logits, diagnostics.soft_probabilities, diagnostics.hard_assignments)
        for concept in output.routing
        for diagnostics in concept
    ]


def _objective(
    output: DualAdapterSam3Output,
    batch: dict[str, Any],
    *,
    craquelure_loss: str = "original",
) -> LossOutput:
    objective = (
        multilabel_objective
        if craquelure_loss == "original"
        else hybrid_multilabel_objective
    )
    if craquelure_loss not in ("original", "sam2_bce_dice"):
        raise ValueError(f"unsupported craquelure loss profile: {craquelure_loss}")
    return objective(
        output.logits,
        output.presence_logits,
        batch["targets"],
        batch["valid_masks"],
        batch["class_present"],
        routing=_routing_terms(output),
    )


def _routing_rows(output: DualAdapterSam3Output, *, epoch: int, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept_index, concept_routing in enumerate(output.routing):
        for layer_offset, diagnostics in enumerate(concept_routing):
            for expert in range(4):
                rows.append({
                    "epoch": epoch,
                    "stage": stage,
                    "concept": CHANNEL_ORDER[concept_index],
                    "layer": layer_offset,
                    "expert": expert,
                    "top1_usage": float(diagnostics.top1_usage[expert].detach()),
                    "top2_usage": float(diagnostics.top2_usage[expert].detach()),
                    "entropy": float(diagnostics.entropy.detach()),
                    "logits_magnitude": float(diagnostics.logits.detach().abs().mean()),
                })
    return rows


def _mean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["epoch"], row["stage"], row["concept"], row["layer"], row["expert"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append({
            "epoch": key[0], "stage": key[1], "concept": key[2], "layer": key[3], "expert": key[4],
            **{column: sum(float(row[column]) for row in values) / len(values) for column in ("top1_usage", "top2_usage", "entropy", "logits_magnitude")},
        })
    return result


def _run_train_epoch(
    model: DualAdapterSam3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    stage: str,
    craquelure_loss: str = "original",
) -> tuple[float, list[dict[str, Any]]]:
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    routing_rows: list[dict[str, Any]] = []
    first_batch = True
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
            loss = _objective(output, batch, craquelure_loss=craquelure_loss)
        loss.total.backward()
        if first_batch:
            assert_finite_gradients(model)
            first_batch = False
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        batch_size = int(batch["image"].shape[0])
        total += float(loss.total.detach()) * batch_size
        count += batch_size
        routing_rows.extend(_routing_rows(output, epoch=epoch, stage=stage))
    return total / count, _mean_rows(routing_rows)


@torch.no_grad()
def _run_validation(
    model: DualAdapterSam3,
    loader: DataLoader,
    device: torch.device,
    *,
    craquelure_loss: str = "original",
) -> tuple[float, dict[str, Any]]:
    model.eval()
    confusion = MultilabelConfusion.empty(device=device)
    boundary = BoundaryConfusion.empty(device=device, tolerance=2)
    segmentation_total = 0.0
    count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
            loss = _objective(output, batch, craquelure_loss=craquelure_loss)
        size = int(batch["image"].shape[0])
        segmentation_total += float(loss.segmentation) * size
        count += size
        confusion.update(output.logits, batch["targets"], batch["valid_masks"])
        predictions = output.logits[:, 0].sigmoid() >= 0.5
        boundary.update(predictions, batch["targets"][:, 0], batch["valid_masks"][:, 0])
    metrics = confusion.compute()
    metrics["boundary"] = {"crack_craquelure": boundary.compute()}
    return segmentation_total / count, metrics


def _save_checkpoint(
    path: Path,
    model: DualAdapterSam3,
    *,
    stage: str,
    epoch: int,
    validation_segmentation_loss: float,
    registry_hash: str,
    split_hash: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    validation_metrics: Mapping[str, Any] | None = None,
    selection_criterion: str = "minimum validation segmentation loss",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 2,
        "model_variant": model.model_variant,
        "stage": stage,
        "epoch": epoch,
        "validation_segmentation_loss": validation_segmentation_loss,
        "validation_metrics": dict(validation_metrics or {}),
        "selection_criterion": selection_criterion,
        "adaptation_state": _adaptation_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "prompt_contract_sha256": registry_hash,
        "split_contract_sha256": split_hash,
        "model_contract": model.model_contract,
    }, path)


def _validation_candidate(
    path: Path,
    *,
    stage: str,
    epoch: int,
    metrics: Mapping[str, Any],
) -> ValidationCandidate:
    per_class = metrics["per_class"]
    boundary = metrics["boundary"]["crack_craquelure"]
    return ValidationCandidate(
        path=path,
        stage=stage,
        epoch=epoch,
        craquelure_f1=float(per_class["crack_craquelure"]["f1"]),
        boundary_f1=float(boundary["f1"]),
        loss_f1=float(per_class["loss"]["f1"]),
    )


def _update_checkpoint_front(
    candidates: tuple[ValidationCandidate, ...],
    *,
    layout: RunLayout,
    model: DualAdapterSam3,
    stage: str,
    epoch: int,
    validation_loss: float,
    metrics: Mapping[str, Any],
    registry_hash: str,
    split_hash: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[ValidationCandidate, ...]:
    candidate_path = layout.checkpoints / f"{stage}_candidates" / f"epoch_{epoch:03d}.pt"
    candidate = _validation_candidate(
        candidate_path,
        stage=stage,
        epoch=epoch,
        metrics=metrics,
    )
    updated, removed, accepted = update_pareto_front(candidates, candidate)
    if not accepted:
        return candidates
    _save_checkpoint(
        candidate_path,
        model,
        stage=stage,
        epoch=epoch,
        validation_segmentation_loss=validation_loss,
        registry_hash=registry_hash,
        split_hash=split_hash,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_metrics=metrics,
        selection_criterion=(
            "maximize 0.5*craquelure_F1 + 0.5*craquelure_boundary_F1 subject "
            f"to loss_F1 >= best_loss_F1 - {LOSS_F1_TOLERANCE}"
        ),
    )
    for obsolete in removed:
        obsolete.path.unlink(missing_ok=True)
    return updated


def _materialize_constrained_checkpoints(
    candidates: tuple[ValidationCandidate, ...],
    *,
    layout: RunLayout,
    stage: str,
) -> dict[str, Any]:
    joint = select_constrained_joint(candidates)
    best_craquelure = max(
        candidates,
        key=lambda candidate: (candidate.craquelure_score, candidate.loss_f1, -candidate.epoch),
    )
    best_loss = max(
        candidates,
        key=lambda candidate: (candidate.loss_f1, candidate.craquelure_score, -candidate.epoch),
    )
    destinations = {
        f"{stage}_best.pt": joint,
        f"{stage}_best_craquelure.pt": best_craquelure,
        f"{stage}_best_loss.pt": best_loss,
    }
    if stage == "stage2":
        destinations.update({
            "best_joint.pt": joint,
            "best_craquelure.pt": best_craquelure,
            "best_loss.pt": best_loss,
        })
    for name, candidate in destinations.items():
        shutil.copy2(candidate.path, layout.checkpoints / name)
    best_loss_f1 = best_loss.loss_f1
    return {
        "criterion": (
            "maximize 0.5*craquelure_F1 + 0.5*craquelure_boundary_F1 subject "
            f"to loss_F1 >= best_loss_F1 - {LOSS_F1_TOLERANCE}"
        ),
        "loss_f1_tolerance": LOSS_F1_TOLERANCE,
        "best_loss_f1": best_loss_f1,
        "minimum_eligible_loss_f1": best_loss_f1 - LOSS_F1_TOLERANCE,
        "selected": {
            "path": str(joint.path),
            "epoch": joint.epoch,
            "craquelure_f1": joint.craquelure_f1,
            "craquelure_boundary_f1": joint.boundary_f1,
            "craquelure_score": joint.craquelure_score,
            "loss_f1": joint.loss_f1,
        },
        "diagnostic_best_craquelure": {
            "path": str(best_craquelure.path),
            "epoch": best_craquelure.epoch,
            "craquelure_score": best_craquelure.craquelure_score,
            "loss_f1": best_craquelure.loss_f1,
        },
        "diagnostic_best_loss": {
            "path": str(best_loss.path),
            "epoch": best_loss.epoch,
            "craquelure_score": best_loss.craquelure_score,
            "loss_f1": best_loss.loss_f1,
        },
        "pareto_candidate_count": len(candidates),
    }


@torch.no_grad()
def _training_tile_losses(
    model: DualAdapterSam3,
    loader: DataLoader,
    device: torch.device,
    *,
    craquelure_loss: str = "original",
) -> tuple[dict[str, float], set[str]]:
    losses: dict[str, float] = {}
    both_class: set[str] = set()
    model.eval()
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
        for index, name in enumerate(batch["image_id"]):
            one_output = DualAdapterSam3Output(
                logits=output.logits[index : index + 1],
                presence_logits=output.presence_logits[index : index + 1],
                routing=(),
                vision_forward_calls=output.vision_forward_calls,
            )
            one_batch = {
                "targets": batch["targets"][index : index + 1],
                "valid_masks": batch["valid_masks"][index : index + 1],
                "class_present": batch["class_present"][index : index + 1],
            }
            one = _objective(
                one_output,
                one_batch,
                craquelure_loss=craquelure_loss,
            )
            losses[str(name)] = float(one.segmentation)
            if bool(batch["class_present"][index].all()):
                both_class.add(str(name))
    return losses, both_class


@torch.no_grad()
def _qualitative_rows(model: DualAdapterSam3, loader: DataLoader, device: torch.device, layout: RunLayout) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
        predictions = output.logits.sigmoid() >= 0.5
        for sample_index, image_id in enumerate(batch["image_id"]):
            rgb = (batch["image"][sample_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            for concept_index, concept in enumerate(CHANNEL_ORDER):
                target = batch["targets"][sample_index, concept_index].detach().cpu().numpy().astype(bool)
                valid = batch["valid_masks"][sample_index, concept_index].detach().cpu().numpy().astype(bool)
                prediction = predictions[sample_index, concept_index].detach().cpu().numpy().astype(bool)
                rows.append(save_concept_qualitative(
                    layout, concept=concept, image_id=str(image_id), input_rgb=rgb,
                    target=target & valid, prediction=prediction & valid,
                ))
    return rows


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT.parent, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def train_fold(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full official SAM3 training")
    set_reproducible_seed(42)
    device = torch.device("cuda")
    registry = load_concept_registry(args.concepts)
    split = load_split_contract(args.splits, dataset_root=args.dataset)
    membership = fold_membership(split, args.fold)
    group_index = source_group_index(split)
    train_dataset = MonumentDeteriorationDataset(args.dataset, membership.train, group_index, train_augmentation=True, seed=42)
    train_eval_dataset = MonumentDeteriorationDataset(args.dataset, membership.train, group_index, train_augmentation=False, seed=42)
    validation_dataset = MonumentDeteriorationDataset(args.dataset, membership.validation, group_index, train_augmentation=False, seed=42)
    train_loader = _loader(train_dataset, batch_size=args.batch_size, train=True, workers=args.workers)
    train_eval_loader = _loader(train_eval_dataset, batch_size=args.batch_size, train=False, workers=args.workers)
    validation_loader = _loader(validation_dataset, batch_size=args.batch_size, train=False, workers=args.workers)

    experiment_root = PROJECT_ROOT / "runs" / args.experiment_id
    run_root = _variant_run_root(experiment_root, args.model_variant, args.fold)
    if run_root.exists() and any(run_root.iterdir()) and not args.resume_compatible:
        raise FileExistsError(f"run directory already contains artifacts: {run_root}")

    model = build_dual_adapter_model(
        registry,
        model_variant=args.model_variant,
        checkpoint=args.checkpoint,
        device=device,
        full_pixel_decoder=args.full_pixel_decoder,
    ).to(device)
    info = experiment_root / "info"
    _write_or_validate_model_contract(info / "model_contract.json", model.model_contract)
    layout = RunLayout.create(run_root)
    info.mkdir(parents=True, exist_ok=True)
    write_json(info / "prompt_contract.json", concept_contract_record(registry))
    write_json(info / "split_contract.json", split.payload)
    write_json(info / "dataset_contract.json", json.loads((Path(args.dataset) / "manifest.json").read_text(encoding="utf-8")))
    write_json(info / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "git_revision": _git_revision(), "seed": 42,
    })
    write_json(info / "experiment_contract.json", {
        "experiment_type": "full_stage1_stage2_training",
        "model_variant": args.model_variant,
        "full_effective_pixel_decoder": args.full_pixel_decoder,
        "craquelure_loss": args.craquelure_loss,
        "class_weights": {"crack_craquelure": 0.5, "loss": 0.5},
        "checkpoint_selection": args.checkpoint_selection,
        "loss_f1_tolerance": (
            LOSS_F1_TOLERANCE if args.checkpoint_selection == "constrained_joint" else None
        ),
        "outer_test_excluded_from_selection": True,
    })
    write_json(layout.config / "resolved_config.json", vars(args))

    writer = SummaryWriter(log_dir=layout.tensorboard)
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    router_path = layout.metrics / "router_usage.csv"
    router_handle = router_path.open("w", encoding="utf-8", newline="")
    router_writer = csv.DictWriter(router_handle, fieldnames=("epoch", "stage", "concept", "layer", "expert", "top1_usage", "top2_usage", "entropy", "logits_magnitude"))
    router_writer.writeheader()
    router_health = RouterHealthGate()
    global_epoch = 0
    stage1_best = float("inf")
    stage1_best_path = layout.checkpoints / "stage1_best.pt"
    stage1_bundle = build_optimizer_and_scheduler(model, stage="stage1", epochs=args.stage1_epochs)
    stage1_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    write_json(layout.config / "stage1_trainable_scope.json", {
        "names": stage1_names,
        "tensor_count": len(stage1_names),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "optimizer_groups": [
            {"learning_rate": group["lr"], "weight_decay": group["weight_decay"], "parameter_count": sum(parameter.numel() for parameter in group["params"])}
            for group in stage1_bundle.optimizer.param_groups
        ],
    })
    stage1_candidates: tuple[ValidationCandidate, ...] = ()
    started = time.time()
    for local_epoch in range(args.stage1_epochs):
        global_epoch += 1
        temperature = set_epoch_router_temperature(model, local_epoch)
        train_loss, router_rows = _run_train_epoch(
            model, train_loader, stage1_bundle.optimizer, device,
            epoch=global_epoch, stage="stage1", craquelure_loss=args.craquelure_loss,
        )
        val_loss, metrics = _run_validation(
            model, validation_loader, device, craquelure_loss=args.craquelure_loss
        )
        lr = stage1_bundle.optimizer.param_groups[0]["lr"]
        macro = metrics["macro"]
        reporter.record(global_epoch, train_loss=train_loss, validation_loss=val_loss, f1=macro["f1"], precision=macro["precision"], recall=macro["recall"], iou=macro["iou"], accuracy=macro["accuracy"], learning_rate=lr)
        for concept in CHANNEL_ORDER:
            writer.add_scalar(f"metrics/{concept}/f1", metrics["per_class"][concept]["f1"], global_epoch)
            writer.add_scalar(f"metrics/{concept}/iou", metrics["per_class"][concept]["iou"], global_epoch)
            writer.add_scalar(f"metrics/{concept}/precision", metrics["per_class"][concept]["precision"], global_epoch)
            writer.add_scalar(f"metrics/{concept}/recall", metrics["per_class"][concept]["recall"], global_epoch)
        boundary = metrics["boundary"]["crack_craquelure"]
        writer.add_scalar("metrics/crack_craquelure/boundary_f1", boundary["f1"], global_epoch)
        writer.add_scalar("metrics/macro_f1", macro["f1"], global_epoch)
        writer.add_scalar("router/temperature", temperature, global_epoch)
        router_writer.writerows(router_rows)
        router_handle.flush()
        aggregated_usage = router_health.update(router_rows)
        for (layer_index, expert_index), usage in aggregated_usage.items():
            writer.add_scalar(f"router/layer_{layer_index}/expert_{expert_index}_usage", usage, global_epoch)
        if args.checkpoint_selection == "constrained_joint":
            stage1_candidates = _update_checkpoint_front(
                stage1_candidates,
                layout=layout,
                model=model,
                stage="stage1",
                epoch=global_epoch,
                validation_loss=val_loss,
                metrics=metrics,
                registry_hash=registry.sha256,
                split_hash=split.sha256,
                optimizer=stage1_bundle.optimizer,
                scheduler=stage1_bundle.scheduler,
            )
        elif val_loss < stage1_best:
            stage1_best = val_loss
            _save_checkpoint(stage1_best_path, model, stage="stage1", epoch=global_epoch, validation_segmentation_loss=val_loss, registry_hash=registry.sha256, split_hash=split.sha256, optimizer=stage1_bundle.optimizer, scheduler=stage1_bundle.scheduler, validation_metrics=metrics)
        _save_checkpoint(layout.checkpoints / "stage1_last.pt", model, stage="stage1", epoch=global_epoch, validation_segmentation_loss=val_loss, registry_hash=registry.sha256, split_hash=split.sha256, optimizer=stage1_bundle.optimizer, scheduler=stage1_bundle.scheduler, validation_metrics=metrics)
        stage1_bundle.scheduler.step()
        print(json.dumps({"stage": "stage1", "epoch": local_epoch + 1, "train_loss": train_loss, "validation_segmentation_loss": val_loss, "macro_f1": macro["f1"], "craquelure_f1": metrics["per_class"]["crack_craquelure"]["f1"], "craquelure_boundary_f1": boundary["f1"], "loss_f1": metrics["per_class"]["loss"]["f1"], "elapsed_minutes": (time.time() - started) / 60}, ensure_ascii=False), flush=True)

    stage1_selection: dict[str, Any] | None = None
    if args.checkpoint_selection == "constrained_joint":
        stage1_selection = _materialize_constrained_checkpoints(
            stage1_candidates, layout=layout, stage="stage1"
        )
        stage1_checkpoint = torch.load(stage1_best_path, map_location="cpu", weights_only=True)
        stage1_best = float(stage1_checkpoint["validation_segmentation_loss"])
    _load_adaptation(stage1_best_path, model, expected_prompt_hash=registry.sha256, expected_split_hash=split.sha256)
    per_tile_losses, both_class = _training_tile_losses(
        model, train_eval_loader, device, craquelure_loss=args.craquelure_loss
    )
    hard_pool = build_stage2_hard_pool(membership.train, both_class, per_tile_losses)
    write_json(layout.config / "stage2_hard_pool.json", {"source": "stage1_best", "tile_ids": list(hard_pool), "both_class_tile_ids": sorted(both_class), "per_tile_stage1_segmentation_loss": per_tile_losses, "top_fraction": 0.25, "seed": 42})
    hard_set = set(hard_pool)
    train_count = len(train_dataset)
    hard_count = len(hard_set)
    weights = torch.tensor([0.5 / train_count + (0.5 / hard_count if name in hard_set else 0.0) for name in train_dataset.names], dtype=torch.double)
    stage2_loader = _loader(train_dataset, batch_size=args.batch_size, train=True, weights=weights, workers=args.workers)
    stage2_bundle = build_optimizer_and_scheduler(model, stage="stage2", epochs=args.stage2_epochs)
    stage2_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    write_json(layout.config / "stage2_trainable_scope.json", {
        "names": stage2_names,
        "tensor_count": len(stage2_names),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "optimizer_groups": [
            {"learning_rate": group["lr"], "weight_decay": group["weight_decay"], "parameter_count": sum(parameter.numel() for parameter in group["params"])}
            for group in stage2_bundle.optimizer.param_groups
        ],
    })
    stage2_best = float("inf")
    stage2_best_path = layout.checkpoints / "stage2_best.pt"
    stage2_candidates: tuple[ValidationCandidate, ...] = ()
    for local_epoch in range(args.stage2_epochs):
        global_epoch += 1
        model.set_router_temperature(1.0)
        train_loss, router_rows = _run_train_epoch(
            model, stage2_loader, stage2_bundle.optimizer, device,
            epoch=global_epoch, stage="stage2", craquelure_loss=args.craquelure_loss,
        )
        val_loss, metrics = _run_validation(
            model, validation_loader, device, craquelure_loss=args.craquelure_loss
        )
        lr = stage2_bundle.optimizer.param_groups[0]["lr"]
        macro = metrics["macro"]
        reporter.record(global_epoch, train_loss=train_loss, validation_loss=val_loss, f1=macro["f1"], precision=macro["precision"], recall=macro["recall"], iou=macro["iou"], accuracy=macro["accuracy"], learning_rate=lr)
        for concept in CHANNEL_ORDER:
            writer.add_scalar(f"metrics/{concept}/f1", metrics["per_class"][concept]["f1"], global_epoch)
            writer.add_scalar(f"metrics/{concept}/iou", metrics["per_class"][concept]["iou"], global_epoch)
            writer.add_scalar(f"metrics/{concept}/precision", metrics["per_class"][concept]["precision"], global_epoch)
            writer.add_scalar(f"metrics/{concept}/recall", metrics["per_class"][concept]["recall"], global_epoch)
        boundary = metrics["boundary"]["crack_craquelure"]
        writer.add_scalar("metrics/crack_craquelure/boundary_f1", boundary["f1"], global_epoch)
        writer.add_scalar("metrics/macro_f1", macro["f1"], global_epoch)
        router_writer.writerows(router_rows)
        router_handle.flush()
        aggregated_usage = router_health.update(router_rows)
        for (layer_index, expert_index), usage in aggregated_usage.items():
            writer.add_scalar(f"router/layer_{layer_index}/expert_{expert_index}_usage", usage, global_epoch)
        if args.checkpoint_selection == "constrained_joint":
            stage2_candidates = _update_checkpoint_front(
                stage2_candidates,
                layout=layout,
                model=model,
                stage="stage2",
                epoch=global_epoch,
                validation_loss=val_loss,
                metrics=metrics,
                registry_hash=registry.sha256,
                split_hash=split.sha256,
                optimizer=stage2_bundle.optimizer,
                scheduler=stage2_bundle.scheduler,
            )
        elif val_loss < stage2_best:
            stage2_best = val_loss
            _save_checkpoint(stage2_best_path, model, stage="stage2", epoch=global_epoch, validation_segmentation_loss=val_loss, registry_hash=registry.sha256, split_hash=split.sha256, optimizer=stage2_bundle.optimizer, scheduler=stage2_bundle.scheduler, validation_metrics=metrics)
        _save_checkpoint(layout.checkpoints / "stage2_last.pt", model, stage="stage2", epoch=global_epoch, validation_segmentation_loss=val_loss, registry_hash=registry.sha256, split_hash=split.sha256, optimizer=stage2_bundle.optimizer, scheduler=stage2_bundle.scheduler, validation_metrics=metrics)
        stage2_bundle.scheduler.step()
        print(json.dumps({"stage": "stage2", "epoch": local_epoch + 1, "train_loss": train_loss, "validation_segmentation_loss": val_loss, "macro_f1": macro["f1"], "craquelure_f1": metrics["per_class"]["crack_craquelure"]["f1"], "craquelure_boundary_f1": boundary["f1"], "loss_f1": metrics["per_class"]["loss"]["f1"], "elapsed_minutes": (time.time() - started) / 60}, ensure_ascii=False), flush=True)

    router_handle.close()
    stage2_selection: dict[str, Any] | None = None
    if args.checkpoint_selection == "constrained_joint":
        stage2_selection = _materialize_constrained_checkpoints(
            stage2_candidates, layout=layout, stage="stage2"
        )
        stage2_checkpoint = torch.load(stage2_best_path, map_location="cpu", weights_only=True)
        stage2_best = float(stage2_checkpoint["validation_segmentation_loss"])
    selected = _load_adaptation(stage2_best_path, model, expected_prompt_hash=registry.sha256, expected_split_hash=split.sha256)
    validation_rows = _qualitative_rows(model, validation_loader, device, layout)
    write_json(layout.metrics / "selection.json", {
        "criterion": selected.get("selection_criterion", "minimum validation segmentation loss"),
        "stage1_best": str(stage1_best_path),
        "stage1_best_loss": stage1_best,
        "stage1_selection": stage1_selection,
        "stage2_best": str(stage2_best_path),
        "stage2_best_loss": stage2_best,
        "stage2_selection": stage2_selection,
        "selected_epoch": selected["epoch"],
        "outer_test_excluded_from_selection": True,
    })
    finalize_reporting(
        layout, writer=writer, reporter=reporter, validation_rows=validation_rows,
        selected_epoch=int(selected["epoch"]),
        outer_test_metrics={
            "status": "deferred_until_all_five_validation_checkpoints_are_locked",
            "selected_epoch": int(selected["epoch"]),
            "selection_source": selected.get("selection_criterion", "stage2 minimum validation segmentation loss"),
            "outer_test_excluded_from_selection": True,
        },
    )
    print(json.dumps({"status": "complete", "run_root": str(run_root), "selected_checkpoint": str(stage2_best_path)}, ensure_ascii=False), flush=True)
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--model-variant",
        choices=SUPPORTED_MODEL_VARIANTS,
        default="da_sam3",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--concepts", type=Path, default=PROJECT_ROOT / "configs" / "concepts.yaml")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "configs" / "splits.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage1-epochs", type=int, default=60)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--full-pixel-decoder",
        action="store_true",
        help="train both PixelDecoder stages used by the three-FPN path and semantic_seg_head",
    )
    parser.add_argument(
        "--craquelure-loss",
        choices=("original", "sam2_bce_dice"),
        default="original",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation_loss", "constrained_joint"),
        default="validation_loss",
    )
    parser.add_argument("--resume-compatible", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage1_epochs < 1 or args.stage2_epochs < 1:
        raise ValueError("both training stages require at least one epoch")
    if args.checkpoint_selection == "constrained_joint" and not args.full_pixel_decoder:
        raise ValueError("constrained_joint selection is reserved for full PixelDecoder training")
    train_fold(args)


if __name__ == "__main__":
    main()
