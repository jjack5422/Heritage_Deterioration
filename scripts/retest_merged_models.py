"""Leak-free OOF retest of trained crack/craquelure models on merged labels.

The trained checkpoints use ``datasets/dataset_v2_3class`` folds.  This runner
therefore keeps each checkpoint's original validation and outer-test members,
but reads RGB images and merged masks from
``datasets/dataset_clean_v2_merged_craquelure``.  The two datasets must have
the same image-manifest hash.  Other deterioration labels in the merged masks
are excluded from binary scoring; only raw label 0 (background) and raw label
1 (merged crack/craquelure) are valid.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from sam2_sac.metrics import binary_summary, counts_summary
from sam2_sac.reporting import (
    EpochReporter,
    RunLayout,
    append_log,
    binary_metric_row,
    finalize_reporting,
    save_qualitative_example,
    write_json,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = WORKSPACE_ROOT / "datasets" / "dataset_clean_v2_merged_craquelure"
DEFAULT_SOURCE_DATASET = WORKSPACE_ROOT / "datasets" / "dataset_v2_3class"
DEFAULT_SAM2_CHECKPOINT = (
    WORKSPACE_ROOT / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_large.pt"
)
FIXED_THRESHOLD = 0.5
IGNORE_VALUE = 255
IMAGE_SIZE = 512
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    project: str
    source_experiment: str
    experiment_id: str
    kind: str
    history_experts: tuple[str, ...]
    checkpoint_experts: tuple[str, ...]
    batch_size: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "resunet50": ModelSpec(
        key="resunet50",
        project="unet",
        source_experiment="2026-08-17_dataset-v2_3class_resunet50_seed42",
        experiment_id="2026-08-20_merged-craquelure_oof-retest_resunet50_seed42",
        kind="unet",
        history_experts=("crack_craquelure",),
        checkpoint_experts=("crack_craquelure",),
        batch_size=8,
    ),
    "convnext-large": ModelSpec(
        key="convnext-large",
        project="unet",
        source_experiment="2026-08-17_dataset-v2_3class_convnext-large_seed42",
        experiment_id="2026-08-20_merged-craquelure_oof-retest_convnext-large_seed42",
        kind="unet",
        history_experts=("crack_craquelure",),
        checkpoint_experts=("crack_craquelure",),
        batch_size=4,
    ),
    "segformer-b5": ModelSpec(
        key="segformer-b5",
        project="segformer",
        source_experiment="2026-08-18_dataset-v2_3class_segformer-b5_seed42",
        experiment_id="2026-08-20_merged-craquelure_oof-retest_segformer-b5_seed42",
        kind="segformer",
        history_experts=("crack_craquelure",),
        checkpoint_experts=("crack_craquelure",),
        batch_size=4,
    ),
    "sam2-sac": ModelSpec(
        key="sam2-sac",
        project="sam2_sac",
        source_experiment="2026-08-18_sam2-hiera-large_h0_512_80ep_seed42",
        experiment_id="2026-08-20_merged-craquelure_oof-retest_sam2-sac_seed42",
        kind="sam2_sac",
        history_experts=("union",),
        checkpoint_experts=("union",),
        batch_size=2,
    ),
    "sam2-adapter": ModelSpec(
        key="sam2-adapter",
        project="sam2_adapter",
        source_experiment="2026-08-20_sam2-adapter-hiera-large_dual-expert_512_80ep_seed42",
        experiment_id="2026-08-20_merged-craquelure_oof-retest_sam2-adapter_seed42",
        kind="sam2_adapter",
        history_experts=("crack", "craquelure"),
        checkpoint_experts=("crack", "craquelure"),
        batch_size=1,
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def panel_name(tile: str) -> str:
    match = re.match(r"(.+?)_R\d+_C\d+", Path(tile).stem)
    if not match:
        raise ValueError(f"cannot parse source panel from tile name: {tile}")
    return match.group(1)


def merged_binary_target(raw_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return binary target and scoring validity for the merged contract."""

    if raw_mask.ndim != 2:
        raise ValueError(f"merged mask must be HxW, got {raw_mask.shape}")
    return raw_mask == 1, np.logical_or(raw_mask == 0, raw_mask == 1)


def merged_foreground_probability(probabilities: np.ndarray) -> np.ndarray:
    """Collapse background/crack/craquelure probabilities to foreground."""

    if probabilities.ndim < 3 or probabilities.shape[-3] != 3:
        raise ValueError("expected probabilities with a three-class channel axis")
    return probabilities[..., 1, :, :] + probabilities[..., 2, :, :]


def source_oof_partitions(
    source_dataset: str | Path, *, outer_fold: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]:
    """Recreate the source checkpoint's nested split without reading target folds."""

    root = Path(source_dataset)
    numbered = sorted(
        int(path.stem[4:])
        for path in (root / "splits").glob("fold*.json")
        if path.stem[4:].isdigit()
    )
    if outer_fold not in numbered or len(numbered) < 2:
        raise ValueError("source dataset needs at least two numbered folds")
    outer = _read_json(root / "splits" / f"fold{outer_fold}.json")
    inner_fold = numbered[(numbered.index(outer_fold) + 1) % len(numbered)]
    inner = _read_json(root / "splits" / f"fold{inner_fold}.json")
    outer_train = tuple(str(name) for name in outer["folds"][0]["train"])
    outer_test = tuple(str(name) for name in outer["holdout_tiles"])
    validation_set = {str(name) for name in inner["holdout_tiles"]}
    if not validation_set.issubset(outer_train):
        raise ValueError(
            f"inner fold {inner_fold} holdout is not contained in outer fold {outer_fold} train"
        )
    train = tuple(name for name in outer_train if name not in validation_set)
    validation = tuple(name for name in outer_train if name in validation_set)
    if not validation or not outer_test:
        raise ValueError("validation and outer-test partitions must be non-empty")
    return train, validation, outer_test, inner_fold


class MergedTileDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, root: Path, names: Sequence[str]) -> None:
        self.root = root
        self.names = tuple(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        name = self.names[index]
        with Image.open(self.root / "images" / name) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(self.root / "masks" / name) as image:
            mask = np.asarray(image, dtype=np.uint8).copy()
        if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"unexpected merged tile shape: {name}: {rgb.shape}, {mask.shape}")
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        image_tensor = (image_tensor - MEAN) / STD
        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask.astype(np.int64, copy=False)),
            "name": name,
            "source_group": panel_name(name),
        }


def _make_loader(
    root: Path,
    names: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader[dict[str, Tensor | str]]:
    workers = min(num_workers, os.cpu_count() or 1)
    return DataLoader(
        MergedTileDataset(root, names),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _autocast(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Predictor(Protocol):
    selected_epoch: int
    checkpoint_records: list[dict[str, Any]]

    def probability(self, images: Tensor) -> Tensor: ...

    def close(self) -> None: ...


class SMPPredictor:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        model_family: str,
    ) -> None:
        source_path = WORKSPACE_ROOT / model_family / "src"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        arguments = payload["args"]
        if model_family == "unet":
            from unet_model import backbone_profile, build_resunet

            profile = backbone_profile(str(arguments["backbone"]))
            encoder = str(arguments.get("encoder") or profile.encoder)
            model = build_resunet(
                encoder=encoder,
                encoder_weights=None,
                num_classes=3,
            )
        elif model_family == "segformer":
            from segformer_model import backbone_profile, build_segformer

            profile = backbone_profile(str(arguments["backbone"]))
            encoder = str(arguments.get("encoder") or profile.encoder)
            model = build_segformer(
                encoder=encoder,
                encoder_weights=None,
                decoder_channels=profile.decoder_channels,
                num_classes=3,
            )
        else:
            raise ValueError(f"unknown SMP model family: {model_family}")
        self.model = model.to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = device
        self.selected_epoch = int(payload["epoch"])
        self.checkpoint_records = [
            {
                "role": "joint crack/craquelure softmax",
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "selected_epoch": self.selected_epoch,
                "source_manifest_hash": payload.get("manifest_hash"),
            }
        ]

    @torch.no_grad()
    def probability(self, images: Tensor) -> Tensor:
        images = images.to(self.device, non_blocking=True)
        with _autocast(self.device):
            probabilities = torch.softmax(self.model(images), dim=1)
        return probabilities[:, 1].float() + probabilities[:, 2].float()

    def close(self) -> None:
        del self.model


class SAM2SACPredictor:
    def __init__(self, checkpoint: Path, base_checkpoint: Path, device: torch.device) -> None:
        from sam2_sac.h0_core import (
            NativeSAM2MaskDecoder,
            freeze_except_layer_norm,
            load_trainable_state_dict,
        )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("stage") != "union":
            raise ValueError(f"expected SAM2-SAC union checkpoint: {checkpoint}")
        base_hash = _sha256(base_checkpoint)
        if payload.get("base_checkpoint_sha256") != base_hash:
            raise ValueError("SAM2-SAC base checkpoint hash mismatch")
        self.model = NativeSAM2MaskDecoder(
            checkpoint=base_checkpoint,
            config=str(payload.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")),
            image_size=int(payload.get("image_size", IMAGE_SIZE)),
            device=device,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for component in (
            self.model.model.image_encoder,
            self.model.model.sam_prompt_encoder,
            self.model.model.sam_mask_decoder,
        ):
            freeze_except_layer_norm(component)
        load_trainable_state_dict(self.model, payload["adaptation_state"])
        self.model.eval()
        self.device = device
        self.selected_epoch = int(payload["epoch"])
        self.checkpoint_records = [
            {
                "role": "union",
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "selected_epoch": self.selected_epoch,
                "source_manifest_hash": payload.get("manifest_hash"),
                "base_checkpoint": str(base_checkpoint.resolve()),
                "base_checkpoint_sha256": base_hash,
            }
        ]

    @torch.no_grad()
    def probability(self, images: Tensor) -> Tensor:
        images = images.to(self.device, non_blocking=True)
        with _autocast(self.device):
            logits = self.model(images)
        return torch.sigmoid(logits[:, 0]).float()

    def close(self) -> None:
        del self.model


class SAM2AdapterPredictor:
    def __init__(
        self,
        checkpoints: dict[str, Path],
        base_checkpoint: Path,
        device: torch.device,
    ) -> None:
        from sam2_adapter.adapter_model import SAM2AdapterMaskDecoder
        from sam2_adapter.h0_core import load_trainable_state_dict

        base_hash = _sha256(base_checkpoint)
        self.models: dict[str, nn.Module] = {}
        self.device = device
        records: list[dict[str, Any]] = []
        epochs: list[int] = []
        for expert in ("crack", "craquelure"):
            path = checkpoints[expert]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("expert") != expert:
                raise ValueError(f"SAM2-Adapter expert mismatch: {path}")
            if payload.get("base_checkpoint_sha256") != base_hash:
                raise ValueError("SAM2-Adapter base checkpoint hash mismatch")
            adapter = payload.get("adapter", {})
            model = SAM2AdapterMaskDecoder(
                checkpoint=base_checkpoint,
                config=str(payload.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")),
                image_size=int(payload.get("image_size", IMAGE_SIZE)),
                device=device,
                scale_factor=int(adapter.get("scale_factor", 32)),
                highpass_rate=float(adapter.get("highpass_rate", 0.25)),
            )
            load_trainable_state_dict(model, payload["adaptation_state"])
            model.eval()
            self.models[expert] = model
            epoch = int(payload["epoch"])
            epochs.append(epoch)
            records.append(
                {
                    "role": expert,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "selected_epoch": epoch,
                    "source_manifest_hash": payload.get("manifest_hash"),
                    "base_checkpoint": str(base_checkpoint.resolve()),
                    "base_checkpoint_sha256": base_hash,
                }
            )
        self.selected_epoch = max(epochs)
        self.checkpoint_records = records

    @torch.no_grad()
    def probability(self, images: Tensor) -> Tensor:
        images = images.to(self.device, non_blocking=True)
        probabilities = []
        with _autocast(self.device):
            for expert in ("crack", "craquelure"):
                probabilities.append(torch.sigmoid(self.models[expert](images)[:, 0]).float())
        return torch.maximum(probabilities[0], probabilities[1])

    def close(self) -> None:
        self.models.clear()


def _checkpoint_paths(spec: ModelSpec, fold: int) -> dict[str, Path]:
    root = WORKSPACE_ROOT / spec.project / "runs" / spec.source_experiment / "5fold"
    return {
        expert: root / expert / f"fold{fold}" / "artifacts" / "checkpoints" / "best.pt"
        for expert in spec.checkpoint_experts
    }


def _build_predictor(
    spec: ModelSpec, fold: int, *, base_checkpoint: Path, device: torch.device
) -> Predictor:
    paths = _checkpoint_paths(spec, fold)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source checkpoints: {missing}")
    if spec.kind in {"unet", "segformer"}:
        return SMPPredictor(
            next(iter(paths.values())),
            device,
            model_family=spec.kind,
        )
    if spec.kind == "sam2_sac":
        return SAM2SACPredictor(next(iter(paths.values())), base_checkpoint, device)
    if spec.kind == "sam2_adapter":
        return SAM2AdapterPredictor(paths, base_checkpoint, device)
    raise ValueError(f"unknown model kind: {spec.kind}")


def _merge_counts(
    current: tuple[int, int, int] | None, update: tuple[int, int, int]
) -> tuple[int, int, int]:
    if current is None:
        return update
    return tuple(a + b for a, b in zip(current, update, strict=True))


def _restored_rgb(image: Tensor) -> np.ndarray:
    restored = (image.detach().cpu() * STD + MEAN).clamp(0.0, 1.0)
    return restored.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()


def _evaluate(
    predictor: Predictor,
    loader: DataLoader[dict[str, Tensor | str]],
    *,
    layout: RunLayout | None,
    split: str,
) -> dict[str, Any]:
    panel_counts: dict[str, tuple[int, int, int]] = {}
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    valid_count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            masks = batch["mask"]
            if not isinstance(images, Tensor) or not isinstance(masks, Tensor):
                raise TypeError("image and mask batches must be tensors")
            probability = predictor.probability(images).detach().cpu().clamp_(1e-7, 1 - 1e-7)
            raw_masks = masks.numpy()
            names = [str(value) for value in batch["name"]]  # type: ignore[index]
            groups = [str(value) for value in batch["source_group"]]  # type: ignore[index]
            for index, (name, group) in enumerate(zip(names, groups, strict=True)):
                target, valid = merged_binary_target(raw_masks[index])
                current_probability = probability[index].numpy()
                prediction = (current_probability >= FIXED_THRESHOLD) & valid
                target = target & valid
                valid_probability = torch.from_numpy(current_probability[valid])
                valid_target = torch.from_numpy(target[valid].astype(np.float32, copy=False))
                loss_sum += float(
                    F.binary_cross_entropy(valid_probability, valid_target, reduction="sum").item()
                )
                valid_count += int(valid.sum())
                metric = binary_metric_row(target, prediction)
                count = (int(metric["tp"]), int(metric["fp"]), int(metric["fn"]))
                panel_counts[group] = _merge_counts(panel_counts.get(group), count)
                if layout is not None:
                    row = save_qualitative_example(
                        layout,
                        image_id=name,
                        input_rgb=_restored_rgb(images[index]),
                        target=target,
                        prediction=prediction,
                        target_class="craquelure (merged crack+craquelure)",
                    )
                    rows.append(row)
                else:
                    rows.append(
                        {
                            "image": name,
                            "split": split,
                            "source_group": group,
                            "target_class": "craquelure (merged crack+craquelure)",
                            **metric,
                        }
                    )
    summary = binary_summary(panel_counts)
    return {
        "loss": loss_sum / valid_count if valid_count else None,
        "loss_definition": "binary cross entropy on raw mask labels {0,1}; labels {2,3,4,5,255} excluded",
        "valid_pixels": valid_count,
        **summary,
        "panel_counts": {key: list(value) for key, value in sorted(panel_counts.items())},
        "per_image_rows": rows,
    }


def _history_rows(spec: ModelSpec, fold: int) -> list[dict[str, float | int]]:
    root = WORKSPACE_ROOT / spec.project / "runs" / spec.source_experiment / "5fold"
    by_epoch: dict[int, list[dict[str, float]]] = defaultdict(list)
    for expert in spec.history_experts:
        path = root / expert / f"fold{fold}" / "metrics" / "epochs.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                epoch = int(row["epoch"])
                by_epoch[epoch].append(
                    {
                        key: float(row[key])
                        for key in (
                            "train_loss",
                            "val_loss",
                            "f1",
                            "precision",
                            "recall",
                            "iou",
                            "learning_rate",
                        )
                        if row.get(key) not in (None, "")
                    }
                )
    combined: list[dict[str, float | int]] = []
    for epoch, rows in sorted(by_epoch.items()):
        record: dict[str, float | int] = {"epoch": epoch}
        for key in (
            "train_loss",
            "val_loss",
            "f1",
            "precision",
            "recall",
            "iou",
            "learning_rate",
        ):
            values = [row[key] for row in rows if key in row]
            if values:
                record[key] = float(sum(values) / len(values))
        combined.append(record)
    if not combined:
        raise ValueError(f"source training history is empty for {spec.key} fold{fold}")
    return combined


def _write_source_history(
    spec: ModelSpec, fold: int, layout: RunLayout
) -> tuple[SummaryWriter, EpochReporter]:
    writer = SummaryWriter(log_dir=str(layout.tensorboard))
    reporter = EpochReporter(layout.metrics / "epochs.csv", writer)
    for row in _history_rows(spec, fold):
        reporter.record(
            int(row["epoch"]),
            train_loss=float(row["train_loss"]),
            validation_loss=float(row["val_loss"]),
            f1=float(row["f1"]) if "f1" in row else None,
            precision=float(row["precision"]) if "precision" in row else None,
            recall=float(row["recall"]) if "recall" in row else None,
            iou=float(row["iou"]) if "iou" in row else None,
            learning_rate=float(row["learning_rate"]),
        )
    return writer, reporter


def _write_outer_rows(layout: RunLayout, rows: list[dict[str, Any]]) -> None:
    columns = (
        "image",
        "split",
        "source_group",
        "target_class",
        "f1",
        "precision",
        "recall",
        "iou",
        "tp",
        "fp",
        "fn",
        "gt_pixels",
        "pred_pixels",
        "error_reason",
    )
    with (layout.metrics / "per_image_outer_test.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["image"])))


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_fold_metadata(
    layout: RunLayout,
    *,
    spec: ModelSpec,
    dataset: Path,
    source_dataset: Path,
    fold: int,
    inner_fold: int,
    train: Sequence[str],
    validation: Sequence[str],
    outer_test: Sequence[str],
    predictor: Predictor,
) -> None:
    merged_manifest = _read_json(dataset / "manifest.json")
    source_manifest = _read_json(source_dataset / "manifest.json")
    write_json(
        layout.config / "dataset.json",
        {
            "dataset_root": str(dataset.resolve()),
            "merged_manifest_hash": merged_manifest.get("manifest_sha256"),
            "merged_mask_manifest_hash": merged_manifest.get("mask_manifest_sha256"),
            "image_manifest_hash": merged_manifest.get("image_manifest_sha256"),
            "source_split_dataset": str(source_dataset.resolve()),
            "source_manifest_hash": source_manifest.get("manifest_sha256"),
            "source_image_manifest_hash": source_manifest.get("image_manifest_sha256"),
            "image_hash_match": merged_manifest.get("image_manifest_sha256")
            == source_manifest.get("image_manifest_sha256"),
            "outer_fold": fold,
            "inner_fold": inner_fold,
            "train": list(train),
            "validation": list(validation),
            "outer_test": list(outer_test),
            "ground_truth_rule": "raw_mask == 1",
            "scored_background_rule": "raw_mask == 0",
            "ignored_labels": [2, 3, 4, 5, 255],
        },
    )
    write_json(
        layout.config / "model.json",
        {
            "model_key": spec.key,
            "source_project": spec.project,
            "source_experiment": spec.source_experiment,
            "checkpoints": predictor.checkpoint_records,
        },
    )
    write_json(
        layout.config / "run.json",
        {
            "experiment_id": spec.experiment_id,
            "purpose": "checkpoint-frozen merged-label OOF retest; no training",
            "outer_fold": fold,
            "inner_fold": inner_fold,
            "checkpoint_selection": "retained from source validation; merged labels excluded",
            "threshold": FIXED_THRESHOLD,
            "threshold_source": "pre-registered merged dataset evaluation contract",
            "threshold_tuning_on_merged_validation_or_outer_test": False,
            "probability_rule": {
                "unet": "softmax P(crack) + P(craquelure)",
                "segformer": "softmax P(crack) + P(craquelure)",
                "sam2_sac": "sigmoid(union logit)",
                "sam2_adapter": "max(sigmoid(crack logit), sigmoid(craquelure logit))",
            }[spec.kind],
            "training_curve_source": "copied as scalar values from source epochs.csv; no retraining",
            "command": [sys.executable, *sys.argv],
        },
    )
    write_json(
        layout.config / "environment.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "git_revision": _git_revision(),
        },
    )
    (layout.logs / "train.log").write_text(
        "retest_only=true\ntraining_performed=false\n", encoding="utf-8"
    )


def _validate_dataset_pair(dataset: Path, source_dataset: Path) -> None:
    merged = _read_json(dataset / "manifest.json")
    source = _read_json(source_dataset / "manifest.json")
    expected_ids = {
        "background": 0,
        "craquelure": 1,
        "loss": 2,
        "shrinkage": 3,
        "flaking": 4,
        "stain": 5,
    }
    if merged.get("label_contract", {}).get("class_ids") != expected_ids:
        raise ValueError("target dataset is not the expected merged craquelure contract")
    if merged.get("image_manifest_sha256") != source.get("image_manifest_sha256"):
        raise ValueError("merged and source datasets do not contain byte-identical images")


def _update_experiment_info(
    spec: ModelSpec,
    dataset: Path,
    source_dataset: Path,
    completed_folds: Sequence[int],
) -> None:
    root = WORKSPACE_ROOT / spec.project / "runs" / spec.experiment_id
    write_json(
        root / "info" / "experiment.json",
        {
            "schema_version": 1,
            "experiment_id": spec.experiment_id,
            "layout": "5fold",
            "fold_count": 5,
            "experts": ["craquelure"],
            "folds": {"craquelure": list(completed_folds)},
            "runs": [f"5fold/craquelure/fold{fold}" for fold in completed_folds],
            "dataset_root": str(dataset.resolve()),
            "source_split_dataset": str(source_dataset.resolve()),
            "source_experiment": spec.source_experiment,
            "evaluation": "leak-free source-fold OOF retest on merged crack+craquelure labels",
            "threshold": FIXED_THRESHOLD,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _aggregate_folds(spec: ModelSpec, folds: Sequence[int]) -> dict[str, Any]:
    root = WORKSPACE_ROOT / spec.project / "runs" / spec.experiment_id
    total = [0, 0, 0]
    panel_counts: dict[str, tuple[int, int, int]] = {}
    total_loss_numerator = 0.0
    total_valid = 0
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        run = root / "5fold" / "craquelure" / f"fold{fold}"
        metrics = _read_json(run / "metrics" / "outer_test_metrics.json")
        dataset_record = _read_json(run / "config" / "dataset.json")
        micro = metrics["tile_micro"]
        fold_records.append(
            {
                "fold": fold,
                "f1": micro.get("mf1"),
                "precision": micro.get("mprecision"),
                "recall": micro.get("mrecall"),
                "iou": micro.get("miou"),
                "loss": metrics.get("loss"),
                "tile_count": len(dataset_record["outer_test"]),
            }
        )
        total[0] += int(micro["tp"])
        total[1] += int(micro["fp"])
        total[2] += int(micro["fn"])
        for group, values in metrics["panel_counts"].items():
            panel_counts[group] = _merge_counts(
                panel_counts.get(group), tuple(int(value) for value in values)
            )
        valid_pixels = metrics.get("valid_pixels")
        if isinstance(valid_pixels, int) and isinstance(metrics.get("loss"), (int, float)):
            total_valid += valid_pixels
            total_loss_numerator += float(metrics["loss"]) * valid_pixels
    panel = binary_summary(panel_counts)["expert_panel_macro"]
    aggregate = {
        "scope": "five-fold source-split out-of-fold retest; each of 929 images scored once",
        "model": spec.key,
        "threshold": FIXED_THRESHOLD,
        "tile_micro": counts_summary(*total),
        "expert_panel_macro": panel,
        "folds": fold_records,
        "outer_tile_count": sum(record["tile_count"] for record in fold_records),
    }
    if total_valid:
        aggregate["loss"] = total_loss_numerator / total_valid
    write_json(root / "info" / "oof_summary.json", aggregate)
    return aggregate


def run_model(
    spec: ModelSpec,
    *,
    dataset: Path,
    source_dataset: Path,
    base_checkpoint: Path,
    folds: Sequence[int],
    device: torch.device,
    num_workers: int,
    allow_existing: bool,
) -> dict[str, Any]:
    experiment_root = WORKSPACE_ROOT / spec.project / "runs" / spec.experiment_id
    completed: list[int] = []
    for fold in folds:
        run_root = experiment_root / "5fold" / "craquelure" / f"fold{fold}"
        completed_marker = run_root / "reports" / "index.html"
        if run_root.exists() and any(run_root.iterdir()):
            if allow_existing and completed_marker.is_file():
                print(f"{spec.key} fold{fold}: keeping completed run", flush=True)
                completed.append(fold)
                continue
            raise FileExistsError(f"refusing to overwrite existing retest run: {run_root}")
        train, validation, outer_test, inner_fold = source_oof_partitions(
            source_dataset, outer_fold=fold
        )
        layout = RunLayout.create(run_root)
        predictor = _build_predictor(
            spec, fold, base_checkpoint=base_checkpoint, device=device
        )
        _write_fold_metadata(
            layout,
            spec=spec,
            dataset=dataset,
            source_dataset=source_dataset,
            fold=fold,
            inner_fold=inner_fold,
            train=train,
            validation=validation,
            outer_test=outer_test,
            predictor=predictor,
        )
        writer, reporter = _write_source_history(spec, fold, layout)
        finalized = False
        try:
            print(
                f"{spec.key} fold{fold}: validation={len(validation)} outer_test={len(outer_test)}",
                flush=True,
            )
            validation_metrics = _evaluate(
                predictor,
                _make_loader(
                    dataset,
                    validation,
                    batch_size=spec.batch_size,
                    num_workers=num_workers,
                ),
                layout=layout,
                split="validation",
            )
            outer_metrics = _evaluate(
                predictor,
                _make_loader(
                    dataset,
                    outer_test,
                    batch_size=spec.batch_size,
                    num_workers=num_workers,
                ),
                layout=None,
                split="outer_test",
            )
            _write_outer_rows(layout, outer_metrics.pop("per_image_rows"))
            validation_rows = validation_metrics.pop("per_image_rows")
            write_json(layout.metrics / "selected_validation_metrics.json", validation_metrics)
            outer_record = {
                "scope": "outer_test_not_used_for_checkpoint_or_threshold_selection",
                "model": spec.key,
                "selected_checkpoints": predictor.checkpoint_records,
                "threshold": FIXED_THRESHOLD,
                "threshold_scope": "pre-registered constant; merged GT excluded",
                **outer_metrics,
            }
            finalize_reporting(
                layout,
                writer=writer,
                reporter=reporter,
                validation_rows=validation_rows,
                selected_epoch=predictor.selected_epoch,
                outer_test_metrics=outer_record,
            )
            finalized = True
            micro = outer_record["tile_micro"]
            append_log(
                layout,
                f"completed outer_f1={micro.get('mf1')} outer_iou={micro.get('miou')}",
            )
            completed.append(fold)
            _update_experiment_info(spec, dataset, source_dataset, completed)
        finally:
            if not finalized:
                reporter.close()
                writer.close()
            predictor.close()
            del predictor
            torch.cuda.empty_cache()
    return _aggregate_folds(spec, completed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*MODEL_SPECS, "all"),
        default=["all"],
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)
    if any(fold not in range(5) for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        parser.error("--folds must contain unique values from 0 through 4")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.resolve()
    source_dataset = args.source_dataset.resolve()
    base_checkpoint = args.sam2_checkpoint.resolve()
    _validate_dataset_pair(dataset, source_dataset)
    requested = list(MODEL_SPECS) if "all" in args.models else list(args.models)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if any(MODEL_SPECS[key].kind.startswith("sam2") for key in requested) and not base_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 base checkpoint is missing: {base_checkpoint}")
    torch.set_float32_matmul_precision("high")
    summaries = {}
    for key in requested:
        summaries[key] = run_model(
            MODEL_SPECS[key],
            dataset=dataset,
            source_dataset=source_dataset,
            base_checkpoint=base_checkpoint,
            folds=args.folds,
            device=device,
            num_workers=args.num_workers,
            allow_existing=args.allow_existing,
        )
        print(json.dumps({key: summaries[key]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
