"""Small runtime helpers shared by the independent Adapter trainer."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sam2_adapter.data import H0DataPlan, H0TileDataset


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET = WORKSPACE_ROOT / "datasets" / "dataset_clean_v2_merged_craquelure"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_large.pt"


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


def _batch_tensor(batch: dict[str, Tensor | str], name: str, device: torch.device) -> Tensor:
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] should be a Tensor")
    return value.to(device, non_blocking=True)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "pillow",
        "tensorboard",
        "torch",
        "torchvision",
        "hydra-core",
        "omegaconf",
    )
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
