"""Small runtime helpers shared by the independent Adapter trainer."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from model_projects.sam2_adapter.data import H0DataPlan, H0TileDataset


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DATASET = REPOSITORY_ROOT / "datasets" / "dataset_clean_v2_merged_craquelure"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_large.pt"




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


