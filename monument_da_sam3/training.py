"""Two-stage optimizer configuration and runtime training gates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .da_moe import router_temperature


@dataclass(frozen=True)
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer_and_scheduler(model: nn.Module, *, stage: str, epochs: int) -> OptimizerBundle:
    names = model.configure_stage(stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or not names:
        raise RuntimeError(f"{stage} has no trainable adaptation parameters")
    learning_rate = 5e-4 if stage == "stage1" else 1e-4
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return OptimizerBundle(optimizer, scheduler)


def set_epoch_router_temperature(model: nn.Module, epoch: int) -> float:
    temperature = router_temperature(epoch)
    model.set_router_temperature(temperature)
    return temperature


def assert_finite_gradients(model: nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"frozen parameter unexpectedly received a gradient: {name}")
            continue
        if parameter.grad is None:
            raise RuntimeError(f"trainable parameter has no gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")
        norms[name] = float(parameter.grad.detach().norm())
    return norms


def build_stage2_hard_pool(
    train_ids: Iterable[str],
    both_class_ids: Iterable[str],
    per_tile_losses: dict[str, float],
) -> tuple[str, ...]:
    train = set(train_ids)
    if set(per_tile_losses) != train:
        raise ValueError("hard-pool losses must contain exactly the training tile IDs")
    if not set(both_class_ids).issubset(train):
        raise ValueError("hard-pool both-class IDs include validation/test data")
    ranked = sorted(train, key=lambda name: (-per_tile_losses[name], name))
    count = max(1, math.ceil(len(ranked) * 0.25))
    pool = set(ranked[:count]) | set(both_class_ids)
    if not pool:
        raise RuntimeError("stage2 hard pool is empty")
    return tuple(sorted(pool))
