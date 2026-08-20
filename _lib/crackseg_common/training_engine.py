"""Memory-aware training primitives shared by training entry points."""
from __future__ import annotations

from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import DataLoader


def _expected_group_samples(
    loader: DataLoader,
    batch_index: int,
    group_batches: int,
) -> int | None:
    """Return the sample count in a group for ordinary fixed-size loaders."""

    if loader.batch_size is None:
        return None
    try:
        sample_count = len(loader.sampler)
    except (TypeError, AttributeError):
        return None
    final_batch_size = sample_count % loader.batch_size or loader.batch_size
    final_batch_index = len(loader) - 1
    return sum(
        final_batch_size if index == final_batch_index else loader.batch_size
        for index in range(batch_index, batch_index + group_batches)
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    """Train one epoch, accumulating micro-batches without changing mean loss."""

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    model.train()
    totals = defaultdict(float)
    samples = 0
    batch_count = len(loader)
    optimizer.zero_grad(set_to_none=True)
    group_samples: int | None = None
    group_batches = 1
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if batch_index % gradient_accumulation_steps == 0:
            group_batches = min(
                gradient_accumulation_steps,
                batch_count - batch_index,
            )
            group_samples = _expected_group_samples(loader, batch_index, group_batches)
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, parts = criterion(model(images), masks)
        else:
            loss, parts = criterion(model(images), masks)
        batch_size = images.size(0)
        loss_scale = (
            batch_size / group_samples
            if group_samples is not None
            else 1.0 / group_batches
        )
        backward_loss = loss * loss_scale
        if scaler is not None:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()
        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == batch_count
        )
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        samples += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in parts.items():
            totals[name] += float(value) * batch_size
    return {name: value / samples for name, value in totals.items()}
