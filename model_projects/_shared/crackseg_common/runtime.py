"""Model-neutral runtime and reproducibility helpers."""

from __future__ import annotations

import contextlib
import hashlib
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def batch_tensor(batch: dict[str, Any], name: str, device: torch.device) -> Tensor:
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] should be a Tensor")
    return value.to(device, non_blocking=True)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def package_versions() -> dict[str, str | None]:
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


def git_revision(repository_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
