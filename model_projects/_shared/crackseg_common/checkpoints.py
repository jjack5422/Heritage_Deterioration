"""Model-neutral helpers for trainable adaptation checkpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn


def trainable_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Return CPU copies of precisely the trainable parameters."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Load trainable parameters while rejecting architecture drift."""

    parameters = dict(module.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    actual = set(state)
    if expected != actual:
        raise ValueError(
            "adaptation checkpoint parameter mismatch; "
            f"missing={sorted(expected - actual)[:5]}, "
            f"unexpected={sorted(actual - expected)[:5]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def model_parameter_counts(module: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""

    parameters: Iterable[nn.Parameter] = module.parameters()
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
