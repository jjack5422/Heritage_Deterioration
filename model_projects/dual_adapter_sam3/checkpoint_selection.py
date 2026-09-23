"""Validation-only checkpoint selection for craquelure-priority multitask training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LOSS_F1_TOLERANCE = 0.03


@dataclass(frozen=True)
class ValidationCandidate:
    path: Path
    stage: str
    epoch: int
    craquelure_f1: float
    boundary_f1: float
    loss_f1: float

    @property
    def craquelure_score(self) -> float:
        return 0.5 * (self.craquelure_f1 + self.boundary_f1)


def update_pareto_front(
    candidates: tuple[ValidationCandidate, ...],
    candidate: ValidationCandidate,
) -> tuple[tuple[ValidationCandidate, ...], tuple[ValidationCandidate, ...], bool]:
    """Keep the loss-F1/craquelure-score frontier needed for final selection."""

    dominated = any(
        existing.loss_f1 >= candidate.loss_f1
        and existing.craquelure_score >= candidate.craquelure_score
        for existing in candidates
    )
    if dominated:
        return candidates, (), False
    removed = tuple(
        existing
        for existing in candidates
        if candidate.loss_f1 >= existing.loss_f1
        and candidate.craquelure_score >= existing.craquelure_score
    )
    retained = tuple(existing for existing in candidates if existing not in removed)
    updated = tuple(sorted((*retained, candidate), key=lambda item: item.epoch))
    return updated, removed, True


def select_constrained_joint(
    candidates: tuple[ValidationCandidate, ...],
    *,
    loss_f1_tolerance: float = LOSS_F1_TOLERANCE,
) -> ValidationCandidate:
    """Maximize craquelure score while retaining near-best validation loss F1."""

    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    if loss_f1_tolerance < 0:
        raise ValueError("loss F1 tolerance must be non-negative")
    best_loss_f1 = max(candidate.loss_f1 for candidate in candidates)
    minimum_loss_f1 = best_loss_f1 - loss_f1_tolerance
    eligible = [candidate for candidate in candidates if candidate.loss_f1 >= minimum_loss_f1]
    return max(
        eligible,
        key=lambda candidate: (
            candidate.craquelure_score,
            candidate.loss_f1,
            -candidate.epoch,
        ),
    )


__all__ = [
    "LOSS_F1_TOLERANCE",
    "ValidationCandidate",
    "select_constrained_joint",
    "update_pareto_front",
]
