from pathlib import Path

from dual_adapter_sam3.checkpoint_selection import (
    ValidationCandidate,
    select_constrained_joint,
    update_pareto_front,
)


def _candidate(epoch: int, craquelure: float, boundary: float, loss: float) -> ValidationCandidate:
    return ValidationCandidate(Path(f"epoch_{epoch}.pt"), "stage2", epoch, craquelure, boundary, loss)


def test_constrained_selection_rejects_craquelure_peak_when_loss_falls_too_far() -> None:
    candidates = (
        _candidate(1, 0.70, 0.70, 0.40),
        _candidate(2, 0.68, 0.68, 0.48),
        _candidate(3, 0.66, 0.66, 0.50),
    )

    selected = select_constrained_joint(candidates, loss_f1_tolerance=0.03)

    assert selected.epoch == 2


def test_pareto_front_discards_dominated_candidates_without_losing_selection() -> None:
    candidates: tuple[ValidationCandidate, ...] = ()
    removed_epochs: list[int] = []
    for candidate in (
        _candidate(1, 0.40, 0.40, 0.40),
        _candidate(2, 0.45, 0.45, 0.42),
        _candidate(3, 0.50, 0.50, 0.38),
    ):
        candidates, removed, accepted = update_pareto_front(candidates, candidate)
        assert accepted
        removed_epochs.extend(item.epoch for item in removed)

    assert removed_epochs == [1]
    assert {candidate.epoch for candidate in candidates} == {2, 3}
    assert select_constrained_joint(candidates).epoch == 2
