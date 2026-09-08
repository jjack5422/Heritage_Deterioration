import torch
import pytest

from dual_adapter_sam3.metrics import BoundaryConfusion, MultilabelConfusion


def test_metrics_ignore_invalid_pixels_and_macro_excludes_background() -> None:
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]], [[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[0, 0, 0, 1] = False
    counts = MultilabelConfusion.empty()
    counts.update(logits, targets, valid)
    metrics = counts.compute()
    assert metrics["per_class"]["crack_craquelure"]["f1"] == 1.0
    assert metrics["per_class"]["loss"]["f1"] == 2 / 3
    assert metrics["macro"]["f1"] == (1.0 + 2 / 3) / 2


def test_boundary_f1_matches_within_two_pixels_and_rejects_distant_edges() -> None:
    target = torch.zeros(1, 9, 9, dtype=torch.bool)
    target[:, 4, 1] = True
    near = torch.zeros_like(target)
    near[:, 4, 3] = True
    far = torch.zeros_like(target)
    far[:, 4, 7] = True
    valid = torch.ones_like(target)

    accepted = BoundaryConfusion.empty(tolerance=2)
    accepted.update(near, target, valid)
    rejected = BoundaryConfusion.empty(tolerance=2)
    rejected.update(far, target, valid)

    assert accepted.compute()["f1"] == pytest.approx(1.0)
    assert rejected.compute()["f1"] == pytest.approx(0.0)
