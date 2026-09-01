import torch

from monument_da_sam3.metrics import MultilabelConfusion


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
