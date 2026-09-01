import torch

from monument_da_sam3.losses import masked_positive_dice, masked_weighted_focal, multilabel_objective, router_z_loss


def test_weighted_focal_has_exact_two_to_one_positive_penalty() -> None:
    positive_logit = torch.tensor([0.0], requires_grad=True)
    negative_logit = torch.tensor([0.0], requires_grad=True)
    valid = torch.tensor([True])
    positive = masked_weighted_focal(positive_logit, torch.ones(1), valid)
    negative = masked_weighted_focal(negative_logit, torch.zeros(1), valid)
    assert torch.allclose(positive, 2.0 * negative)
    positive.backward()
    negative.backward()
    assert torch.allclose(positive_logit.grad.abs(), 2.0 * negative_logit.grad.abs())


def test_empty_positive_dice_is_graph_connected_zero() -> None:
    logits = torch.randn(2, 3, 3, requires_grad=True)
    loss = masked_positive_dice(logits, torch.zeros_like(logits), torch.ones_like(logits, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_ignored_logits_do_not_change_segmentation_loss() -> None:
    logits = torch.zeros(1, 2, 2, 2)
    targets = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[:, :, 0, 0] = False
    present = torch.zeros(1, 2, dtype=torch.bool)
    first = multilabel_objective(logits, torch.zeros(1, 2), targets, valid, present).segmentation
    logits[:, :, 0, 0] = 100.0
    second = multilabel_objective(logits, torch.zeros(1, 2), targets, valid, present).segmentation
    assert torch.allclose(first, second)


def test_router_z_loss_uses_raw_logits() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = torch.logsumexp(logits, dim=-1).square().mean()
    assert torch.allclose(router_z_loss(logits), expected)
