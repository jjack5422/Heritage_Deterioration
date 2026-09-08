import torch

from dual_adapter_sam3.losses import (
    hybrid_multilabel_objective,
    masked_positive_dice,
    masked_sam2_bce_dice,
    masked_weighted_focal,
    multilabel_objective,
    router_z_loss,
    sam2_craquelure_hybrid_segmentation_loss,
)
from sam2_adapter.h0_core import binary_bce_dice_loss


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


def test_masked_sam2_bce_dice_matches_sam2_adapter_objective() -> None:
    logits = torch.tensor([[[[0.5, -0.4], [1.2, -2.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 255.0]]]])
    valid = target != 255.0

    expected = binary_bce_dice_loss(logits, target, ignore_value=255)
    actual = masked_sam2_bce_dice(logits, target.clamp_max(1.0), valid)

    torch.testing.assert_close(actual, expected)


def test_hybrid_objective_changes_only_craquelure_pixel_loss() -> None:
    logits = torch.tensor(
        [[[[0.2, -0.7], [1.1, -0.4]], [[-0.3, 0.8], [0.5, -1.2]]]],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )
    valid = torch.ones_like(targets, dtype=torch.bool)
    present = torch.ones(1, 2, dtype=torch.bool)
    presence_logits = torch.tensor([[0.4, -0.2]], requires_grad=True)

    original = multilabel_objective(
        logits, presence_logits, targets, valid, present
    )
    actual = sam2_craquelure_hybrid_segmentation_loss(
        logits, presence_logits, targets, valid, present
    )
    craquelure = masked_sam2_bce_dice(
        logits[:, 0], targets[:, 0], valid[:, 0]
    ) + 0.1 * original.presence[0]
    expected = 0.5 * (craquelure + original.per_class[1])

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert presence_logits.grad is not None and torch.isfinite(presence_logits.grad).all()


def test_hybrid_multilabel_objective_keeps_equal_class_weights_and_router_terms() -> None:
    logits = torch.tensor(
        [[[[0.2, -0.7]], [[-0.3, 0.8]]]], requires_grad=True
    )
    targets = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    present = torch.ones(1, 2, dtype=torch.bool)
    presence_logits = torch.tensor([[0.4, -0.2]], requires_grad=True)
    router_logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    soft = router_logits.softmax(-1)
    hard = torch.tensor([[[1.0, 0.0]]])

    output = hybrid_multilabel_objective(
        logits,
        presence_logits,
        targets,
        valid,
        present,
        routing=[(router_logits, soft, hard)],
    )

    torch.testing.assert_close(output.segmentation, output.per_class.mean())
    torch.testing.assert_close(
        output.total,
        output.segmentation + 0.01 * output.balance + 0.001 * output.router_z,
    )
