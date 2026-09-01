import torch

from monument_da_sam3.data import make_multilabel_targets


def test_raw_labels_map_to_independent_targets_and_valid_masks() -> None:
    raw = torch.tensor([[0, 1, 2], [3, 4, 5]])
    targets, valid, present = make_multilabel_targets(raw)
    assert targets[:, 0, 0].tolist() == [0.0, 0.0]
    assert valid[:, 0, 0].tolist() == [True, True]
    assert targets[:, 0, 1].tolist() == [1.0, 0.0]
    assert valid[:, 0, 1].tolist() == [True, False]
    assert targets[:, 0, 2].tolist() == [0.0, 1.0]
    assert valid[:, 0, 2].tolist() == [False, True]
    assert not valid[:, 1].any()
    assert present.tolist() == [True, True]


def test_background_only_tile_is_retained_as_valid_negative() -> None:
    targets, valid, present = make_multilabel_targets(torch.zeros(4, 4, dtype=torch.long))
    assert targets.sum() == 0
    assert valid.all()
    assert not present.any()
