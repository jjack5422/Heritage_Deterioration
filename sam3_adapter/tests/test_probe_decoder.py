"""Behavioral tests for the common A/B probe."""

import torch

from sam3_adapter.probe_decoder import SharedFpnProbe


def test_probe_accepts_native_sam2_and_sam3_pyramids() -> None:
    probe = SharedFpnProbe()
    for sizes in ((256, 128, 64), (288, 144, 72)):
        features = [torch.randn(1, 256, size, size) for size in sizes]
        assert probe(features).shape == (1, 1, sizes[0], sizes[0])


def test_probe_initialization_is_seed_reproducible() -> None:
    torch.manual_seed(42)
    first = SharedFpnProbe().state_dict()
    torch.manual_seed(42)
    second = SharedFpnProbe().state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)
