import torch
from torch import nn

from monument_da_sam3.da_moe import DaMoeFfn, router_temperature


def test_top_two_routing_and_output_contract() -> None:
    module = DaMoeFfn(nn.Linear(8, 16), nn.Linear(16, 8), rank=2)
    tokens = torch.randn(2, 5, 8)
    visual = torch.randn(2, 9, 8)
    concept = torch.randn(2, 8)
    output, diagnostics = module(tokens, visual, concept, temperature=2.0)
    assert output.shape == tokens.shape
    assert (diagnostics.weights > 0).sum(-1).eq(2).all()
    assert torch.allclose(diagnostics.weights.sum(-1), torch.ones(2, 5))
    assert router_temperature(0) == 2.0
    assert router_temperature(4) == 1.0


def test_full_spatial_context_is_required() -> None:
    module = DaMoeFfn(nn.Linear(8, 16), nn.Linear(16, 8), rank=2)
    try:
        module(torch.randn(1, 2, 8), torch.randn(1, 1, 8), torch.randn(1, 8), temperature=1.0)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("single-key context must be rejected")
