import torch
from torch import nn

from dual_adapter_sam3.da_moe import DaMoeFfn


def test_dpe_is_zero_at_initialization_and_receives_gradient() -> None:
    module = DaMoeFfn(nn.Linear(8, 16), nn.Linear(16, 8), rank=2)
    for expert in module.experts:
        assert torch.count_nonzero(expert.linear1_delta.up.weight) == 0
        assert torch.count_nonzero(expert.linear2_delta.up.weight) == 0
    output, _ = module(torch.randn(2, 5, 8), torch.randn(2, 9, 8), torch.randn(2, 8), temperature=1.0)
    output.square().mean().backward()
    for expert in module.experts:
        for delta in (expert.linear1_delta, expert.linear2_delta):
            assert delta.up.weight.grad is not None
            assert torch.isfinite(delta.up.weight.grad).all()
    assert all(parameter.grad is None for parameter in module.base_linear1.parameters())
    assert all(parameter.grad is None for parameter in module.base_linear2.parameters())
