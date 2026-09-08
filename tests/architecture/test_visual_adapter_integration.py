from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dual_adapter_sam3.sam3_integration import activate_official_sam3
from dual_adapter_sam3.visual_adapter import (
    VisualAdapterConfig,
    grad_compatible_mlp_forward,
    inject_visual_adapter,
    install_grad_compatible_mlp_forward,
    validate_official_vit_contract,
)


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 4)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(4, 1024)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(inputs))))))


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinyMlp()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.mlp(inputs)


class _TinyPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs).permute(0, 2, 3, 1)


class _TinyOfficialTrunk(nn.Module):
    def __init__(self, *, depth: int = 32, embed_dim: int = 1024, patch_size: int = 14) -> None:
        super().__init__()
        self.patch_embed = _TinyPatchEmbed()
        if patch_size != 14 or embed_dim != 1024:
            self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(_TinyBlock() for _ in range(depth))
        self.full_attn_ids = [7, 15, 23, 31]
        self.retain_cls_token = False

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        tokens = self.patch_embed(inputs)
        for block in self.blocks:
            tokens = block(tokens)
        return [tokens.permute(0, 3, 1, 2)]


def _official_mlp() -> nn.Module:
    activate_official_sam3()
    from sam3.model.vitdet import Mlp

    mlp = Mlp(in_features=8, hidden_features=16, out_features=8, drop=0.0)
    mlp.eval()
    for parameter in mlp.parameters():
        parameter.requires_grad_(False)
    return mlp


def test_grad_compatible_mlp_matches_official_no_grad_result() -> None:
    torch.manual_seed(11)
    mlp = _official_mlp().to(dtype=torch.bfloat16)
    inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    with torch.no_grad():
        expected = mlp(inputs)
        actual = grad_compatible_mlp_forward(mlp, inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_grad_compatible_mlp_propagates_input_grad_but_not_weight_grad() -> None:
    mlp = _official_mlp()
    inputs = torch.randn(2, 3, 8, requires_grad=True)

    with pytest.raises(ValueError, match="grad"):
        mlp(inputs)

    output = grad_compatible_mlp_forward(mlp, inputs)
    output.square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad) > 0
    assert all(parameter.grad is None for parameter in mlp.parameters())


def test_grad_compatible_mlp_rejects_incompatible_module() -> None:
    with pytest.raises(TypeError, match="fc1"):
        grad_compatible_mlp_forward(nn.Identity(), torch.randn(1, 2, 3))


@pytest.mark.parametrize(
    "trunk,match",
    [
        (_TinyOfficialTrunk(depth=31), "32 blocks"),
        (_TinyOfficialTrunk(embed_dim=512), "1024"),
        (_TinyOfficialTrunk(patch_size=16), "patch size 14"),
    ],
)
def test_official_vit_contract_rejects_incompatible_trunk(trunk: nn.Module, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        validate_official_vit_contract(trunk)


def test_visual_adapter_injection_is_identity_and_prepares_once_per_forward() -> None:
    torch.manual_seed(5)
    trunk = _TinyOfficialTrunk().eval()
    images = torch.randn(1, 3, 28, 28)
    with torch.no_grad():
        expected = trunk(images)[0]

    bank = inject_visual_adapter(trunk, VisualAdapterConfig())
    with torch.no_grad():
        actual = trunk(images)[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bank.prepare_calls == 1
    assert trunk.visual_adapter is bank
    assert sum(1 for name, _ in trunk.named_modules() if name == "visual_adapter") == 1

    with pytest.raises(RuntimeError, match="already"):
        inject_visual_adapter(trunk, VisualAdapterConfig())


def test_mlp_installation_keeps_weights_frozen_and_allows_trunk_input_grad() -> None:
    trunk = _TinyOfficialTrunk().eval()
    for parameter in trunk.parameters():
        parameter.requires_grad_(False)

    installed = install_grad_compatible_mlp_forward(trunk)
    bank = inject_visual_adapter(trunk, VisualAdapterConfig())
    with torch.no_grad():
        bank.stage_up_projections[0].weight.fill_(0.001)
    images = torch.randn(1, 3, 28, 28, requires_grad=True)
    trunk(images)[0].square().mean().backward()

    assert installed == tuple(range(32))
    assert images.grad is not None and torch.count_nonzero(images.grad) > 0
    assert all(parameter.grad is None for block in trunk.blocks for parameter in block.mlp.parameters())


def test_runtime_does_not_import_vendor_upstream() -> None:
    project_root = Path(__file__).resolve().parents[2]
    python_sources = (project_root / "dual_adapter_sam3").glob("*.py")
    offending = [path.name for path in python_sources if "vendor_upstream_runtime" in path.read_text()]
    assert offending == []
