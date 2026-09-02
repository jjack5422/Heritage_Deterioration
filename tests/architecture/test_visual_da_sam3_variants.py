from __future__ import annotations

import inspect
import pytest
import torch
from torch import nn

import dual_adapter_sam3.model as model_module
from dual_adapter_sam3.concepts import load_concept_registry
from dual_adapter_sam3.model import (
    SUPPORTED_MODEL_VARIANTS,
    DualAdapterSam3,
    VisualDualAdapterSam3,
    build_dual_adapter_model,
)


class _FakeDaMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(2, 2)
        self.experts = nn.ModuleList(nn.Linear(2, 2) for _ in range(2))


class _FakeFusionLayer(nn.Module):
    def __init__(self, *, adapted: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2)
        self.norm2 = nn.LayerNorm(2)
        self.norm3 = nn.LayerNorm(2)
        self.base_attention = nn.Linear(2, 2)
        if adapted:
            self.da_moe = _FakeDaMoe()
            self.da_last_diagnostics = None
            self.da_router_temperature = 2.0


class _FakeTrunk(nn.Module):
    def __init__(self, *, visual: bool) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.ones(()))
        if visual:
            self.visual_adapter = nn.Sequential(nn.Conv2d(3, 3, 1), nn.GELU())


class _FakeBackbone(nn.Module):
    def __init__(self, *, visual: bool) -> None:
        super().__init__()
        self.vision_backbone = nn.Module()
        self.vision_backbone.trunk = _FakeTrunk(visual=visual)
        self.text_weight = nn.Parameter(torch.ones(()))
        self.vision_grad_modes: list[bool] = []
        self.vision_calls = 0
        self.text_calls = 0

    def forward_image(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.vision_calls += 1
        self.vision_grad_modes.append(torch.is_grad_enabled())
        trunk = self.vision_backbone.trunk
        if hasattr(trunk, "visual_adapter"):
            images = trunk.visual_adapter(images)
        return {"vision_feature": images.mean(dim=1, keepdim=True) * trunk.base_weight}

    def forward_text(self, prompts: list[str], *, device: torch.device) -> dict[str, torch.Tensor]:
        self.text_calls += 1
        assert len(prompts) == 2
        return {
            "language_features": torch.ones(3, 2, 2, device=device) * self.text_weight,
            "language_mask": torch.zeros(2, 3, dtype=torch.bool, device=device),
            "language_embeds": torch.ones(1, 2, 2, device=device),
        }


class _FakeSam3(nn.Module):
    def __init__(self, *, visual: bool) -> None:
        super().__init__()
        self.backbone = _FakeBackbone(visual=visual)
        layers = [_FakeFusionLayer(adapted=index < 3) for index in range(6)]
        self.transformer = nn.Module()
        self.transformer.encoder = nn.Module()
        self.transformer.encoder.layers = nn.ModuleList(layers)
        self.frozen_decoder_weight = nn.Parameter(torch.ones(()))
        self.grounding_calls = 0

    def eval(self) -> "_FakeSam3":
        nn.Module.train(self, False)
        return self

    def _get_dummy_prompt(self, *, num_prompts: int) -> torch.Tensor:
        return torch.zeros(num_prompts)

    def forward_grounding(self, *, backbone_out: dict[str, torch.Tensor], **_: object) -> dict[str, torch.Tensor]:
        self.grounding_calls += 1
        feature = backbone_out["vision_feature"] * self.frozen_decoder_weight
        return {
            "semantic_seg": feature,
            "presence_logit_dec": feature.mean(dim=(1, 2, 3)),
        }


@pytest.fixture
def registry():
    return load_concept_registry("dual_adapter_sam3/configs/concepts.yaml")


@pytest.fixture
def fake_builders(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"legacy": 0, "visual": 0}

    def legacy_builder(*_: object, **__: object):
        calls["legacy"] += 1
        return _FakeSam3(visual=False), {"model_variant": "da_sam3"}

    def visual_builder(*_: object, **__: object):
        calls["visual"] += 1
        return _FakeSam3(visual=True), {"model_variant": "visual_da_sam3"}

    monkeypatch.setattr(model_module, "build_official_da_sam3", legacy_builder)
    monkeypatch.setattr(model_module, "build_official_visual_da_sam3", visual_builder)
    monkeypatch.setattr(DualAdapterSam3, "_make_find_stage", staticmethod(lambda *_: object()))
    return calls


def test_variant_factory_defaults_to_legacy_and_builds_each_model_once(registry, fake_builders) -> None:
    assert SUPPORTED_MODEL_VARIANTS == ("da_sam3", "visual_da_sam3")

    legacy = build_dual_adapter_model(registry, device="cpu")
    visual = build_dual_adapter_model(registry, model_variant="visual_da_sam3", device="cpu")

    assert type(legacy) is DualAdapterSam3
    assert type(visual) is VisualDualAdapterSam3
    assert legacy.model_variant == "da_sam3"
    assert visual.model_variant == "visual_da_sam3"
    assert fake_builders == {"legacy": 1, "visual": 1}


def test_variant_factory_rejects_unknown_name(registry, fake_builders) -> None:
    with pytest.raises(ValueError, match="da_sam3.*visual_da_sam3"):
        build_dual_adapter_model(registry, model_variant="unknown", device="cpu")
    assert fake_builders == {"legacy": 0, "visual": 0}


def test_shared_vision_forward_uses_two_prompts_without_targets(registry, fake_builders) -> None:
    legacy = build_dual_adapter_model(registry, device="cpu")
    visual = build_dual_adapter_model(registry, model_variant="visual_da_sam3", device="cpu")
    images = torch.rand(2, 3, 512, 512, requires_grad=True)

    legacy_output = legacy(images)
    visual_output = visual(images)

    assert tuple(legacy_output.logits.shape) == (2, 2, 512, 512)
    assert tuple(visual_output.logits.shape) == (2, 2, 512, 512)
    assert tuple(visual_output.presence_logits.shape) == (2, 2)
    assert legacy_output.vision_forward_calls == visual_output.vision_forward_calls == 1
    assert legacy.sam3.backbone.vision_calls == visual.sam3.backbone.vision_calls == 1
    assert legacy.sam3.backbone.text_calls == visual.sam3.backbone.text_calls == 1
    assert legacy.sam3.grounding_calls == visual.sam3.grounding_calls == 2
    assert legacy.sam3.backbone.vision_grad_modes == [False]
    assert visual.sam3.backbone.vision_grad_modes == [True]
    assert tuple(inspect.signature(visual.forward).parameters) == ("images",)


def _trainable_names(model: DualAdapterSam3, stage: str) -> set[str]:
    return set(model.configure_stage(stage))


def test_stage_scopes_include_visual_adapter_only_in_hybrid_stage1(registry, fake_builders) -> None:
    legacy = build_dual_adapter_model(registry, device="cpu")
    visual = build_dual_adapter_model(registry, model_variant="visual_da_sam3", device="cpu")

    legacy_stage1 = _trainable_names(legacy, "stage1")
    visual_stage1 = _trainable_names(visual, "stage1")
    legacy_stage2 = _trainable_names(legacy, "stage2")
    visual_stage2 = _trainable_names(visual, "stage2")

    assert not any("visual_adapter" in name for name in legacy_stage1 | legacy_stage2)
    assert any("visual_adapter" in name for name in visual_stage1)
    assert not any("visual_adapter" in name for name in visual_stage2)
    assert all(".da_moe.router." in name for name in legacy_stage2)
    assert all(".da_moe.router." in name for name in visual_stage2)
    assert any(".da_moe.experts." in name for name in legacy_stage1)
    assert any(".da_moe.experts." in name for name in visual_stage1)
    assert any(".norm1." in name for name in legacy_stage1)

    frozen_names = {
        name
        for name, parameter in visual.named_parameters()
        if not parameter.requires_grad
    }
    assert any("base_weight" in name for name in frozen_names)
    assert any("text_weight" in name for name in frozen_names)
    assert any("frozen_decoder_weight" in name for name in frozen_names)


def test_train_mode_keeps_official_sam3_eval_and_only_enables_stage_modules(registry, fake_builders) -> None:
    visual = build_dual_adapter_model(registry, model_variant="visual_da_sam3", device="cpu")
    visual.configure_stage("stage1")
    visual.train(True)

    assert visual.training is True
    assert visual.sam3.training is False
    assert visual.visual_adapter.training is True

    visual.configure_stage("stage2")
    visual.train(True)
    assert visual.sam3.training is False
    assert visual.visual_adapter.training is False
