from torch import nn

import pytest

from dual_adapter_sam3.training import (
    RouterHealthGate,
    build_optimizer_and_scheduler,
    build_stage2_hard_pool,
)


def test_hard_pool_is_top_quartile_union_both_class_tiles() -> None:
    train = ["a", "b", "c", "d"]
    pool = build_stage2_hard_pool(train, ["a"], {"a": 0.1, "b": 0.9, "c": 0.3, "d": 0.2})
    assert pool == ("a", "b")


def _router_rows(low_usage: float) -> list[dict[str, float | int | str]]:
    return [
        {
            "concept": concept,
            "layer": layer,
            "expert": expert,
            "top2_usage": low_usage if (layer, expert) == (1, 2) else 0.5,
        }
        for concept in ("crack_craquelure", "loss")
        for layer in range(3)
        for expert in range(4)
    ]


def test_router_health_aggregates_prompts_and_stops_after_three_epochs() -> None:
    gate = RouterHealthGate()
    gate.update(_router_rows(0.01))
    gate.update(_router_rows(0.01))
    with pytest.raises(RuntimeError, match="router collapse"):
        gate.update(_router_rows(0.01))


def test_router_health_streak_recovers() -> None:
    gate = RouterHealthGate()
    gate.update(_router_rows(0.01))
    gate.update(_router_rows(0.2))
    gate.update(_router_rows(0.01))
    gate.update(_router_rows(0.01))


class _FullDecoderModel(nn.Module):
    full_pixel_decoder = True

    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.sam3 = nn.Module()
        self.sam3.segmentation_head = nn.Module()
        self.sam3.segmentation_head.pixel_decoder = nn.Module()
        self.sam3.segmentation_head.pixel_decoder.conv_layers = nn.ModuleList(
            nn.Conv2d(2, 2, 3, padding=1) for _ in range(3)
        )
        self.sam3.segmentation_head.pixel_decoder.norms = nn.ModuleList(
            nn.GroupNorm(1, 2) for _ in range(3)
        )
        self.sam3.segmentation_head.semantic_seg_head = nn.Conv2d(2, 1, 1)

    def configure_stage(self, _stage: str) -> tuple[str, ...]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(True)
        for name, parameter in self.named_parameters():
            if any(
                token in name
                for token in (
                    "conv_layers.0",
                    "conv_layers.1",
                    "norms.0",
                    "norms.1",
                    "semantic_seg_head",
                )
            ):
                parameter.requires_grad_(True)
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


def test_full_decoder_optimizer_uses_lower_decoder_lr_and_no_decay_for_norm_bias() -> None:
    model = _FullDecoderModel()

    bundle = build_optimizer_and_scheduler(model, stage="stage1", epochs=2)

    settings = {(group["lr"], group["weight_decay"]) for group in bundle.optimizer.param_groups}
    assert settings == {
        (5e-4, 0.1),
        (5e-5, 1e-4),
        (5e-5, 0.0),
        (1e-4, 1e-4),
        (1e-4, 0.0),
    }
    optimized = {id(parameter) for group in bundle.optimizer.param_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert optimized == trainable
