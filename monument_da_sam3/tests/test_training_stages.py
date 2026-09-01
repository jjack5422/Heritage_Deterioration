from torch import nn

import pytest

from monument_da_sam3.training import RouterHealthGate, build_stage2_hard_pool


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
