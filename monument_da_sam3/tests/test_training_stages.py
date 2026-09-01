from torch import nn

from monument_da_sam3.training import build_stage2_hard_pool


def test_hard_pool_is_top_quartile_union_both_class_tiles() -> None:
    train = ["a", "b", "c", "d"]
    pool = build_stage2_hard_pool(train, ["a"], {"a": 0.1, "b": 0.9, "c": 0.3, "d": 0.2})
    assert pool == ("a", "b")
