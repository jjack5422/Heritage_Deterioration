from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.retest_merged_models import (
    merged_binary_target,
    merged_foreground_probability,
    source_oof_partitions,
)


def test_merged_target_scores_only_background_and_merged_craquelure() -> None:
    raw = np.array([[0, 1, 2, 3, 4, 5, 255]], dtype=np.uint8)

    target, valid = merged_binary_target(raw)

    assert target.tolist() == [[False, True, False, False, False, False, False]]
    assert valid.tolist() == [[True, True, False, False, False, False, False]]


def test_three_class_probability_is_collapsed_before_fixed_threshold() -> None:
    probabilities = np.array(
        [
            [[0.60, 0.45], [0.20, 0.10]],
            [[0.25, 0.30], [0.40, 0.70]],
            [[0.15, 0.25], [0.40, 0.20]],
        ],
        dtype=np.float32,
    )

    merged = merged_foreground_probability(probabilities)

    np.testing.assert_allclose(merged, [[0.40, 0.55], [0.80, 0.90]])


def test_source_oof_partitions_use_checkpoint_split_not_target_fold(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    folds = {
        0: {"train": ["c.png", "d.png"], "holdout": ["a.png", "b.png"]},
        1: {"train": ["a.png", "b.png"], "holdout": ["c.png", "d.png"]},
    }
    for fold, values in folds.items():
        (splits / f"fold{fold}.json").write_text(
            json.dumps(
                {
                    "outer_fold": fold,
                    "folds": [{"train": values["train"]}],
                    "holdout_tiles": values["holdout"],
                }
            ),
            encoding="utf-8",
        )

    train, validation, outer_test, inner_fold = source_oof_partitions(
        tmp_path, outer_fold=0
    )

    assert inner_fold == 1
    assert train == ()
    assert validation == ("c.png", "d.png")
    assert outer_test == ("a.png", "b.png")

