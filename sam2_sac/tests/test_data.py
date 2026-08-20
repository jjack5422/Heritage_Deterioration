"""Tests for the H0 data contract and leak-free nested splits."""

from __future__ import annotations

from pathlib import Path

from sam2_sac.data import H0TileDataset, prepare_data_plan


DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "dataset_v2_3class"


def test_fold_zero_is_panel_disjoint_and_uses_next_fold_for_validation() -> None:
    plan = prepare_data_plan(DATASET_ROOT, outer_fold=0)

    assert plan.inner_fold == 1
    assert len(plan.train) + len(plan.val) + len(plan.test) == 929
    assert not (set(plan.train) & set(plan.val))
    assert not (set(plan.train) & set(plan.test))
    assert not (set(plan.val) & set(plan.test))
    assert not (plan.groups(plan.train) & plan.groups(plan.val))
    assert not (plan.groups(plan.train) & plan.groups(plan.test))
    assert not (plan.groups(plan.val) & plan.groups(plan.test))


def test_dataset_keeps_the_512_label_ids_and_normalizes_rgb() -> None:
    plan = prepare_data_plan(DATASET_ROOT, outer_fold=0)
    item = H0TileDataset(plan, plan.train[:1], train_augmentation=False)[0]

    assert item["image"].shape == (3, 512, 512)
    assert item["mask"].shape == (512, 512)
    assert set(item["mask"].unique().tolist()).issubset({0, 1, 2, 255})
    assert item["name"] == plan.train[0]
