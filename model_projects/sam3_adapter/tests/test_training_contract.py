"""Locked optimizer, loss, and batch arguments for SAM3-Adapter experts."""

import pytest

from model_projects.sam3_adapter.train import _checkpoint_improved, parse_args as parse_train_args


def test_effective_batch_is_configurable_but_must_stay_positive() -> None:
    args = parse_train_args(["--expert", "loss", "--batch-size", "2", "--accumulation-steps", "1"])
    assert (args.batch_size, args.accumulation_steps) == (2, 1)
    with pytest.raises(SystemExit):
        parse_train_args(["--expert", "loss", "--batch-size", "0"])


def test_training_defaults_allow_explicit_ablation_overrides() -> None:
    scratch = parse_train_args(["--expert", "scratch_crack"])
    craquelure = parse_train_args(["--expert", "shrinkage_craquelure"])
    loss = parse_train_args(["--expert", "loss"])
    assert (scratch.epochs, craquelure.epochs, loss.epochs) == (60, 60, 60)
    assert scratch.batch_size * scratch.accumulation_steps == 4
    assert scratch.manifest.name == "scratch_crack.json"
    assert craquelure.manifest.name == "shrinkage_craquelure.json"
    assert loss.manifest.name == "loss.json"
    assert {scratch.manifest.parent.name, craquelure.manifest.parent.name, loss.manifest.parent.name} == {"transfer_512"}
    assert scratch.experiment_id == "2026-09-20_transfer-512_seed42"
    assert (scratch.positive_weight, craquelure.positive_weight, loss.positive_weight) == (1.0, 2.0, 1.0)
    assert craquelure.model_input_size == 512
    assert parse_train_args(["--expert", "shrinkage_craquelure", "--model-input-size", "768"]).model_input_size == 768
    custom_loss = parse_train_args([
        "--expert", "scratch_crack", "--positive-weight", "2", "--dice-weight", "0.5",
    ])
    assert (custom_loss.positive_weight, custom_loss.dice_weight) == (2.0, 0.5)


def test_checkpoint_selection_maximizes_f1_then_minimizes_loss() -> None:
    assert _checkpoint_improved(
        validation_f1=0.7,
        validation_loss=0.5,
        best_f1=0.6,
        best_loss=0.4,
    )
    assert _checkpoint_improved(
        validation_f1=0.7,
        validation_loss=0.3,
        best_f1=0.7,
        best_loss=0.4,
    )
    assert not _checkpoint_improved(
        validation_f1=0.6,
        validation_loss=0.2,
        best_f1=0.7,
        best_loss=0.4,
    )
