"""Behavioral contracts for the locked three-expert SAM2 run."""

import pytest

from sam2_adapter.train_experts import _checkpoint_improved, parse_args


def test_all_experts_default_to_1008_and_fifty_epochs() -> None:
    loss = parse_args(["--expert", "loss"])
    crack = parse_args(["--expert", "scratch_crack"])
    craquelure = parse_args(["--expert", "shrinkage_craquelure"])

    assert {loss.epochs, crack.epochs, craquelure.epochs} == {50}
    assert {loss.model_input_size, crack.model_input_size, craquelure.model_input_size} == {1008}
    assert (loss.batch_size, loss.accumulation_steps) == (4, 1)
    assert (crack.batch_size, crack.accumulation_steps) == (4, 1)
    assert (craquelure.batch_size, craquelure.accumulation_steps) == (2, 2)
    assert (loss.positive_weight, crack.positive_weight, craquelure.positive_weight) == (2.0, 1.0, 2.0)


def test_cli_accepts_1024_and_rejects_other_input_sizes() -> None:
    assert parse_args(
        ["--expert", "loss", "--model-input-size", "1024"]
    ).model_input_size == 1024
    with pytest.raises(SystemExit):
        parse_args(["--expert", "loss", "--model-input-size", "992"])


def test_cli_rejects_unapproved_epoch_or_batch_contract() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--expert", "loss", "--epochs", "49"])
    with pytest.raises(SystemExit):
        parse_args(["--expert", "shrinkage_craquelure", "--batch-size", "4"])


def test_checkpoint_selection_maximizes_f1_then_minimizes_loss() -> None:
    assert _checkpoint_improved(validation_f1=0.7, validation_loss=0.5, best_f1=0.6, best_loss=0.4)
    assert _checkpoint_improved(validation_f1=0.7, validation_loss=0.3, best_f1=0.7, best_loss=0.4)
    assert not _checkpoint_improved(validation_f1=0.6, validation_loss=0.2, best_f1=0.7, best_loss=0.4)
