"""Locked optimizer/loss/batch arguments for A and B."""

import pytest
import torch

from sam3_adapter.expert_training_data import make_expert_target

from sam3_adapter.train import _checkpoint_improved, parse_args as parse_train_args
from sam3_adapter.train_probe import parse_args as parse_probe_args


def test_probe_defaults_match_approved_contract() -> None:
    args = parse_probe_args(["--group", "sam2_probe"])
    assert (args.epochs, args.batch_size, args.accumulation_steps) == (80, 4, 1)
    assert (args.learning_rate, args.weight_decay) == (2e-4, 5e-5)
    assert (args.positive_weight, args.dice_weight) == (2.0, 0.65)


def test_effective_batch_cannot_drift() -> None:
    with pytest.raises(SystemExit):
        parse_probe_args(["--group", "sam3_probe", "--batch-size", "2", "--accumulation-steps", "1"])


def test_adapter_has_independent_expert_specific_loss_contracts() -> None:
    scratch = parse_train_args(["--expert", "scratch_crack"])
    craquelure = parse_train_args(["--expert", "shrinkage_craquelure"])
    loss = parse_train_args(["--expert", "loss"])
    assert scratch.batch_size * scratch.accumulation_steps == 4
    assert scratch.manifest.name == "scratch_crack.json"
    assert craquelure.manifest.name == "shrinkage_craquelure.json"
    assert loss.manifest.name == "loss.json"
    assert "jacky-high-positive-validation" in scratch.experiment_id
    assert (scratch.positive_weight, craquelure.positive_weight, loss.positive_weight) == (1.0, 2.0, 1.0)
    assert craquelure.model_input_size == 512
    assert parse_train_args(["--expert", "shrinkage_craquelure", "--model-input-size", "1008"]).model_input_size == 1008
    with pytest.raises(SystemExit):
        parse_train_args(["--expert", "scratch_crack", "--positive-weight", "2"])
    with pytest.raises(SystemExit):
        parse_probe_args(["--group", "sam3_adapter"])


def test_expert_target_uses_available_crack_label_and_preserves_ignore() -> None:
    mask = torch.tensor([[0, 1, 2, 11, 36, 255]])
    target = make_expert_target(mask, (1,))
    assert target.tolist() == [[0, 1, 0, 0, 0, 255]]


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
