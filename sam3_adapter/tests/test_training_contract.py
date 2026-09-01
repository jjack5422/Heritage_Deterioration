"""Locked optimizer/loss/batch arguments for A and B."""

import pytest

from sam3_adapter.train_probe import parse_args


def test_probe_defaults_match_approved_contract() -> None:
    args = parse_args(["--group", "sam2_probe"])
    assert (args.epochs, args.batch_size, args.accumulation_steps) == (80, 4, 1)
    assert (args.learning_rate, args.weight_decay) == (2e-4, 5e-5)
    assert (args.positive_weight, args.dice_weight) == (2.0, 0.65)


def test_effective_batch_cannot_drift() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--group", "sam3_probe", "--batch-size", "2", "--accumulation-steps", "1"])


def test_direct_512_is_an_explicit_sam3_adapter_variant() -> None:
    args = parse_args(["--group", "sam3_adapter", "--model-input-size", "512"])
    assert args.model_input_size == 512
    with pytest.raises(SystemExit):
        parse_args(["--group", "sam3_probe", "--model-input-size", "512"])
