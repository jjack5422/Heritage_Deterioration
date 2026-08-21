"""CLI contract for the controlled SAM2-Adapter five-fold run."""

from sam2_adapter.train_adapter import parse_args


def test_adapter_defaults_match_the_predeclared_primary_experiment() -> None:
    args = parse_args([])

    assert not hasattr(args, "experts")
    assert args.folds == [0, 1, 2, 3, 4]
    assert args.image_size == 512
    assert args.epochs == 80
    assert args.batch_size == 4
    assert args.accumulation_steps == 1
    assert args.learning_rate == 2e-4
    assert args.dice_weight == 0.65
    assert args.scale_factor == 32
    assert args.highpass_rate == 0.25


def test_adapter_cli_can_select_a_single_fold_for_recovery() -> None:
    args = parse_args(["--folds", "3", "--allow-existing"])

    assert args.folds == [3]
    assert args.allow_existing is True
