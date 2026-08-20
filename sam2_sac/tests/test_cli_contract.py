"""CLI defaults must select the checkpoint-compatible official SAM2.1 config."""

from sam2_sac.train_h0 import parse_args


def test_h0_defaults_use_checkpoint_compatible_sam21_hiera_large_config() -> None:
    args = parse_args([])

    assert args.sam2_config == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert args.image_size == 512


def test_h0_can_disable_early_stopping_for_best_vs_last_comparison() -> None:
    args = parse_args(["--epochs", "80", "--no-early-stop"])

    assert args.epochs == 80
    assert args.no_early_stop is True
