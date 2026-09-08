from pathlib import Path

import pytest

from dual_adapter_sam3.decoder_calibration import CALIBRATION_EXPERT
from dual_adapter_sam3.train_decoder_calibration import (
    DEFAULT_EXPERIMENT,
    DEFAULT_SOURCE_EXPERIMENT,
    build_parser,
    calibration_run_root,
    source_checkpoint_path,
)


def test_decoder_calibration_cli_has_locked_defaults() -> None:
    args = build_parser().parse_args(["--fold", "2"])

    assert args.experiment_id == DEFAULT_EXPERIMENT
    assert args.source_experiment_id == DEFAULT_SOURCE_EXPERIMENT
    assert args.epochs == 20
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 0.1
    assert args.batch_size == 4
    assert args.workers == 4
    assert args.craquelure_loss == "original"


def test_decoder_calibration_cli_accepts_sam2_craquelure_loss() -> None:
    args = build_parser().parse_args(
        ["--fold", "2", "--craquelure-loss", "sam2_bce_dice"]
    )

    assert args.craquelure_loss == "sam2_bce_dice"


def test_decoder_calibration_paths_are_fold_and_expert_specific(tmp_path: Path) -> None:
    assert calibration_run_root(tmp_path / "target", 3) == (
        tmp_path / "target" / "5fold" / CALIBRATION_EXPERT / "fold3"
    )
    assert source_checkpoint_path(tmp_path / "source", 3) == (
        tmp_path
        / "source"
        / "5fold"
        / "da_sam3"
        / "fold3"
        / "artifacts"
        / "checkpoints"
        / "stage2_best.pt"
    )
    with pytest.raises(ValueError, match="fold"):
        calibration_run_root(tmp_path, 5)
