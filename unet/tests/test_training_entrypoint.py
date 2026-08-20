from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

import train  # noqa: E402
from reporting.backfill import reconstructed_training_command  # noqa: E402


def _backbone_choices(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    action = next(action for action in parser._actions if action.dest == "backbone")
    assert action.choices is not None
    return tuple(action.choices)


def test_unet_entrypoint_only_exposes_resnet_and_convnext_backbones() -> None:
    assert train.PROJECT_ROOT == PROJECT_ROOT
    assert _backbone_choices(train.parser()) == (
        "resnet50",
        "convnext_tiny",
        "convnext_base",
        "convnext_large",
    )

    with pytest.raises(SystemExit):
        train.parser().parse_args(
            [
                "--dataset-root",
                "dataset",
                "--expert",
                "crack",
                "--backbone",
                "segformer_b2",
            ]
        )


def test_backfill_reconstructs_the_unet_project_entrypoint() -> None:
    command = reconstructed_training_command(
        argparse.Namespace(backbone="convnext_large", expert="crack")
    )

    assert command[:2] == ["crackseg_env/bin/python", "unet/src/train.py"]
