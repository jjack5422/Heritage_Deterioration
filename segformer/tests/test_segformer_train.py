from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from segformer_model import build_segformer, resolve_model_args  # noqa: E402


SPEC = importlib.util.spec_from_file_location("segformer_train_entry", SRC / "train.py")
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def _backbone_choices(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    action = next(action for action in parser._actions if action.dest == "backbone")
    assert action.choices is not None
    return tuple(action.choices)


def test_segformer_entrypoint_only_exposes_mit_backbones() -> None:
    assert train.PROJECT_ROOT == PROJECT_ROOT
    assert _backbone_choices(train.parser()) == (
        "segformer_b2",
        "segformer_b3",
        "segformer_b5",
    )

    with pytest.raises(SystemExit):
        train.parser().parse_args(
            [
                "--dataset-root",
                "dataset",
                "--expert",
                "crack",
                "--backbone",
                "resnet50",
            ]
        )


@pytest.mark.parametrize(
    ("backbone", "encoder", "batch_size", "accumulation_steps"),
    [
        ("segformer_b2", "mit_b2", 16, 1),
        ("segformer_b3", "mit_b3", 16, 1),
        ("segformer_b5", "mit_b5", 8, 2),
    ],
)
def test_segformer_profiles_preserve_effective_batch_16(
    backbone: str,
    encoder: str,
    batch_size: int,
    accumulation_steps: int,
) -> None:
    args = train.parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--expert",
            "crack_craquelure",
            "--backbone",
            backbone,
        ]
    )

    resolve_model_args(args)

    assert args.architecture == "segformer"
    assert args.encoder == encoder
    assert args.decoder_channels == 768
    assert args.batch_size == batch_size
    assert args.gradient_accumulation_steps == accumulation_steps
    assert args.batch_size * args.gradient_accumulation_steps == 16
    assert args.lr == pytest.approx(2e-4)
    assert args.encoder_lr_mult == pytest.approx(0.05)


def test_segformer_b2_builds_a_three_class_full_resolution_model() -> None:
    model = build_segformer(
        encoder="mit_b2",
        encoder_weights=None,
        decoder_channels=768,
        num_classes=3,
    )

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))

    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 3, 64, 64)
    assert parameters == 27_348_931
