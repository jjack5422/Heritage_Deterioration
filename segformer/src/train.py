"""Train SegFormer crack segmentation models."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from crackseg_common.training_runtime import (
    TrainingProject,
    main_for_project,
    training_parser,
)
from segformer_model import add_model_arguments, build_model, resolve_model_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = TrainingProject(
    root=PROJECT_ROOT,
    description=__doc__ or "Train SegFormer segmentation models.",
    add_model_arguments=add_model_arguments,
    resolve_model_args=resolve_model_args,
    build_model=build_model,
)


def parser() -> argparse.ArgumentParser:
    return training_parser(PROJECT)


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_project(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
