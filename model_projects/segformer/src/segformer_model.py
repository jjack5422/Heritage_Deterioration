"""SegFormer construction and capacity-aware MiT backbone profiles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import segmentation_models_pytorch as smp
from torch import nn


@dataclass(frozen=True)
class BackboneProfile:
    encoder: str
    batch_size: int
    lr: float
    encoder_lr_mult: float
    gradient_accumulation_steps: int = 1
    decoder_channels: int = 768


BACKBONE_PROFILES = {
    "segformer_b2": BackboneProfile(
        encoder="mit_b2",
        batch_size=16,
        lr=2e-4,
        encoder_lr_mult=0.05,
    ),
    "segformer_b3": BackboneProfile(
        encoder="mit_b3",
        batch_size=16,
        lr=2e-4,
        encoder_lr_mult=0.05,
    ),
    "segformer_b5": BackboneProfile(
        encoder="mit_b5",
        batch_size=8,
        lr=2e-4,
        encoder_lr_mult=0.05,
        gradient_accumulation_steps=2,
    ),
}


def backbone_names() -> tuple[str, ...]:
    return tuple(BACKBONE_PROFILES)


def backbone_profile(name: str) -> BackboneProfile:
    try:
        return BACKBONE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(backbone_names())
        raise ValueError(f"unknown backbone {name!r}; choose one of: {choices}") from exc


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--backbone",
        choices=backbone_names(),
        default="segformer_b2",
        help="capacity-aware SegFormer profile",
    )
    group.add_argument("--encoder", help="advanced SMP encoder override")
    group.add_argument("--encoder-weights", default="imagenet")
    group.add_argument("--batch-size", type=int)
    group.add_argument("--gradient-accumulation-steps", type=int)
    group.add_argument("--lr", type=float, help="decoder learning rate")
    group.add_argument("--encoder-lr-mult", type=float)


def resolve_model_args(args: argparse.Namespace) -> None:
    profile = backbone_profile(args.backbone)
    args.architecture = "segformer"
    args.encoder = args.encoder if args.encoder is not None else profile.encoder
    args.decoder_channels = profile.decoder_channels
    args.batch_size = args.batch_size if args.batch_size is not None else profile.batch_size
    args.gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else profile.gradient_accumulation_steps
    )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    args.lr = args.lr if args.lr is not None else profile.lr
    args.encoder_lr_mult = (
        args.encoder_lr_mult
        if args.encoder_lr_mult is not None
        else profile.encoder_lr_mult
    )


def build_segformer(
    encoder: str = "mit_b2",
    encoder_weights: str | None = "imagenet",
    decoder_channels: int = 768,
    num_classes: int = 2,
    in_channels: int = 3,
) -> nn.Module:
    return smp.Segformer(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        decoder_segmentation_channels=decoder_channels,
        in_channels=in_channels,
        classes=num_classes,
    )


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    return build_segformer(
        encoder=args.encoder,
        encoder_weights=encoder_weights,
        decoder_channels=args.decoder_channels,
        num_classes=num_classes,
    )
