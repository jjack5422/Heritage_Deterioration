"""U-Net model construction and capacity-aware backbone profiles."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch.nn as nn
import segmentation_models_pytorch as smp


@dataclass(frozen=True)
class BackboneProfile:
    """Resolve one user-facing backbone to its model and training defaults."""

    encoder: str
    batch_size: int
    lr: float
    encoder_lr_mult: float
    gradient_accumulation_steps: int = 1


BACKBONE_PROFILES = {
    "resnet50": BackboneProfile(
        encoder="resnet50",
        batch_size=32,
        lr=3e-4,
        encoder_lr_mult=0.1,
    ),
    "convnext_tiny": BackboneProfile(
        encoder="tu-convnext_tiny.fb_in22k_ft_in1k",
        batch_size=16,
        lr=2e-4,
        encoder_lr_mult=0.05,
    ),
    "convnext_base": BackboneProfile(
        encoder="tu-convnext_base.fb_in22k_ft_in1k",
        batch_size=16,
        lr=2e-4,
        encoder_lr_mult=0.05,
    ),
    "convnext_large": BackboneProfile(
        encoder="tu-convnext_large",
        batch_size=16,
        lr=2e-4,
        encoder_lr_mult=0.05,
    ),
}


UNET_BACKBONES = (
    "resnet50",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
)


def backbone_names() -> tuple[str, ...]:
    return UNET_BACKBONES


def backbone_profile(name: str) -> BackboneProfile:
    try:
        return BACKBONE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(backbone_names())
        raise ValueError(f"unknown backbone {name!r}; choose one of: {choices}") from exc


def add_model_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    group = parser.add_argument_group("model")
    choices = backbone_names()
    group.add_argument(
        "--backbone",
        choices=choices,
        default=choices[0],
        help="capacity-aware model profile",
    )
    group.add_argument(
        "--encoder",
        help="advanced SMP encoder override; normally use --backbone",
    )
    group.add_argument("--encoder-weights", default="imagenet")
    group.add_argument("--batch-size", type=int)
    group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="micro-batches per optimizer step",
    )
    group.add_argument("--lr", type=float, help="decoder learning rate")
    group.add_argument("--encoder-lr-mult", type=float)


def resolve_model_args(
    args: argparse.Namespace,
) -> None:
    """Apply backbone defaults once while preserving explicit CLI overrides."""

    profile = backbone_profile(args.backbone)
    args.architecture = "unet"
    args.decoder_channels = None
    args.encoder = args.encoder if args.encoder is not None else profile.encoder
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


def build_resunet(encoder: str = "resnet50",
                  encoder_weights: str | None = "imagenet",
                  num_classes: int = 2,
                  in_channels: int = 3) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    """Build the U-Net selected by resolved command-line arguments."""

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    return build_resunet(
        encoder=args.encoder,
        encoder_weights=encoder_weights,
        num_classes=num_classes,
    )
