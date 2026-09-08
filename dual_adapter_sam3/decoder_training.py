"""Trainable scope for the effective SAM3 PixelDecoder path."""

from __future__ import annotations

from torch import nn


ACTIVE_PIXEL_DECODER_STAGES = (0, 1)


def _decoder_parts(model: nn.Module) -> tuple[tuple[nn.Module, ...], nn.Module]:
    sam3 = getattr(model, "sam3", None)
    segmentation_head = getattr(sam3, "segmentation_head", None)
    pixel_decoder = getattr(segmentation_head, "pixel_decoder", None)
    conv_layers = getattr(pixel_decoder, "conv_layers", None)
    norms = getattr(pixel_decoder, "norms", None)
    semantic_head = getattr(segmentation_head, "semantic_seg_head", None)
    if not isinstance(conv_layers, nn.ModuleList) or len(conv_layers) != 3:
        raise RuntimeError("official SAM3 PixelDecoder must expose three convolution stages")
    if not isinstance(norms, nn.ModuleList) or len(norms) != len(conv_layers):
        raise RuntimeError("PixelDecoder convolution/normalization stages do not match")
    if not isinstance(semantic_head, nn.Module):
        raise RuntimeError("segmentation head exposes no semantic_seg_head")
    modules = tuple(
        module
        for index in ACTIVE_PIXEL_DECODER_STAGES
        for module in (conv_layers[index], norms[index])
    )
    return modules, semantic_head


def expected_full_pixel_decoder_keys(model: nn.Module) -> tuple[str, ...]:
    """Return parameters used by both active PixelDecoder stages and semantic head."""

    decoder_modules, semantic_head = _decoder_parts(model)
    target_ids = {
        id(parameter)
        for module in (*decoder_modules, semantic_head)
        for parameter in module.parameters()
    }
    keys = tuple(
        name for name, parameter in model.named_parameters() if id(parameter) in target_ids
    )
    if len(keys) != 10:
        raise RuntimeError(
            f"full PixelDecoder training expected ten tensors, found {len(keys)}: {keys}"
        )
    return keys


def enable_full_pixel_decoder(model: nn.Module) -> tuple[str, ...]:
    """Enable every parameter on the PixelDecoder path that executes for three FPN maps."""

    keys = expected_full_pixel_decoder_keys(model)
    named = dict(model.named_parameters())
    for name in keys:
        named[name].requires_grad_(True)
    return keys


def set_full_pixel_decoder_train_mode(model: nn.Module, mode: bool) -> None:
    decoder_modules, semantic_head = _decoder_parts(model)
    for module in (*decoder_modules, semantic_head):
        module.train(mode)


__all__ = [
    "ACTIVE_PIXEL_DECODER_STAGES",
    "enable_full_pixel_decoder",
    "expected_full_pixel_decoder_keys",
    "set_full_pixel_decoder_train_mode",
]
