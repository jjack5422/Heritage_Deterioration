"""Repository-native FFT Visual Adapter for the official SAM3 ViT trunk."""

from __future__ import annotations

import math
import types
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class VisualAdapterConfig:
    """Fixed Visual Adapter contract approved for ``visual_da_sam3``."""

    image_size: int = 512
    patch_size: int = 14
    depth: int = 32
    stages: int = 4
    embed_dim: int = 1024
    bottleneck_dim: int = 32
    high_pass_ratio: float = 0.25

    def __post_init__(self) -> None:
        expected = {
            "image_size": 512,
            "patch_size": 14,
            "depth": 32,
            "stages": 4,
            "embed_dim": 1024,
            "bottleneck_dim": 32,
        }
        actual = asdict(self)
        mismatched = {
            name: (actual[name], value)
            for name, value in expected.items()
            if actual[name] != value
        }
        if mismatched:
            raise ValueError(f"unsupported Visual Adapter contract: {mismatched}")
        if not 0.0 < self.high_pass_ratio < 1.0:
            raise ValueError("high_pass_ratio must be in (0, 1)")
        if self.depth % self.stages:
            raise ValueError("depth must divide evenly across stages")

    @property
    def blocks_per_stage(self) -> int:
        return self.depth // self.stages

    def to_contract(self) -> dict[str, int | float]:
        return {**asdict(self), "blocks_per_stage": self.blocks_per_stage}


def _require_finite(tensor: Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} must contain only finite values")


def _validate_images(images: Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")
    if images.shape[0] < 1 or min(images.shape[-2:]) < 2:
        raise ValueError(f"images must have non-empty spatial dimensions, got {tuple(images.shape)}")
    _require_finite(images, name="images")


def fft_high_pass(images: Tensor, *, area_ratio: float) -> Tensor:
    """Return the absolute high-frequency reconstruction in float32."""

    _validate_images(images)
    if not 0.0 < area_ratio < 1.0:
        raise ValueError("area_ratio must be in (0, 1)")
    source = images.float()
    height, width = source.shape[-2:]
    half_side = int(math.sqrt(height * width * area_ratio) // 2)
    if half_side < 1:
        raise ValueError("area_ratio is too small for the input spatial dimensions")

    low_frequency = torch.zeros(
        (height, width),
        dtype=torch.bool,
        device=source.device,
    )
    center_y, center_x = height // 2, width // 2
    low_frequency[
        max(0, center_y - half_side) : min(height, center_y + half_side),
        max(0, center_x - half_side) : min(width, center_x + half_side),
    ] = True

    spectrum = torch.fft.fftshift(
        torch.fft.fft2(source, norm="forward"),
        dim=(-2, -1),
    )
    filtered = spectrum * (~low_frequency).view(1, 1, height, width)
    reconstruction = torch.fft.ifft2(
        torch.fft.ifftshift(filtered, dim=(-2, -1)),
        norm="forward",
    ).real.abs()
    _require_finite(reconstruction, name="FFT high-pass reconstruction")
    return reconstruction


class OverlapPatchEmbedding(nn.Module):
    """Convolutional overlap embedding with channel-last LayerNorm."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.normalization = nn.LayerNorm(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.projection(inputs)
        output = self.normalization(output.permute(0, 2, 3, 1))
        return output.permute(0, 3, 1, 2).contiguous()


class HandcraftedFeaturePyramid(nn.Module):
    """Build four FFT-derived handcrafted feature stages once per image."""

    def __init__(self, config: VisualAdapterConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.bottleneck_dim
        self.embeddings = nn.ModuleList(
            [
                OverlapPatchEmbedding(3, channels, kernel_size=7, stride=4),
                OverlapPatchEmbedding(channels, channels, kernel_size=3, stride=2),
                OverlapPatchEmbedding(channels, channels, kernel_size=3, stride=2),
                OverlapPatchEmbedding(channels, channels, kernel_size=3, stride=2),
            ]
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = fft_high_pass(images, area_ratio=self.config.high_pass_ratio)
        stages: list[Tensor] = []
        for embedding in self.embeddings:
            output = embedding(output)
            _require_finite(output, name="handcrafted feature")
            stages.append(output)
        return (stages[0], stages[1], stages[2], stages[3])


class VisualAdapterBank(nn.Module):
    """One shared adapter bank used before all 32 official SAM3 ViT blocks."""

    def __init__(self, config: VisualAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisualAdapterConfig()
        self.handcrafted_pyramid = HandcraftedFeaturePyramid(self.config)
        self.embedding_projections = nn.ModuleList(
            nn.Linear(self.config.embed_dim, self.config.bottleneck_dim)
            for _ in range(self.config.stages)
        )
        self.block_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.config.bottleneck_dim, self.config.bottleneck_dim),
                nn.GELU(),
            )
            for _ in range(self.config.depth)
        )
        self.stage_up_projections = nn.ModuleList(
            nn.Linear(self.config.bottleneck_dim, self.config.embed_dim)
            for _ in range(self.config.stages)
        )
        self.prepare_calls = 0
        self.apply(self._initialize_module)
        for projection in self.stage_up_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def stage_for_block(self, block_index: int) -> int:
        if not 0 <= block_index < self.config.depth:
            raise IndexError(f"block_index must be in [0, {self.config.depth}), got {block_index}")
        return block_index // self.config.blocks_per_stage

    def prepare(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self.prepare_calls += 1
        return self.handcrafted_pyramid(images)

    def inject(
        self,
        tokens: Tensor,
        pyramid: Sequence[Tensor],
        *,
        block_index: int,
    ) -> Tensor:
        stage_index = self.stage_for_block(block_index)
        expected = f"[B,H,W,{self.config.embed_dim}]"
        if tokens.ndim != 4 or tokens.shape[-1] != self.config.embed_dim:
            raise ValueError(f"tokens must be {expected}, got {tuple(tokens.shape)}")
        if len(pyramid) != self.config.stages:
            raise ValueError(f"pyramid must contain {self.config.stages} stages")
        handcrafted = pyramid[stage_index]
        if (
            handcrafted.ndim != 4
            or handcrafted.shape[0] != tokens.shape[0]
            or handcrafted.shape[1] != self.config.bottleneck_dim
        ):
            raise ValueError(
                "handcrafted feature must be "
                f"[B,{self.config.bottleneck_dim},H,W], got {tuple(handcrafted.shape)}"
            )
        _require_finite(tokens, name="tokens")
        _require_finite(handcrafted, name="handcrafted feature")

        resized = F.interpolate(
            handcrafted,
            size=tuple(tokens.shape[1:3]),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        embedding = self.embedding_projections[stage_index](tokens)
        prompt = embedding + resized.to(dtype=embedding.dtype)
        transformed = self.block_transforms[block_index](prompt)
        residual = self.stage_up_projections[stage_index](transformed)
        _require_finite(residual, name="Visual Adapter residual")
        return tokens + residual.to(dtype=tokens.dtype)


_MLP_FORWARD_ATTRIBUTES = ("fc1", "act", "drop1", "norm", "fc2", "drop2")


def _validate_mlp_module(mlp: nn.Module) -> None:
    missing = [name for name in _MLP_FORWARD_ATTRIBUTES if not hasattr(mlp, name)]
    if missing:
        raise TypeError(f"SAM3 MLP is missing required attributes: {', '.join(missing)}")
    if not isinstance(mlp.fc1, (nn.Linear, nn.Conv2d)) or not isinstance(mlp.fc2, (nn.Linear, nn.Conv2d)):
        raise TypeError("SAM3 MLP fc1/fc2 must be Linear or Conv2d modules")


def grad_compatible_mlp_forward(mlp: nn.Module, inputs: Tensor) -> Tensor:
    """Mathematically equivalent, autograd-compatible official SAM3 MLP path."""

    _validate_mlp_module(mlp)
    output = mlp.fc1(inputs)
    output = mlp.act(output)
    output = mlp.drop1(output)
    output = mlp.norm(output)
    output = mlp.fc2(output)
    return mlp.drop2(output)


def _grad_compatible_mlp_method(self: nn.Module, inputs: Tensor) -> Tensor:
    return grad_compatible_mlp_forward(self, inputs)


def validate_official_vit_contract(trunk: nn.Module) -> dict[str, Any]:
    """Fail fast if an installed official SAM3 ViT no longer matches the design."""

    blocks = getattr(trunk, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 32:
        count = len(blocks) if blocks is not None and hasattr(blocks, "__len__") else None
        raise RuntimeError(f"official SAM3 trunk must have 32 blocks, got {count}")

    patch_embed = getattr(trunk, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    if not isinstance(projection, nn.Conv2d):
        raise RuntimeError("official SAM3 patch embedding projection is missing")
    patch_size = tuple(int(value) for value in projection.kernel_size)
    patch_stride = tuple(int(value) for value in projection.stride)
    if patch_size != (14, 14) or patch_stride != (14, 14):
        raise RuntimeError(
            f"official SAM3 trunk must use patch size 14/stride 14, got {patch_size}/{patch_stride}"
        )
    if int(projection.out_channels) != 1024:
        raise RuntimeError(
            f"official SAM3 trunk must use 1024-dimensional tokens, got {projection.out_channels}"
        )
    full_attention = tuple(int(index) for index in getattr(trunk, "full_attn_ids", ()))
    if full_attention != (7, 15, 23, 31):
        raise RuntimeError(
            "official SAM3 trunk logical stages must end at blocks 7, 15, 23, and 31, "
            f"got {full_attention}"
        )
    if bool(getattr(trunk, "retain_cls_token", False)):
        raise RuntimeError("official SAM3 visual adapter requires spatial tokens without a class token")
    for index, block in enumerate(blocks):
        mlp = getattr(block, "mlp", None)
        if not isinstance(mlp, nn.Module):
            raise RuntimeError(f"official SAM3 block {index} has no MLP module")
        try:
            _validate_mlp_module(mlp)
        except TypeError as error:
            raise RuntimeError(f"official SAM3 block {index} MLP is incompatible: {error}") from error
    return {
        "depth": len(blocks),
        "embed_dim": int(projection.out_channels),
        "patch_size": patch_size[0],
        "patch_stride": patch_stride[0],
        "logical_stage_ends": list(full_attention),
    }


def install_grad_compatible_mlp_forward(trunk: nn.Module) -> tuple[int, ...]:
    """Replace only this trunk instance's inference-only fused MLP forwards."""

    validate_official_vit_contract(trunk)
    if bool(getattr(trunk, "_visual_da_grad_mlp_installed", False)):
        raise RuntimeError("grad-compatible MLP forward is already installed")
    installed: list[int] = []
    for index, block in enumerate(trunk.blocks):
        block.mlp.forward = types.MethodType(_grad_compatible_mlp_method, block.mlp)
        block.mlp._visual_da_grad_compatible = True
        installed.append(index)
    trunk._visual_da_grad_mlp_installed = True
    return tuple(installed)


def _image_tensor(inputs: object) -> Tensor:
    images = getattr(inputs, "tensors", inputs)
    if not isinstance(images, Tensor):
        raise TypeError(f"official SAM3 trunk input must contain a Tensor, got {type(images).__name__}")
    return images


def inject_visual_adapter(
    trunk: nn.Module,
    config: VisualAdapterConfig | None = None,
) -> VisualAdapterBank:
    """Attach one shared Visual Adapter bank through instance-local forward hooks."""

    validate_official_vit_contract(trunk)
    if hasattr(trunk, "visual_adapter"):
        raise RuntimeError("Visual Adapter is already injected into this SAM3 trunk")
    bank = VisualAdapterBank(config or VisualAdapterConfig())
    trunk.add_module("visual_adapter", bank)
    trunk._visual_adapter_pyramid = None

    def prepare_hook(module: nn.Module, args: tuple[object, ...]) -> None:
        if len(args) != 1:
            raise RuntimeError(f"official SAM3 trunk expected one positional input, got {len(args)}")
        module._visual_adapter_pyramid = module.visual_adapter.prepare(_image_tensor(args[0]))

    def clear_hook(module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
        module._visual_adapter_pyramid = None

    handles: list[Any] = [trunk.register_forward_pre_hook(prepare_hook)]
    for block_index, block in enumerate(trunk.blocks):

        def inject_hook(
            _module: nn.Module,
            args: tuple[object, ...],
            *,
            index: int = block_index,
        ) -> tuple[object, ...]:
            if len(args) != 1 or not isinstance(args[0], Tensor):
                raise RuntimeError("official SAM3 ViT block expected one Tensor input")
            pyramid = trunk._visual_adapter_pyramid
            if pyramid is None:
                raise RuntimeError("Visual Adapter handcrafted features were not prepared")
            return (bank.inject(args[0], pyramid, block_index=index),)

        handles.append(block.register_forward_pre_hook(inject_hook))
    handles.append(trunk.register_forward_hook(clear_hook))
    trunk._visual_adapter_hook_handles = handles
    return bank
