"""SAM2-Adapter primitives and dual-expert fusion.

The prompt generator follows the SAM2-Adapter paper/code contract: an FFT
high-pass pyramid is added to a projection of the frozen Hiera tokens, each
transformer block has its own bottleneck MLP, and the up-projection is shared
within a Hiera stage.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor, nn


Expert = Literal["crack", "craquelure"]


def make_expert_target(source_mask: Tensor, *, expert: Expert, ignore_value: int) -> Tensor:
    """Map 0/1/2/ignore labels to one-vs-rest supervision for an expert."""

    target = torch.full_like(source_mask, ignore_value)
    valid = source_mask != ignore_value
    target[valid] = 0
    if expert == "crack":
        target[source_mask == 1] = 1
    elif expert == "craquelure":
        target[source_mask == 2] = 1
    else:
        raise ValueError(f"unknown expert: {expert!r}")
    return target


def high_pass_filter(images: Tensor, *, rate: float = 0.25) -> Tensor:
    """Return the absolute FFT high-pass reconstruction in float32."""

    if images.ndim != 4:
        raise ValueError(f"images must be BxCxHxW, got {tuple(images.shape)}")
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be between zero and one")
    height, width = images.shape[-2:]
    half_side = max(1, int((height * width * rate) ** 0.5 // 2))
    center_y, center_x = height // 2, width // 2
    low_frequency = torch.zeros(
        (1, 1, height, width), dtype=torch.bool, device=images.device
    )
    low_frequency[
        :,
        :,
        max(0, center_y - half_side) : min(height, center_y + half_side),
        max(0, center_x - half_side) : min(width, center_x + half_side),
    ] = True
    source = images.float()
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(source, dim=(-2, -1), norm="forward"), dim=(-2, -1)
    )
    spectrum = spectrum.masked_fill(low_frequency, 0)
    reconstructed = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1), norm="forward"
    ).real
    return reconstructed.abs()


class OverlapPromptEmbed(nn.Module):
    """Overlap convolution followed by channel-last LayerNorm."""

    def __init__(self, in_channels: int, out_channels: int, *, first: bool) -> None:
        super().__init__()
        kernel_size = 7 if first else 3
        stride = 4 if first else 2
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(out_channels)
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_out")
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, images: Tensor) -> Tensor:
        features = self.projection(images).permute(0, 2, 3, 1)
        return self.norm(features)


class StageAdapterBank(nn.Module):
    """Four-stage SAM2 adapter with layer-specific tune and stage-shared up MLPs."""

    def __init__(
        self,
        *,
        stage_dims: Sequence[int],
        block_stage_indices: Sequence[int],
        scale_factor: int = 32,
        highpass_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if not stage_dims or scale_factor <= 0:
            raise ValueError("stage_dims and a positive scale_factor are required")
        if not block_stage_indices:
            raise ValueError("at least one block stage index is required")
        if min(block_stage_indices) < 0 or max(block_stage_indices) >= len(stage_dims):
            raise ValueError("block_stage_indices reference a missing stage")
        self.stage_dims = tuple(int(value) for value in stage_dims)
        self.block_stage_indices = tuple(int(value) for value in block_stage_indices)
        self.bottleneck_dims = tuple(max(1, value // scale_factor) for value in self.stage_dims)
        self.scale_factor = int(scale_factor)
        self.highpass_rate = float(highpass_rate)

        handcrafted: list[nn.Module] = []
        input_channels = 3
        for index, output_channels in enumerate(self.bottleneck_dims):
            handcrafted.append(
                OverlapPromptEmbed(input_channels, output_channels, first=index == 0)
            )
            input_channels = output_channels
        self.handcrafted_generators = nn.ModuleList(handcrafted)
        self.embedding_generators = nn.ModuleList(
            nn.Linear(stage_dim, bottleneck)
            for stage_dim, bottleneck in zip(
                self.stage_dims, self.bottleneck_dims, strict=True
            )
        )
        self.block_tune = nn.ModuleList(
            nn.Sequential(
                nn.Linear(
                    self.bottleneck_dims[stage_index],
                    self.bottleneck_dims[stage_index],
                ),
                nn.GELU(),
            )
            for stage_index in self.block_stage_indices
        )
        self.stage_up = nn.ModuleList(
            nn.Linear(bottleneck, stage_dim)
            for stage_dim, bottleneck in zip(
                self.stage_dims, self.bottleneck_dims, strict=True
            )
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def prepare_handcrafted(self, images: Tensor) -> tuple[Tensor, ...]:
        current: Tensor = high_pass_filter(images, rate=self.highpass_rate)
        outputs: list[Tensor] = []
        for generator in self.handcrafted_generators:
            channel_last = generator(current)
            outputs.append(channel_last)
            current = channel_last.permute(0, 3, 1, 2).contiguous()
        return tuple(outputs)

    def begin_stage(
        self,
        tokens: Tensor,
        handcrafted: Tensor,
        *,
        stage_index: int,
    ) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.stage_dims[stage_index]:
            raise ValueError("tokens do not match the requested adapter stage")
        embedded = self.embedding_generators[stage_index](tokens)
        if embedded.shape != handcrafted.shape:
            raise ValueError(
                "handcrafted/token prompt shape mismatch: "
                f"{tuple(handcrafted.shape)} vs {tuple(embedded.shape)}"
            )
        return handcrafted.to(dtype=embedded.dtype) + embedded

    def inject(self, tokens: Tensor, base_prompt: Tensor, *, block_index: int) -> Tensor:
        stage_index = self.block_stage_indices[block_index]
        prompt = self.stage_up[stage_index](self.block_tune[block_index](base_prompt))
        if prompt.shape != tokens.shape:
            raise ValueError(
                f"adapter prompt/token shape mismatch: {tuple(prompt.shape)} vs {tuple(tokens.shape)}"
            )
        return tokens + prompt


def dual_expert_labels(
    crack_probability: Tensor,
    craquelure_probability: Tensor,
    *,
    crack_threshold: float,
    craquelure_threshold: float,
) -> Tensor:
    """Fuse calibrated independent probabilities into exclusive 0/1/2 labels."""

    if crack_probability.shape != craquelure_probability.shape:
        raise ValueError("dual-expert probability maps must have identical shapes")
    if not 0.0 <= crack_threshold <= 1.0 or not 0.0 <= craquelure_threshold <= 1.0:
        raise ValueError("expert thresholds must be in [0, 1]")
    crack_passes = crack_probability >= crack_threshold
    craquelure_passes = craquelure_probability >= craquelure_threshold
    labels = torch.zeros_like(crack_probability, dtype=torch.long)
    labels[crack_passes & ~craquelure_passes] = 1
    labels[craquelure_passes & ~crack_passes] = 2
    overlap = crack_passes & craquelure_passes
    crack_margin = crack_probability - crack_threshold
    craquelure_margin = craquelure_probability - craquelure_threshold
    labels[overlap & (crack_margin >= craquelure_margin)] = 1
    labels[overlap & (craquelure_margin > crack_margin)] = 2
    return labels
