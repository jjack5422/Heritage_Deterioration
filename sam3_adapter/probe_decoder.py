"""Shared prompt-free segmentation probe for frozen SAM2 and SAM3 features."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SharedFpnProbe(nn.Module):
    """A small, identical top-down FPN probe for three 256-channel feature maps."""

    def __init__(self, in_channels: tuple[int, int, int] = (256, 256, 256), width: int = 128) -> None:
        super().__init__()
        if len(in_channels) != 3 or width <= 0:
            raise ValueError("the controlled probe requires three feature levels and positive width")
        self.lateral = nn.ModuleList(nn.Conv2d(channels, width, 1) for channels in in_channels)
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.GroupNorm(8, width),
                nn.GELU(),
            )
            for _ in in_channels
        )
        self.output = nn.Conv2d(width, 1, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        if len(features) != 3:
            raise ValueError(f"expected three feature maps, got {len(features)}")
        projected = [layer(feature.float()) for layer, feature in zip(self.lateral, features, strict=True)]
        fused = self.refine[-1](projected[-1])
        for index in (1, 0):
            fused = F.interpolate(fused, size=projected[index].shape[-2:], mode="bilinear", align_corners=False)
            fused = self.refine[index](fused + projected[index])
        return self.output(fused)
