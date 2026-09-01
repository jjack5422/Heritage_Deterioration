"""Frozen official SAM2/SAM3 feature extractors with the shared probe."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sam3_adapter.probe_decoder import SharedFpnProbe


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _FrozenProbeModel(nn.Module):
    model_input_size: int

    def __init__(self, backbone: nn.Module, *, model_input_size: int, probe_seed: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_input_size = model_input_size
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        backbone_device = next(self.backbone.parameters()).device
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(probe_seed)
            self.probe = SharedFpnProbe().to(backbone_device)
        self.probe_seed = probe_seed

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _rgb_from_dataset(self, images: Tensor) -> Tensor:
        mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

    def _prepare_backbone_input(self, images: Tensor) -> Tensor:
        raise NotImplementedError

    def _features(self, images: Tensor) -> list[Tensor]:
        raise NotImplementedError

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 512, 512):
            raise ValueError(f"source images must be Bx3x512x512, got {tuple(images.shape)}")
        prepared = self._prepare_backbone_input(images)
        with torch.no_grad():
            features = self._features(prepared)
        # SAM3's fused ViT path requires bf16 autocast, while the small trainable
        # probe is intentionally kept in fp32 for stable, identical A/B updates.
        with torch.autocast(device_type=images.device.type, enabled=False):
            logits = self.probe([feature.float() for feature in features])
        return F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


class Sam2ProbeModel(_FrozenProbeModel):
    """Official SAM2.1 Hiera-L at native 1024 input resolution."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda", probe_seed: int = 42) -> None:
        from sam2.build_sam import build_sam2

        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path=str(Path(checkpoint).resolve()),
            device=device,
            mode="eval",
            apply_postprocessing=False,
        )
        super().__init__(model.image_encoder, model_input_size=1024, probe_seed=probe_seed)

    def _prepare_backbone_input(self, images: Tensor) -> Tensor:
        rgb = F.interpolate(
            self._rgb_from_dataset(images), size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True
        )
        mean = rgb.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = rgb.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (rgb - mean) / std

    def _features(self, images: Tensor) -> list[Tensor]:
        output = self.backbone(images)
        return list(output["backbone_fpn"])


class Sam3ProbeModel(_FrozenProbeModel):
    """Official SAM3 visual backbone at native 1008 input resolution."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda", probe_seed: int = 42) -> None:
        from sam3.model_builder import build_sam3_image_model

        complete = build_sam3_image_model(
            device="cpu",
            checkpoint_path=str(Path(checkpoint).resolve()),
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=False,
        )
        vision = complete.backbone.vision_backbone
        del complete
        gc.collect()
        super().__init__(vision.to(device), model_input_size=1008, probe_seed=probe_seed)

    def _prepare_backbone_input(self, images: Tensor) -> Tensor:
        rgb = F.interpolate(
            self._rgb_from_dataset(images), size=(1008, 1008), mode="bilinear", align_corners=False, antialias=True
        )
        return (rgb - 0.5) / 0.5

    def _features(self, images: Tensor) -> list[Tensor]:
        features, _, _, _ = self.backbone(images)
        # Official SAM3 image model uses scalp=1, retaining the first three levels.
        return list(features[:-1])
