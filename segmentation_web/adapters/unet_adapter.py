"""Strict web adapters for trained segmentation-models-pytorch U-Nets."""

from __future__ import annotations

from pathlib import Path

import torch

from adapters.base import TiledTorchAdapter
from adapters.checkpoint_loading import (
    CheckpointContractError,
    load_checkpoint_payload,
    validate_unet_checkpoint,
)
from config import Settings, settings as default_settings


class UnetAdapter(TiledTorchAdapter):
    """Load one registered U-Net backbone from a self-describing checkpoint."""

    expected_backbone: str
    architecture_label: str

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        app_settings = settings or default_settings
        super().__init__(
            device=device,
            tile_size=app_settings.inference_tile_size,
            stride=app_settings.inference_stride,
            batch_size=app_settings.inference_batch_size,
        )

    def load(self, weight_path: Path | None) -> None:
        if weight_path is None:
            raise CheckpointContractError(
                f"{self.architecture_label} requires a checkpoint"
            )
        self._require_runtime_device()
        checkpoint = validate_unet_checkpoint(
            load_checkpoint_payload(weight_path),
            expected_backbone=self.expected_backbone,
        )

        from unet.src.unet_model import build_resunet

        model = build_resunet(
            encoder=checkpoint.encoder,
            encoder_weights=None,
            num_classes=len(checkpoint.class_names),
        )
        try:
            model.load_state_dict(checkpoint.model_state, strict=True)
        except RuntimeError as exc:
            del model
            self.unload()
            raise CheckpointContractError(
                f"{self.architecture_label} model state does not match its architecture"
            ) from exc
        model.to(self.device).eval()
        self.model = model
        self.load_metadata = {
            "architecture": self.architecture_label,
            "encoder": checkpoint.encoder,
            "checkpoint": weight_path.name,
        }

    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(batch)
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(
                f"U-Net logits must have shape Bx2xHxW, got {tuple(logits.shape)}"
            )
        return torch.softmax(logits.float(), dim=1)[:, 1]


class ResUNetAdapter(UnetAdapter):
    expected_backbone = "resnet50"
    architecture_label = "ResUNet50"


class ConvNextUnetAdapter(UnetAdapter):
    expected_backbone = "convnext_large"
    architecture_label = "ConvNeXt-Large U-Net"
