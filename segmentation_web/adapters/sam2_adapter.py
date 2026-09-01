"""Web adapter for the trained prompt-free SAM2 Adapter model."""

from __future__ import annotations

from pathlib import Path

import torch

from adapters.base import TiledTorchAdapter
from adapters.checkpoint_loading import (
    CheckpointContractError,
    load_checkpoint_payload,
    validate_sam2_checkpoint,
    verify_base_checkpoint,
)
from config import Settings, settings as default_settings


class SAM2Adapter(TiledTorchAdapter):
    """Construct SAM2.1 Hiera-L and apply its trained Adapter state."""

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
        self.base_checkpoint = app_settings.sam2_base_checkpoint

    def load(self, weight_path: Path | None) -> None:
        if weight_path is None:
            raise CheckpointContractError("SAM2 Adapter requires a task checkpoint")
        if self.base_checkpoint is None:
            raise CheckpointContractError("SAM2 base checkpoint is not configured")
        self._require_runtime_device()
        checkpoint = validate_sam2_checkpoint(
            load_checkpoint_payload(weight_path),
            image_size=self.tile_size,
        )
        verify_base_checkpoint(
            self.base_checkpoint,
            checkpoint.base_checkpoint_sha256,
        )

        from sam2_adapter.adapter_model import SAM2AdapterMaskDecoder
        from sam2_adapter.h0_core import load_trainable_state_dict

        model = SAM2AdapterMaskDecoder(
            checkpoint=self.base_checkpoint,
            config=checkpoint.model_config or "configs/sam2.1/sam2.1_hiera_l.yaml",
            image_size=self.tile_size,
            device=self.device,
        )
        if model.adapter_metadata != checkpoint.adapter_metadata:
            del model
            self.unload()
            raise CheckpointContractError("SAM2 adapter metadata does not match")
        load_trainable_state_dict(model, checkpoint.adaptation_state)
        model.eval()
        self.model = model
        self.load_metadata = {
            "architecture": "SAM2.1 Hiera-L SAM2-Adapter",
            "checkpoint": weight_path.name,
        }

    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(batch)
        return torch.sigmoid(logits.float())[:, 0]
