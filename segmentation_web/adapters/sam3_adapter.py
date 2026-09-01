"""Web adapter for the trained 512-input SAM3 Adapter model."""

from __future__ import annotations

from pathlib import Path

import torch

from adapters.base import TiledTorchAdapter
from adapters.checkpoint_loading import (
    CheckpointContractError,
    load_checkpoint_payload,
    validate_sam3_checkpoint,
    verify_base_checkpoint,
)
from config import Settings, settings as default_settings


class SAM3Adapter(TiledTorchAdapter):
    """Construct the author SAM3 Adapter runtime and apply trained state."""

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
        self.base_checkpoint = app_settings.sam3_base_checkpoint
        self.input_size = app_settings.sam3_input_size

    def load(self, weight_path: Path | None) -> None:
        if weight_path is None:
            raise CheckpointContractError("SAM3 Adapter requires a task checkpoint")
        if self.base_checkpoint is None:
            raise CheckpointContractError("SAM3 base checkpoint is not configured")
        if self.input_size != 512 or self.tile_size != 512:
            raise CheckpointContractError(
                "The selected SAM3 Adapter checkpoint requires 512-pixel input"
            )
        self._require_runtime_device()
        checkpoint = validate_sam3_checkpoint(load_checkpoint_payload(weight_path))
        verify_base_checkpoint(
            self.base_checkpoint,
            checkpoint.base_checkpoint_sha256,
        )

        from sam2_adapter.h0_core import load_trainable_state_dict
        from sam3_adapter.sam3_adapter_model import Sam3AdapterModel

        model = Sam3AdapterModel(
            self.base_checkpoint,
            device=self.device,
            input_size=self.input_size,
        )
        load_trainable_state_dict(model, checkpoint.adaptation_state)
        model.eval()
        self.model = model
        self.load_metadata = {
            "architecture": "SAM3 Adapter 512",
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
