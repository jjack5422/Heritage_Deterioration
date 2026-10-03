"""Web adapter for the trained 1008-input SAM3 deterioration experts."""

from __future__ import annotations

from pathlib import Path

import torch

from adapters.base import TiledTorchAdapter
from adapters.checkpoint_loading import (
    CheckpointContractError,
    load_expert_checkpoint,
    verify_base_checkpoint,
)
from adapters.sam3_runtime import activate_vendor_runtime
from config import Settings, settings as default_settings


class SAM3Adapter(TiledTorchAdapter):
    """Construct the author SAM3 Adapter runtime and apply trained state."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        device: str | torch.device | None = None,
        expert: str = "shrinkage_craquelure",
    ) -> None:
        app_settings = settings or default_settings
        super().__init__(
            device=device,
            tile_size=app_settings.inference_tile_size,
            stride=app_settings.inference_stride,
            batch_size=app_settings.inference_batch_size,
        )
        self.base_checkpoint = app_settings.sam3_base_checkpoint
        self.expert = expert

    def prepare_runtime(self) -> None:
        activate_vendor_runtime()

    def load(self, weight_path: Path | None) -> None:
        if weight_path is None:
            raise CheckpointContractError("SAM3 Adapter requires a task checkpoint")
        if self.base_checkpoint is None:
            raise CheckpointContractError("SAM3 base checkpoint is not configured")
        if self.tile_size != 512:
            raise CheckpointContractError(
                "SAM3 experts require 512-pixel source tiles"
            )
        self._require_runtime_device()
        expert_checkpoint = load_expert_checkpoint(
            weight_path, expected_expert=self.expert, architecture="sam3_adapter"
        )
        checkpoint = expert_checkpoint.adaptation
        verify_base_checkpoint(
            self.base_checkpoint,
            checkpoint.base_checkpoint_sha256,
        )

        from sam2_adapter.h0_core import load_trainable_state_dict
        from sam3_adapter.sam3_adapter_model import Sam3AdapterModel

        model = Sam3AdapterModel(
            self.base_checkpoint,
            device=self.device,
            input_size=expert_checkpoint.input_size,
        )
        load_trainable_state_dict(model, checkpoint.adaptation_state)
        model.eval()
        self.model = model
        self.load_metadata = {
            "architecture": "SAM3 Adapter 1008",
            "checkpoint": weight_path.name,
            "expert": self.expert,
            "model_input_size": expert_checkpoint.input_size,
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
