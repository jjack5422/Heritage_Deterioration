"""Web adapter for the two-channel Visual DA-SAM3 checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from adapters.base import TiledTorchAdapter
from adapters.checkpoint_loading import CheckpointContractError
from adapters.sam3_runtime import activate_official_runtime
from config import Settings, WORKSPACE_ROOT, settings as default_settings


DA_SAM3_MODEL_VARIANT = "visual_da_sam3"
DA_SAM3_FULL_PIXEL_DECODER = True
DEFAULT_CONCEPT_REGISTRY = WORKSPACE_ROOT / "dual_adapter_sam3/configs/concepts.yaml"
DEFAULT_SPLIT_CONTRACT = WORKSPACE_ROOT / "dual_adapter_sam3/configs/splits.json"


class DASAM3Adapter(TiledTorchAdapter):
    """Load Visual DA-SAM3 once and select one fixed deterioration channel."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        device: str | torch.device | None = None,
        concept_registry_path: str | Path = DEFAULT_CONCEPT_REGISTRY,
        split_contract_path: str | Path = DEFAULT_SPLIT_CONTRACT,
    ) -> None:
        app_settings = settings or default_settings
        super().__init__(
            device=device,
            tile_size=app_settings.inference_tile_size,
            stride=app_settings.inference_stride,
            batch_size=app_settings.inference_batch_size,
            normalization="zero_one",
        )
        self.base_checkpoint = app_settings.sam3_base_checkpoint
        self.input_size = app_settings.sam3_input_size
        self.concept_registry_path = Path(concept_registry_path)
        self.split_contract_path = Path(split_contract_path)

    def prepare_runtime(self) -> None:
        activate_official_runtime()

    def load(self, weight_path: Path | None) -> None:
        if weight_path is None:
            raise CheckpointContractError("DA-SAM3 requires a task checkpoint")
        if self.base_checkpoint is None:
            raise CheckpointContractError("SAM3 base checkpoint is not configured")
        if self.input_size != 512 or self.tile_size != 512:
            raise CheckpointContractError(
                "The selected DA-SAM3 checkpoint requires 512-pixel input"
            )
        self._require_runtime_device()

        from dual_adapter_sam3.checkpoints import load_adaptation_checkpoint
        from dual_adapter_sam3.concepts import load_concept_registry
        from dual_adapter_sam3.model import build_dual_adapter_model
        from dual_adapter_sam3.splits import load_split_contract

        registry = load_concept_registry(self.concept_registry_path)
        split = load_split_contract(self.split_contract_path)
        model = build_dual_adapter_model(
            registry,
            model_variant=DA_SAM3_MODEL_VARIANT,
            checkpoint=self.base_checkpoint,
            device=self.device,
            full_pixel_decoder=DA_SAM3_FULL_PIXEL_DECODER,
        ).to(self.device)
        try:
            checkpoint = load_adaptation_checkpoint(
                weight_path,
                model,
                expected_prompt_hash=registry.sha256,
                expected_split_hash=split.sha256,
            )
        except Exception:
            del model
            self.unload()
            raise
        model.eval()
        self.model = model
        self.load_metadata = {
            "architecture": "DA-SAM3",
            "model_variant": DA_SAM3_MODEL_VARIANT,
            "checkpoint": weight_path.name,
            "checkpoint_stage": checkpoint.get("stage"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "full_pixel_decoder": DA_SAM3_FULL_PIXEL_DECODER,
        }

    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("DA-SAM3 requires an explicit deterioration class")

    def _predict_batch_for_class(
        self,
        batch: torch.Tensor,
        deterioration_class: str | None,
    ) -> torch.Tensor:
        from dual_adapter_sam3.concepts import CHANNEL_ORDER

        if deterioration_class not in CHANNEL_ORDER:
            raise ValueError("Invalid deterioration class for DA-SAM3")
        assert self.model is not None
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(batch)
        if output.logits.ndim != 4 or output.logits.shape[1] != len(CHANNEL_ORDER):
            raise RuntimeError(
                "DA-SAM3 logits must have shape Bx2xHxW, "
                f"got {tuple(output.logits.shape)}"
            )
        channel_index = CHANNEL_ORDER.index(deterioration_class)
        return torch.sigmoid(output.logits.float())[:, channel_index]
