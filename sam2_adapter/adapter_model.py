"""Native SAM2.1 static-image model with stage-aware visual prompt adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sam2_adapter.adapter_core import StageAdapterBank
from sam2_adapter.h0_core import NativeSAM2MaskDecoder


class AdapterHieraTrunk(nn.Module):
    """Wrap Meta's Hiera trunk without modifying the vendored SAM2 package."""

    def __init__(
        self,
        trunk: nn.Module,
        *,
        scale_factor: int = 32,
        highpass_rate: float = 0.25,
    ) -> None:
        super().__init__()
        required = ("patch_embed", "blocks", "stage_ends", "_get_pos_embed", "channel_list")
        missing = [name for name in required if not hasattr(trunk, name)]
        if missing:
            raise TypeError(f"Hiera trunk is missing required attributes: {missing}")
        self.base = trunk
        self.channel_list = trunk.channel_list
        self.return_interm_layers = bool(trunk.return_interm_layers)
        input_dims = tuple(int(block.dim) for block in trunk.blocks)
        stage_dims = tuple(dict.fromkeys(input_dims))
        stage_lookup = {dimension: index for index, dimension in enumerate(stage_dims)}
        block_stage_indices = tuple(stage_lookup[dimension] for dimension in input_dims)
        self.adapters = StageAdapterBank(
            stage_dims=stage_dims,
            block_stage_indices=block_stage_indices,
            scale_factor=scale_factor,
            highpass_rate=highpass_rate,
        )
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: Tensor) -> list[Tensor]:
        restored = (images.float() * self.image_std + self.image_mean).clamp_(0.0, 1.0)
        handcrafted = self.adapters.prepare_handcrafted(restored)
        tokens = self.base.patch_embed(images)
        tokens = tokens + self.base._get_pos_embed(tokens.shape[1:3])
        outputs: list[Tensor] = []
        active_stage = -1
        stage_prompt: Tensor | None = None
        for block_index, block in enumerate(self.base.blocks):
            stage_index = self.adapters.block_stage_indices[block_index]
            if stage_index != active_stage:
                active_stage = stage_index
                stage_prompt = self.adapters.begin_stage(
                    tokens, handcrafted[stage_index], stage_index=stage_index
                )
            assert stage_prompt is not None
            tokens = self.adapters.inject(tokens, stage_prompt, block_index=block_index)
            tokens = block(tokens)
            if (block_index == self.base.stage_ends[-1]) or (
                block_index in self.base.stage_ends and self.return_interm_layers
            ):
                outputs.append(tokens.permute(0, 3, 1, 2))
        return outputs


def configure_adapter_training(sam_model: nn.Module) -> tuple[str, ...]:
    """Freeze SAM2 except visual adapters and the native mask decoder."""

    for parameter in sam_model.parameters():
        parameter.requires_grad_(False)
    trunk = sam_model.image_encoder.trunk
    if not isinstance(trunk, AdapterHieraTrunk):
        raise TypeError("image encoder trunk has not been wrapped with AdapterHieraTrunk")
    for parameter in trunk.adapters.parameters():
        parameter.requires_grad_(True)
    for parameter in sam_model.sam_mask_decoder.parameters():
        parameter.requires_grad_(True)
    # These confidence heads do not contribute to the single-mask logit used by
    # this unprompted segmentation objective, so keeping them nominally
    # trainable would produce optimizer parameters with no gradients.
    for inactive_head in ("iou_prediction_head", "pred_obj_score_head"):
        head = getattr(sam_model.sam_mask_decoder, inactive_head, None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(False)
    return tuple(
        name for name, parameter in sam_model.named_parameters() if parameter.requires_grad
    )


class SAM2AdapterMaskDecoder(NativeSAM2MaskDecoder):
    """Batchable, unprompted SAM2.1 Hiera-L with trainable adapters/decoder."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        image_size: int = 512,
        device: str | torch.device = "cuda",
        scale_factor: int = 32,
        highpass_rate: float = 0.25,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,
            config=config,
            image_size=image_size,
            device=device,
        )
        original_trunk = self.model.image_encoder.trunk
        self.model.image_encoder.trunk = AdapterHieraTrunk(
            original_trunk,
            scale_factor=scale_factor,
            highpass_rate=highpass_rate,
        ).to(device)
        self.trainable_names = configure_adapter_training(self.model)

    @property
    def adapter_metadata(self) -> dict[str, Any]:
        adapters = self.model.image_encoder.trunk.adapters
        return {
            "stage_dims": list(adapters.stage_dims),
            "bottleneck_dims": list(adapters.bottleneck_dims),
            "block_stage_indices": list(adapters.block_stage_indices),
            "scale_factor": adapters.scale_factor,
            "highpass_rate": adapters.highpass_rate,
        }
