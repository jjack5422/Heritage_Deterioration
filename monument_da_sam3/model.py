"""Shared-vision dual-concept semantic segmentation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .concepts import CHANNEL_ORDER, ConceptRegistry, canonical_prompt_texts
from .sam3_integration import DEFAULT_CHECKPOINT, MOE_LAYER_INDICES, build_official_da_sam3


@dataclass(frozen=True)
class MonumentModelOutput:
    logits: Tensor
    presence_logits: Tensor
    routing: tuple[tuple[Any, ...], ...]
    vision_forward_calls: int


class MonumentDaSam3(nn.Module):
    """Run frozen vision once and two fixed concept-conditioned SAM3 paths."""

    def __init__(
        self,
        registry: ConceptRegistry,
        *,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str | torch.device = "cuda",
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.sam3, self.model_contract = build_official_da_sam3(checkpoint, device=device, rank=rank)
        self.prompts = canonical_prompt_texts(registry)
        self._vision_forward_calls = 0
        self._train_stage = "stage1"

    @property
    def adaptation_layers(self) -> tuple[nn.Module, ...]:
        return tuple(self.sam3.transformer.encoder.layers[index] for index in MOE_LAYER_INDICES)

    def set_router_temperature(self, temperature: float) -> None:
        for layer in self.adaptation_layers:
            layer.da_router_temperature = float(temperature)

    def configure_stage(self, stage: str) -> tuple[str, ...]:
        if stage not in ("stage1", "stage2"):
            raise ValueError("stage must be stage1 or stage2")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for layer in self.adaptation_layers:
            for parameter in layer.da_moe.router.parameters():
                parameter.requires_grad_(True)
            if stage == "stage1":
                for expert in layer.da_moe.experts:
                    for parameter in expert.parameters():
                        parameter.requires_grad_(True)
        if stage == "stage1":
            for layer in self.sam3.transformer.encoder.layers:
                for norm in (layer.norm1, layer.norm2, layer.norm3):
                    for parameter in norm.parameters():
                        parameter.requires_grad_(True)
        self._train_stage = stage
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> "MonumentDaSam3":
        # Frozen SAM3 stays in eval mode so DAC/matcher/dropout are never activated.
        super().train(False)
        self.training = mode
        if mode:
            for layer in self.adaptation_layers:
                layer.da_moe.train(True)
                if self._train_stage == "stage1":
                    layer.norm1.train(True)
                    layer.norm2.train(True)
                    layer.norm3.train(True)
        return self

    def _encode_text_once(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        with torch.no_grad():
            encoded = self.sam3.backbone.forward_text(list(self.prompts), device=device)
        if encoded["language_features"].shape[1] != len(CHANNEL_ORDER):
            raise RuntimeError("SAM3 text encoder did not return two prompt batches")
        return encoded

    @staticmethod
    def _concept_text(encoded: dict[str, Tensor], concept_index: int, batch_size: int) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for key, value in encoded.items():
            if key == "language_features":
                result[key] = value[:, concept_index : concept_index + 1].expand(-1, batch_size, -1)
            elif key == "language_mask":
                result[key] = value[concept_index : concept_index + 1].expand(batch_size, -1)
            elif key == "language_embeds":
                if value.ndim >= 2 and value.shape[1] == len(CHANNEL_ORDER):
                    result[key] = value[:, concept_index : concept_index + 1].expand(-1, batch_size, *value.shape[2:])
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def forward(self, images: Tensor) -> MonumentModelOutput:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 512, 512):
            raise ValueError(f"model input must be [B,3,512,512], got {tuple(images.shape)}")
        if not torch.isfinite(images).all():
            raise FloatingPointError("input contains NaN or Inf")
        batch_size = images.shape[0]
        prepared = (images - 0.5) / 0.5
        before = self._vision_forward_calls
        with torch.no_grad():
            vision = self.sam3.backbone.forward_image(prepared)
        self._vision_forward_calls += 1
        text = self._encode_text_once(batch_size, images.device)
        logits, presence_logits, routing = [], [], []
        self.sam3.eval()
        for concept_index in range(len(CHANNEL_ORDER)):
            backbone_out = {**vision, **self._concept_text(text, concept_index, batch_size)}
            find_input = self._make_find_stage(batch_size, images.device)
            output = self.sam3.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=self.sam3._get_dummy_prompt(num_prompts=batch_size),
            )
            semantic = output["semantic_seg"]
            if semantic.ndim != 4 or semantic.shape[:2] != (batch_size, 1):
                raise RuntimeError(f"unexpected semantic output for {CHANNEL_ORDER[concept_index]}: {semantic.shape}")
            logits.append(F.interpolate(semantic.float(), size=(512, 512), mode="bilinear", align_corners=False)[:, 0])
            presence = output.get("presence_logit_dec")
            if presence is None:
                raise RuntimeError("official SAM3 decoder presence logit is missing")
            presence_logits.append(presence.reshape(batch_size, -1).mean(dim=1).float())
            routing.append(tuple(layer.da_last_diagnostics for layer in self.adaptation_layers))
        stacked = torch.stack(logits, dim=1)
        stacked_presence = torch.stack(presence_logits, dim=1)
        if not torch.isfinite(stacked).all() or not torch.isfinite(stacked_presence).all():
            raise FloatingPointError("SAM3 output contains NaN or Inf")
        calls = self._vision_forward_calls - before
        if calls != 1:
            raise RuntimeError(f"vision encoder must run once per batch, observed {calls}")
        return MonumentModelOutput(stacked, stacked_presence, tuple(routing), calls)

    @staticmethod
    def _make_find_stage(batch_size: int, device: torch.device) -> Any:
        from sam3.model.data_misc import FindStage

        ids = torch.arange(batch_size, device=device, dtype=torch.long)
        return FindStage(
            img_ids=ids,
            text_ids=ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
