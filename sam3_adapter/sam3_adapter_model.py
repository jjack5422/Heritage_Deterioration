"""Controlled wrapper around the SAM3-Adapter authors' released implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "vendor_upstream_runtime"
RUNTIME_MODELS = RUNTIME_ROOT / "models"


def _activate_runtime() -> None:
    # The authors split their SAM3 branch and accompanying archive into two
    # incomplete trees. The local ignored runtime combines them and removes a
    # hard-coded credential; these paths force its modified ViT to win over the
    # separately installed pristine Meta package.
    for path in (str(RUNTIME_MODELS), str(RUNTIME_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _retarget_encoder_grid(model: nn.Module, input_size: int) -> None:
    """Retarget global-attention RoPE to a direct smaller input grid.

    The released runtime constructs the ViT with a 1008-pixel grid and
    precomputes global-block RoPE frequencies at that grid.  Its patch
    embedding itself accepts smaller tensors, but the cached frequencies do
    not.  Recomputing only these non-parameter buffers preserves checkpoint
    weights while making the 512 ablation mathematically well-defined.
    """
    if input_size == 1008:
        return
    trunk = model.image_encoder.vision_backbone.trunk
    patch_size = int(trunk.patch_embed.proj.kernel_size[0])
    grid = input_size // patch_size
    for block in trunk.blocks:
        attention = block.attn
        # Window blocks retain their fixed 24x24 local grid. Global blocks
        # carry the full-image grid and must be recomputed.
        if tuple(attention.input_size) != tuple(attention.rope_pt_size):
            attention.input_size = (grid, grid)
            attention._setup_rope_freqs()
    model.image_embedding_size = grid


class Sam3AdapterModel(nn.Module):
    """Author Adapter-injected SAM3 encoder plus pretrained SAM-family decoder."""

    model_input_size = 1008

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda", input_size: int = 1008) -> None:
        super().__init__()
        _activate_runtime()
        import models  # type: ignore[import-not-found]

        if input_size not in (512, 1008):
            raise ValueError(f"SAM3-Adapter input_size must be 512 or 1008, got {input_size}")
        config = yaml.safe_load((RUNTIME_ROOT / "configs" / "cod-sam-vit-l.yaml").read_text())
        # The author runtime derives both the encoder patch grid and native
        # decoder embedding size from these two config values.  Changing only
        # the wrapper resize would leave the decoder tied to 1008.
        config["model"]["args"]["inp_size"] = input_size
        config["model"]["args"]["encoder_mode"]["img_size"] = input_size
        model = models.make(config["model"])
        _retarget_encoder_grid(model, input_size)
        checkpoint_data = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if "model" in checkpoint_data and isinstance(checkpoint_data["model"], dict):
            checkpoint_data = checkpoint_data["model"]
        reference = model.state_dict()
        mapped: dict[str, Tensor] = {}
        skipped_shapes: dict[str, dict[str, list[int]]] = {}
        for name, value in checkpoint_data.items():
            if name.startswith("detector.backbone."):
                mapped_name = name.replace("detector.backbone.", "image_encoder.")
            elif "mask_decoder" in name:
                mapped_name = f"mask_decoder.{name.split('mask_decoder.')[-1]}"
            elif "pe_layer" in name:
                mapped_name = f"pe_layer.{name.split('pe_layer.')[-1]}"
            elif "no_mask_embed" in name:
                mapped_name = "no_mask_embed.weight"
            else:
                mapped_name = name
            if mapped_name in reference and value.shape != reference[mapped_name].shape:
                skipped_shapes[mapped_name] = {"checkpoint": list(value.shape), "model": list(reference[mapped_name].shape)}
                continue
            mapped[mapped_name] = value
        incompatible = model.load_state_dict(mapped, strict=False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not ("image_encoder" in name and "prompt_generator" not in name))
        for inactive in ("iou_prediction_head", "pred_obj_score_head"):
            head = getattr(model.mask_decoder, inactive, None)
            if head is not None:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
        # The upstream PromptGenerator creates ``depth + 1`` lightweight MLPs
        # for every stage, while ViT.forward addresses only indices
        # ``0 .. depth - 1``.  The final module in each stage is therefore a
        # dead branch (and receives no gradient even in the authors' trainer).
        # Exclude these unused parameters from the optimizer contract.
        prompt_generator = model.image_encoder.vision_backbone.trunk.prompt_generator
        for stage, depth in enumerate(prompt_generator.depths, start=1):
            unused = getattr(prompt_generator, f"lightweight_mlp{stage}_{depth}", None)
            if unused is not None:
                for parameter in unused.parameters():
                    parameter.requires_grad_(False)
        # The release enables activation checkpointing for low-memory GPUs.
        # Our locked batch of four fits a 32 GiB RTX 5090 without recomputing
        # every ViT block; disabling it changes neither outputs nor gradients.
        model.image_encoder.vision_backbone.trunk.use_act_checkpoint = False
        self.model = model.to(device)
        self.model_input_size = input_size
        self.load_report = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "shape_mismatches": skipped_shapes,
        }

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 512, 512):
            raise ValueError(f"source images must be Bx3x512x512, got {tuple(images.shape)}")
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        rgb = (images * std + mean).clamp(0.0, 1.0)
        if self.model_input_size == 512:
            prepared = rgb
        else:
            prepared = F.interpolate(rgb, size=(self.model_input_size, self.model_input_size), mode="bilinear", align_corners=False, antialias=True)
        logits = self.model((prepared - 0.5) / 0.5)
        return F.interpolate(logits.float(), size=(512, 512), mode="bilinear", align_corners=False)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    @property
    def adapter_metadata(self) -> dict[str, Any]:
        prompt = self.model.image_encoder.vision_backbone.trunk.prompt_generator
        return {
            "implementation": "SAM-Adapter authors SAM3-Adapter release",
            "tuning_stage": prompt.tuning_stage,
            "scale_factor": prompt.scale_factor,
            "input_type": prompt.input_type,
            "frequency_ratio": prompt.freq_nums,
            "load_report": self.load_report,
        }
