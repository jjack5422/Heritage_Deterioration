"""Official SAM3 construction, 512 retargeting, and in-place DA-MoE injection."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .da_moe import DaMoeFfn
from .visual_adapter import (
    VisualAdapterConfig,
    inject_visual_adapter,
    install_grad_compatible_mlp_forward,
    validate_official_vit_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = PROJECT_ROOT / "segment-anything-3"
DEFAULT_CHECKPOINT = SAM3_ROOT / "checkpoints" / "sam3.pt"
MOE_LAYER_INDICES = (0, 1, 2)


class Sam3IntegrationError(RuntimeError):
    """Raised when the installed official SAM3 cannot satisfy the fixed contract."""


def activate_official_sam3() -> None:
    path = str(SAM3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retarget_vision_to_512(model: nn.Module) -> dict[str, Any]:
    trunk = model.backbone.vision_backbone.trunk
    patch_size = int(trunk.patch_embed.proj.kernel_size[0])
    grid = 512 // patch_size
    changed: list[int] = []
    for index, block in enumerate(trunk.blocks):
        attention = block.attn
        if tuple(attention.input_size) != tuple(attention.rope_pt_size):
            attention.input_size = (grid, grid)
            attention._setup_rope_freqs()
            changed.append(index)
    if grid != 36 or len(changed) != 4:
        raise Sam3IntegrationError(f"unexpected 512 ViT retarget: grid={grid}, global_blocks={changed}")
    decoder = model.transformer.decoder
    decoder.compilable_cord_cache = None
    decoder.compilable_stored_size = None
    decoder.coord_cache.clear()
    return {"input_size": 512, "patch_size": patch_size, "token_grid": [grid, grid], "global_blocks": changed}


def _pool_concept(memory: Tensor, padding_mask: Tensor | None) -> Tensor:
    if padding_mask is None:
        return memory.mean(dim=1)
    valid = (~padding_mask).to(memory.dtype).unsqueeze(-1)
    return (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _da_forward_pre(
    self: nn.Module,
    tgt: Tensor,
    memory: Tensor,
    dac: bool = False,
    tgt_mask: Tensor | None = None,
    memory_mask: Tensor | None = None,
    tgt_key_padding_mask: Tensor | None = None,
    memory_key_padding_mask: Tensor | None = None,
    pos: Tensor | None = None,
    query_pos: Tensor | None = None,
) -> Tensor:
    if dac:
        raise Sam3IntegrationError("DAC is forbidden in frozen-decoder DA-SAM3 training")
    tgt2 = self.norm1(tgt)
    q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
    tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt2 = self.norm2(tgt)
    tgt2 = self.cross_attn_image(
        query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
        key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
        value=memory,
        attn_mask=memory_mask,
        key_padding_mask=memory_key_padding_mask,
    )[0]
    tgt = tgt + self.dropout2(tgt2)
    normalized = self.norm3(tgt)
    concept = _pool_concept(memory, memory_key_padding_mask)
    transformed, diagnostics = self.da_moe(
        normalized,
        normalized,
        concept,
        temperature=float(self.da_router_temperature),
    )
    self.da_last_diagnostics = diagnostics
    return tgt + self.dropout3(transformed)


def inject_da_moe(model: nn.Module, *, rank: int = 8, experts: int = 4, top_k: int = 2) -> tuple[int, ...]:
    layers = model.transformer.encoder.layers
    if len(layers) != 6:
        raise Sam3IntegrationError(f"expected six SAM3 fusion layers, got {len(layers)}")
    for layer_index in MOE_LAYER_INDICES:
        layer = layers[layer_index]
        layer.da_moe = DaMoeFfn(
            layer.linear1,
            layer.linear2,
            activation=layer.activation,
            dropout=layer.dropout,
            rank=rank,
            experts=experts,
            top_k=top_k,
        )
        layer.da_router_temperature = 2.0
        layer.da_last_diagnostics = None
        layer.forward_pre = types.MethodType(_da_forward_pre, layer)
    actual = tuple(index for index, layer in enumerate(layers) if hasattr(layer, "da_moe"))
    if actual != MOE_LAYER_INDICES:
        raise Sam3IntegrationError(f"DA-MoE injection mismatch: {actual}")
    return actual


def build_official_da_sam3(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device = "cuda",
    rank: int = 8,
) -> tuple[nn.Module, dict[str, Any]]:
    activate_official_sam3()
    from sam3.model_builder import build_sam3_image_model

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise Sam3IntegrationError(f"missing official SAM3 checkpoint: {checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        eval_mode=True,
        device=str(device),
        compile=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    retarget = retarget_vision_to_512(model)
    injected = inject_da_moe(model, rank=rank)
    model.eval()
    contract = {
        "model_variant": "da_sam3",
        "implementation": "official facebookresearch/sam3 concept-conditioned image model",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "retarget": retarget,
        "fusion_layer_count": len(model.transformer.encoder.layers),
        "moe_layer_indices": list(injected),
        "experts": 4,
        "top_k": 2,
        "rank": rank,
    }
    return model, contract


def build_official_visual_da_sam3(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device = "cuda",
    rank: int = 8,
    visual_config: VisualAdapterConfig | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the explicit hybrid Visual Adapter + DA-MoE SAM3 variant."""

    activate_official_sam3()
    from sam3.model_builder import build_sam3_image_model

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise Sam3IntegrationError(f"missing official SAM3 checkpoint: {checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        eval_mode=True,
        device=str(device),
        compile=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    retarget = retarget_vision_to_512(model)
    trunk = model.backbone.vision_backbone.trunk
    vit_contract = validate_official_vit_contract(trunk)
    grad_mlp_blocks = install_grad_compatible_mlp_forward(trunk)
    config = visual_config or VisualAdapterConfig()
    visual_adapter = inject_visual_adapter(trunk, config)
    injected = inject_da_moe(model, rank=rank)
    model.eval()
    contract = {
        "model_variant": "visual_da_sam3",
        "implementation": "official facebookresearch/sam3 concept-conditioned image model",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "retarget": retarget,
        "vit": vit_contract,
        "grad_compatible_mlp_blocks": list(grad_mlp_blocks),
        "visual_adapter": visual_adapter.config.to_contract(),
        "visual_adapter_blocks": list(range(config.depth)),
        "fusion_layer_count": len(model.transformer.encoder.layers),
        "moe_layer_indices": list(injected),
        "experts": 4,
        "top_k": 2,
        "rank": rank,
    }
    return model, contract
