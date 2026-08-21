"""Native SAM2 H0 model primitives.

H0 keeps SAM2 Hiera-L's image encoder, prompt encoder, and mask decoder intact.
It uses the official image-predictor path (image features -> prompt encoder with
``masks=None`` -> mask decoder) instead of SAM2's video tracking path.  The
latter applies an object-presence gate that is unsuitable for a static,
always-supervised segmentation loss.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def binary_bce_dice_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_value: int,
    dice_weight: float = 0.65,
    positive_weight: float,
    epsilon: float = 1e-6,
) -> Tensor:
    """Adapter's weighted BCE + Dice objective, excluding ignore/unlabelled pixels."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape Bx1xHxW")
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape != logits.shape:
        raise ValueError(
            f"target/logit shape mismatch: {tuple(target.shape)} vs {tuple(logits.shape)}"
        )
    valid = target != ignore_value
    if not bool(valid.any()):
        # Preserve a differentiable zero for tiles that have no supervised pixels.
        return logits.sum() * 0.0
    valid_logits = logits[valid]
    valid_target = target[valid].to(dtype=logits.dtype)
    pos_weight = torch.as_tensor(
        positive_weight,
        dtype=valid_logits.dtype,
        device=valid_logits.device,
    )
    bce = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_target,
        pos_weight=pos_weight,
    )
    probability = torch.sigmoid(valid_logits)
    dice = 1.0 - (2.0 * (probability * valid_target).sum() + epsilon) / (
        probability.square().sum() + valid_target.square().sum() + epsilon
    )
    return bce + dice_weight * dice


def expand_no_prompt_embeddings(
    sparse_embeddings: Tensor,
    dense_embeddings: Tensor,
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Broadcast SAM2's one learned no-prompt embedding to a static image batch.

    ``SAM2ImagePredictor`` invokes the native decoder one image at a time, so
    the official ``masks=None`` prompt route has batch size one.  H0 trains the
    identical route in batches; expansion shares that same learned embedding
    with each image and does not create a mask prompt.
    """

    if sparse_embeddings.shape[0] not in (1, batch_size) or dense_embeddings.shape[0] not in (1, batch_size):
        raise ValueError(
            "no-prompt embeddings must have batch size 1 or match image batch size; "
            f"got sparse={sparse_embeddings.shape[0]}, dense={dense_embeddings.shape[0]}, images={batch_size}"
        )
    if sparse_embeddings.shape[0] == batch_size and dense_embeddings.shape[0] == batch_size:
        return sparse_embeddings, dense_embeddings
    return (
        sparse_embeddings.expand(batch_size, *sparse_embeddings.shape[1:]),
        dense_embeddings.expand(batch_size, *dense_embeddings.shape[1:]),
    )


def _is_layer_norm(module: nn.Module) -> bool:
    """Recognize PyTorch LayerNorm and SAM2's channel-first LayerNorm2d."""

    return isinstance(module, nn.LayerNorm) or module.__class__.__name__ == "LayerNorm2d"


def freeze_except_layer_norm(module: nn.Module) -> tuple[str, ...]:
    """Freeze all parameters and expose only affine layer-normalization terms."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable: list[str] = []
    for module_name, child in module.named_modules():
        if not _is_layer_norm(child):
            continue
        for parameter_name, parameter in child.named_parameters(recurse=False):
            if parameter_name not in {"weight", "bias"}:
                continue
            parameter.requires_grad_(True)
            qualified = f"{module_name}.{parameter_name}" if module_name else parameter_name
            trainable.append(qualified)
    return tuple(sorted(trainable))


def trainable_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Return CPU copies of precisely the trainable adaptation parameters."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(module: nn.Module, state: dict[str, Tensor]) -> None:
    """Load an H0 LayerNorm-only checkpoint and reject accidental architecture drift."""

    parameters = dict(module.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    actual = set(state)
    if expected != actual:
        raise ValueError(
            "adaptation checkpoint parameter mismatch; "
            f"missing={sorted(expected - actual)[:5]}, unexpected={sorted(actual - expected)[:5]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


class NativeSAM2MaskDecoder(nn.Module):
    """A batchable static-image route through SAM2's unchanged native mask decoder.

    This is deliberately equivalent to the official ``SAM2ImagePredictor``'s
    feature/prompt/mask-decoder calls, with dynamic feature shapes rather than
    that helper's 1024-specific ``_bb_feat_sizes`` constant.  No mask prompt is
    supplied: SAM2's prompt encoder receives ``masks=None`` and therefore uses
    its learned ``no_mask_embed``.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        image_size: int = 512,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        if image_size <= 0 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        from sam2.build_sam import build_sam2

        self.image_size = image_size
        self.model = build_sam2(
            config,
            ckpt_path=str(Path(checkpoint).resolve()),
            device=device,
            mode="train",
            hydra_overrides_extra=[f"++model.image_size={image_size}"],
            apply_postprocessing=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one raw logit map per static image, at the input resolution."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be Bx3xHxW, got {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"images must be {self.image_size}x{self.image_size}, got {tuple(images.shape[-2:])}"
            )
        backbone_out = self.model.forward_image(images)
        _, vision_features, _, feature_sizes = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_features[-1] = vision_features[-1] + self.model.no_mem_embed

        batch_size = images.shape[0]
        features = [
            feature.permute(1, 2, 0).reshape(batch_size, feature.shape[-1], *feature_size)
            for feature, feature_size in zip(vision_features, feature_sizes)
        ]
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        sparse_embeddings, dense_embeddings = expand_no_prompt_embeddings(
            sparse_embeddings,
            dense_embeddings,
            batch_size=batch_size,
        )
        low_resolution_logits, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features[-1],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features[:-1],
        )
        return F.interpolate(
            low_resolution_logits.float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


def model_parameter_counts(module: nn.Module) -> dict[str, int]:
    """Return total/frozen/trainable parameter counts for immutable run metadata."""

    parameters: Iterable[nn.Parameter] = module.parameters()
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
