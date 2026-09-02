"""Two-step, artifact-free architecture smoke for ``visual_da_sam3``."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dual_adapter_sam3.concepts import load_concept_registry
from dual_adapter_sam3.losses import multilabel_objective
from dual_adapter_sam3.model import DualAdapterSam3Output, build_dual_adapter_model
from dual_adapter_sam3.sam3_integration import DEFAULT_CHECKPOINT
from dual_adapter_sam3.training import (
    assert_finite_gradients,
    build_optimizer_and_scheduler,
    set_reproducible_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONCEPTS = PROJECT_ROOT / "dual_adapter_sam3" / "configs" / "concepts.yaml"


def _synthetic_batch(batch_size: int, device: torch.device) -> dict[str, Tensor]:
    images = torch.rand(batch_size, 3, 512, 512, device=device)
    targets = torch.zeros(batch_size, 2, 512, 512, device=device)
    targets[:, 0, 32:480, 48:464:32] = 1.0
    targets[:, 0, 48:464:32, 32:480] = 1.0
    targets[:, 1, 128:384, 160:352] = 1.0
    valid_masks = torch.ones_like(targets, dtype=torch.bool)
    class_present = targets.flatten(2).any(dim=2)
    return {
        "image": images,
        "targets": targets,
        "valid_masks": valid_masks,
        "class_present": class_present,
    }


def _routing_terms(output: DualAdapterSam3Output) -> list[tuple[Tensor, Tensor, Tensor]]:
    return [
        (diagnostics.logits, diagnostics.soft_probabilities, diagnostics.hard_assignments)
        for concept in output.routing
        for diagnostics in concept
    ]


def _require_finite_output(output: DualAdapterSam3Output) -> None:
    tensors = [output.logits, output.presence_logits]
    for concept in output.routing:
        for diagnostics in concept:
            tensors.extend(
                [
                    diagnostics.logits,
                    diagnostics.soft_probabilities,
                    diagnostics.hard_assignments,
                    diagnostics.weights,
                    diagnostics.entropy,
                    diagnostics.top1_usage,
                    diagnostics.top2_usage,
                ]
            )
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError("visual_da_sam3 smoke output contains non-finite tensors")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_smoke(
    args: argparse.Namespace,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if args.batch_size != 4 or args.steps != 2:
        raise ValueError("approved visual_da_sam3 smoke requires batch_size=4 and steps=2")
    if args.max_vram_gib <= 0:
        raise ValueError("max_vram_gib must be positive")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real visual_da_sam3 smoke test")
        device = torch.device("cuda")

    set_reproducible_seed(args.seed)
    registry = load_concept_registry(args.concepts)
    model = build_dual_adapter_model(
        registry,
        model_variant="visual_da_sam3",
        checkpoint=args.checkpoint,
        device=device,
    ).to(device)
    optimizer_bundle = build_optimizer_and_scheduler(model, stage="stage1", epochs=args.steps)
    model.set_router_temperature(2.0)
    batch = _synthetic_batch(args.batch_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    vision_calls: list[int] = []
    output_shape: list[int] | None = None
    for _step in range(args.steps):
        model.train(True)
        optimizer_bundle.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            output = model(batch["image"])
            loss = multilabel_objective(
                output.logits,
                output.presence_logits,
                batch["targets"],
                batch["valid_masks"],
                batch["class_present"],
                routing=_routing_terms(output),
            )
        expected_shape = (args.batch_size, 2, 512, 512)
        if tuple(output.logits.shape) != expected_shape:
            raise RuntimeError(
                f"visual_da_sam3 smoke output shape mismatch: {tuple(output.logits.shape)}"
            )
        if output.vision_forward_calls != 1:
            raise RuntimeError(
                "visual_da_sam3 must run the image backbone once per batch, "
                f"observed {output.vision_forward_calls}"
            )
        _require_finite_output(output)
        if not torch.isfinite(loss.total):
            raise FloatingPointError("visual_da_sam3 smoke loss is non-finite")
        loss.total.backward()
        assert_finite_gradients(model)
        optimizer_bundle.optimizer.step()
        losses.append(float(loss.total.detach()))
        vision_calls.append(int(output.vision_forward_calls))
        output_shape = list(output.logits.shape)

    visual_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "visual_adapter." in name
    }
    if not visual_parameters:
        raise RuntimeError("visual_da_sam3 smoke found no Visual Adapter parameters")
    missing_visual_gradients = [
        name
        for name, parameter in visual_parameters.items()
        if parameter.grad is None
        or not torch.isfinite(parameter.grad).all()
        or torch.count_nonzero(parameter.grad) == 0
    ]
    if missing_visual_gradients:
        raise RuntimeError(
            "second smoke backward did not reach every Visual Adapter parameter: "
            + ", ".join(missing_visual_gradients)
        )
    frozen_with_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_gradients:
        raise RuntimeError(
            "frozen SAM3 parameters received gradients: " + ", ".join(frozen_with_gradients)
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = 0
        peak_reserved = 0
    limit_bytes = int(args.max_vram_gib * 1024**3)
    if peak_allocated >= limit_bytes:
        raise RuntimeError(
            f"visual_da_sam3 peak allocated VRAM {peak_allocated / 1024**3:.2f} GiB "
            f"exceeds limit {args.max_vram_gib:.2f} GiB"
        )

    return {
        "status": "passed",
        "model_variant": "visual_da_sam3",
        "batch_size": args.batch_size,
        "steps": args.steps,
        "output_shape": output_shape,
        "vision_forward_calls": vision_calls,
        "losses": losses,
        "visual_parameter_tensors": len(visual_parameters),
        "visual_gradient_tensors": len(visual_parameters) - len(missing_visual_gradients),
        "frozen_gradient_tensors": len(frozen_with_gradients),
        "peak_allocated_mib": peak_allocated / 1024**2,
        "peak_reserved_mib": peak_reserved / 1024**2,
        "max_vram_gib": args.max_vram_gib,
        "artifacts_written": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-vram-gib", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def main() -> None:
    summary = run_smoke(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
