"""Run three F1-selected SAM3-Adapter experts on dataset images without GT evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from crackseg_common.checkpoints import load_trainable_state_dict
from model_projects.sam3_adapter.model import Sam3AdapterModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = REPOSITORY_ROOT / "dataset"
DEFAULT_CHECKPOINT_ROOT = (
    REPOSITORY_ROOT
    / "model_projects/sam3_adapter/runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42/1fold"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model_projects/sam3_adapter/runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42/inference/dataset"
)
DEFAULT_BASE_CHECKPOINT = REPOSITORY_ROOT / "segment-anything-3/checkpoints/sam3.pt"
EXPERTS = ("loss", "shrinkage_craquelure", "scratch_crack")
EXPECTED_IMAGES = 456
EXPECTED_GROUPS = 7
THRESHOLD = 0.5
MODEL_INPUT_SIZE = 1008
BATCH_SIZE = 4
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _images(dataset: Path) -> list[tuple[str, Path]]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    rows = sorted(
        (
            image_dir.parent.name,
            path.resolve(),
        )
        for image_dir in dataset.glob("*/*/image_tiles")
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    groups = {group for group, _ in rows}
    if len(rows) != EXPECTED_IMAGES or len(groups) != EXPECTED_GROUPS:
        raise ValueError(
            f"dataset inference contract requires {EXPECTED_IMAGES} images from "
            f"{EXPECTED_GROUPS} groups; found {len(rows)} images from {len(groups)} groups"
        )
    return rows


def _load_checkpoint(path: Path, expert: str, base_checkpoint_hash: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid checkpoint payload: {path}")
    if payload.get("expert") != expert:
        raise ValueError(f"checkpoint expert mismatch: {path}")
    if payload.get("selection_metric") != "validation_pixel_micro_f1":
        raise ValueError(f"checkpoint was not selected by validation F1: {path}")
    if payload.get("selection_threshold") != THRESHOLD:
        raise ValueError(f"checkpoint threshold mismatch: {path}")
    if payload.get("base_checkpoint_sha256") != base_checkpoint_hash:
        raise ValueError(f"base checkpoint hash mismatch: {path}")
    state = payload.get("adaptation_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no adaptation state: {path}")
    return payload


def _batch_tensor(images: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    normalized = [
        torch.from_numpy(((image.astype(np.float32) / 255.0 - MEAN) / STD).copy())
        .permute(2, 0, 1)
        for image in images
    ]
    return torch.stack(normalized).to(device, non_blocking=True)


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    color = np.asarray((6, 182, 212), dtype=np.float32)
    result[mask] = result[mask] * 0.55 + color * 0.45
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def _run_expert(
    *,
    expert: str,
    image_rows: Sequence[tuple[str, Path]],
    checkpoint_root: Path,
    output_root: Path,
    base_checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = checkpoint_root / expert / "fold0/artifacts/checkpoints/best.pt"
    checkpoint_hash = _sha256(checkpoint)
    base_checkpoint_hash = _sha256(base_checkpoint)
    payload = _load_checkpoint(checkpoint, expert, base_checkpoint_hash)
    expert_output = output_root / expert
    if expert_output.exists() and any(expert_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite inference output: {expert_output}")

    model = Sam3AdapterModel(base_checkpoint, device=device, input_size=MODEL_INPUT_SIZE)
    load_trainable_state_dict(model, payload["adaptation_state"])
    model.eval()
    records: list[dict[str, object]] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(image_rows), BATCH_SIZE):
                batch_rows = image_rows[start : start + BATCH_SIZE]
                arrays: list[np.ndarray] = []
                for _, image_path in batch_rows:
                    with Image.open(image_path) as source:
                        if source.size != (512, 512):
                            raise ValueError(f"expected 512x512 source image: {image_path}")
                        arrays.append(np.asarray(source.convert("RGB"), dtype=np.uint8).copy())
                batch = _batch_tensor(arrays, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    probabilities = torch.sigmoid(model(batch).float())[:, 0]
                masks = (probabilities >= THRESHOLD).cpu().numpy()
                for (group, image_path), image, mask in zip(
                    batch_rows,
                    arrays,
                    masks,
                    strict=True,
                ):
                    relative_mask = Path("mask_tiles") / group / f"{image_path.stem}.png"
                    relative_overlay = Path("overlay_tiles") / group / f"{image_path.stem}.png"
                    mask_path = expert_output / relative_mask
                    overlay_path = expert_output / relative_overlay
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
                    Image.fromarray(_overlay(image, mask), mode="RGB").save(overlay_path)
                    records.append(
                        {
                            "expert": expert,
                            "source_group": group,
                            "source_image": str(image_path),
                            "source_sha256": _sha256(image_path),
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": checkpoint_hash,
                            "selected_epoch": int(payload["epoch"]),
                            "validation_f1": float(payload["best_validation_f1"]),
                            "model_input_size": MODEL_INPUT_SIZE,
                            "threshold": THRESHOLD,
                            "predicted_positive_pixels": int(mask.sum()),
                            "mask": str(relative_mask),
                            "mask_sha256": _sha256(mask_path),
                            "overlay": str(relative_overlay),
                            "overlay_sha256": _sha256(overlay_path),
                        }
                    )
    finally:
        del model
        torch.cuda.empty_cache()

    manifest_path = expert_output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {
        "expert": expert,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": int(payload["epoch"]),
        "validation_f1": float(payload["best_validation_f1"]),
        "image_count": len(records),
        "predicted_positive_pixels": sum(int(row["predicted_positive_pixels"]) for row in records),
        "manifest": str(manifest_path.resolve()),
        "supervision_note": (
            "crack ID 1 only; scratch annotations unavailable"
            if expert == "scratch_crack"
            else None
        ),
    }


def run_inference(
    *,
    dataset: Path,
    checkpoint_root: Path,
    output: Path,
    base_checkpoint: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3-Adapter inference requires CUDA")
    image_rows = _images(dataset.resolve())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    summaries = [
        _run_expert(
            expert=expert,
            image_rows=image_rows,
            checkpoint_root=checkpoint_root.resolve(),
            output_root=output,
            base_checkpoint=base_checkpoint.resolve(),
            device=device,
        )
        for expert in EXPERTS
    ]
    summary = {
        "schema_version": 1,
        "mode": "inference_only_no_ground_truth_evaluation",
        "dataset": str(dataset.resolve()),
        "dataset_image_count": len(image_rows),
        "dataset_source_group_count": len({group for group, _ in image_rows}),
        "model_input_size": MODEL_INPUT_SIZE,
        "source_output_size": 512,
        "threshold": THRESHOLD,
        "experts": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_inference(
                dataset=args.dataset,
                checkpoint_root=args.checkpoint_root,
                output=args.output,
                base_checkpoint=args.base_checkpoint,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
