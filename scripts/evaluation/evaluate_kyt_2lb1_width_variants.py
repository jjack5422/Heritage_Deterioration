#!/usr/bin/env python3
"""Evaluate one SAM3 craquelure checkpoint on the new 198-tile 2LB1 width variants."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2_adapter.h0_core import load_trainable_state_dict
from sam2_adapter.runtime import _autocast, _batch_tensor, _sha256
from sam3_adapter.expert_training_data import prepare_expert_data_plan
from sam3_adapter.train import THRESHOLD, _build_model, _loader

LB_GROUP = "KYT-SC-1R-2LB1-1"
EXPECTED_TEST_GROUP_COUNTS = {
    "KJWTomh-MH-M-A3E-3-2": 56,
    "KYT-SC-1R-2LB1-1": 62,
    "KYT-SC-1R-A9-4": 80,
}


def _metric_counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "gt_pixels": tp + fn,
        "pred_pixels": tp + fp,
    }


def _counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    return (
        int(np.count_nonzero(prediction & target)),
        int(np.count_nonzero(prediction & ~target)),
        int(np.count_nonzero(~prediction & target)),
    )


def _target(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        values = np.asarray(source.convert("L"), dtype=np.uint8)
    unique = set(int(value) for value in np.unique(values))
    if values.shape != (512, 512) or not unique.issubset({0, 255}):
        raise ValueError(f"invalid binary 512x512 target: {path}: shape={values.shape}, values={sorted(unique)}")
    return values == 255


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, int | float]:
    return _metric_counts(
        sum(int(row["tp"]) for row in rows),
        sum(int(row["fp"]) for row in rows),
        sum(int(row["fn"]) for row in rows),
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty metrics")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(run_dir: Path, checkpoint_name: str, manifest: Path, lb_variants: Path) -> Path:
    run_dir = run_dir.resolve()
    manifest = manifest.resolve()
    lb_variants = lb_variants.resolve()
    output = run_dir / "evaluations" / "new_test_kyt_2lb1_5px_7px"
    temporary = output.with_name(f".{output.name}.incomplete")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    if temporary.exists():
        shutil.rmtree(temporary)

    config = json.loads((run_dir / "config" / "args.json").read_text(encoding="utf-8"))
    config["manifest"] = manifest
    config["checkpoint"] = Path(config["checkpoint"])
    args = SimpleNamespace(**config)
    plan = prepare_expert_data_plan(manifest, "shrinkage_craquelure")
    current_counts = defaultdict(int)
    for row in plan.test:
        current_counts[str(row["source_group"])] += 1
    if dict(current_counts) != EXPECTED_TEST_GROUP_COUNTS:
        raise ValueError(f"new test membership drifted: {dict(current_counts)} != {EXPECTED_TEST_GROUP_COUNTS}")

    checkpoint_path = run_dir / "artifacts" / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the trained SAM3 evaluation path")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("expert") != "shrinkage_craquelure":
        raise RuntimeError("checkpoint expert is not shrinkage_craquelure")
    device = torch.device("cuda")
    model = _build_model(args, device)
    load_trainable_state_dict(model, payload["adaptation_state"])
    loader = _loader(plan.test, raw_ids=plan.raw_ids, train=False, args=args)
    per_tile: list[dict[str, Any]] = []
    row_index = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                images = _batch_tensor(batch, "image", device)
                with _autocast(device, args.amp):
                    logits = model(images)
                predictions = (torch.sigmoid(logits[:, 0]) >= THRESHOLD).cpu().numpy()
                for prediction in predictions:
                    row = plan.test[row_index]
                    row_index += 1
                    tile = str(row["tile"])
                    source_group = str(row["source_group"])
                    paths = {"7px": Path(row["mask"])}
                    paths["5px"] = (
                        lb_variants / "masks_5px" / f"{tile}.png"
                        if source_group == LB_GROUP
                        else paths["7px"]
                    )
                    for variant in ("5px", "7px"):
                        target = _target(paths[variant])
                        tp, fp, fn = _counts(prediction, target)
                        per_tile.append(
                            {
                                "checkpoint": checkpoint_name,
                                "epoch": int(payload["epoch"]),
                                "variant": variant,
                                "tile": tile,
                                "source_group": source_group,
                                **_metric_counts(tp, fp, fn),
                            }
                        )
        if row_index != 198:
            raise RuntimeError(f"evaluated {row_index} predictions; expected 198")

        variants: dict[str, Any] = {}
        for variant in ("5px", "7px"):
            variant_rows = [row for row in per_tile if row["variant"] == variant]
            variants[variant] = {
                "test_image_count": len(variant_rows),
                "tile_micro": _aggregate(variant_rows),
                "by_source": {
                    source: _aggregate([row for row in variant_rows if row["source_group"] == source])
                    for source in EXPECTED_TEST_GROUP_COUNTS
                },
            }

        temporary.mkdir(parents=True)
        _write_csv(temporary / "per_tile.csv", per_tile)
        summary = {
            "schema_version": 1,
            "scope": "post_training_alternate_test_evaluation_not_used_for_checkpoint_selection",
            "expert": "shrinkage_craquelure",
            "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": int(payload["epoch"]),
            "threshold": THRESHOLD,
            "metric": "pixel-level micro F1 = 2TP/(2TP+FP+FN)",
            "test_image_count_per_variant": 198,
            "test_group_counts": EXPECTED_TEST_GROUP_COUNTS,
            "target_rule": {
                "5px": "A9 uses active 7px D-04 only; 2LB1 uses 5px D-04 only; A3E-3-2 unchanged",
                "7px": "A9 and 2LB1 use active 7px D-04 only; A3E-3-2 unchanged",
            },
            "current_manifest": str(manifest),
            "current_dataset_sha256": plan.dataset_sha256,
            "current_adopted_dataset_sha256": plan.adopted_dataset_sha256,
            "current_split_sha256": plan.split_sha256,
            "checkpoint_adopted_dataset_sha256": payload.get("dataset_sha256"),
            "checkpoint_split_sha256": payload.get("split_sha256"),
            "intentional_contract_mismatch": {
                "dataset": payload.get("dataset_sha256") != plan.adopted_dataset_sha256,
                "split": payload.get("split_sha256") != plan.split_sha256,
                "reason": "checkpoint predates the approved A3E-2 to 2LB1 test replacement",
            },
            "variants": variants,
            "per_tile_metrics": "per_tile.csv",
        }
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del model
        torch.cuda.empty_cache()
    return output / "summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "deterioration_statistics" / "sam3_experts" / "shrinkage_craquelure.json",
    )
    parser.add_argument(
        "--lb-variants",
        type=Path,
        default=ROOT / "outputs" / "KYT-SC-1R-2LB1-1_d04_thinning_preview",
    )
    args = parser.parse_args()
    print(evaluate(args.run_dir, args.checkpoint, args.manifest, args.lb_variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
