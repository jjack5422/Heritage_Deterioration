#!/usr/bin/env python3
"""Evaluate a saved SAM3 expert checkpoint without changing canonical run metrics."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sam2_adapter.h0_core import load_trainable_state_dict
from sam2_adapter.reporting import write_json
from sam2_adapter.runtime import _sha256
from sam3_adapter.expert_training_data import prepare_expert_data_plan
from sam3_adapter.train import THRESHOLD, _build_model, _evaluate, _loader, _write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Expert fold directory")
    parser.add_argument("--checkpoint", default="last.pt", help="Filename under artifacts/checkpoints")
    parser.add_argument("--exclude-source-group", action="append", default=[])
    return parser.parse_args()


def _metric_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mprecision": precision,
        "mrecall": recall,
        "mf1": f1,
        "miou": iou,
    }


def evaluate_checkpoint(
    run_dir: Path,
    checkpoint_name: str,
    excluded_source_groups: Sequence[str],
) -> Path:
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config" / "args.json").read_text(encoding="utf-8"))
    config["manifest"] = Path(config["manifest"])
    config["checkpoint"] = Path(config["checkpoint"])
    args = SimpleNamespace(**config)
    plan = prepare_expert_data_plan(args.manifest, args.expert)
    checkpoint_path = run_dir / "artifacts" / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the trained SAM3 evaluation path")

    output = run_dir / "evaluations" / checkpoint_path.stem
    temporary = output.with_name(f".{output.name}.incomplete")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {output}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    device = torch.device("cuda")
    model = _build_model(args, device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("expert") != plan.expert:
        raise RuntimeError("checkpoint expert does not match the run")
    if payload.get("dataset_sha256") != plan.adopted_dataset_sha256:
        raise RuntimeError("checkpoint dataset hash does not match the current manifest")
    if payload.get("split_sha256") != plan.split_sha256:
        raise RuntimeError("checkpoint split hash does not match the current manifest")
    load_trainable_state_dict(model, payload["adaptation_state"])
    loader = _loader(plan.test, raw_ids=plan.raw_ids, train=False, args=args)

    try:
        result = _evaluate(
            model,
            loader,
            plan,
            args,
            device,
            collect_rows=True,
            split="test",
        )
        rows = result.pop("per_image_rows")
        excluded = set(excluded_source_groups)
        unknown = excluded - {str(row["source_group"]) for row in rows}
        if unknown:
            raise ValueError(f"excluded source groups are not in this test split: {sorted(unknown)}")
        retained_rows = [row for row in rows if str(row["source_group"]) not in excluded]
        excluded_summary = {
            "excluded_source_groups": sorted(excluded),
            "retained_image_count": len(retained_rows),
            "tile_micro": _metric_counts(retained_rows),
        }
        source_rows = [
            {"source_group": source, **metrics}
            for source, metrics in result["by_source"].items()
        ]
        record = {
            "schema_version": 1,
            "scope": "post_training_checkpoint_evaluation_not_used_for_selection",
            "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": int(payload["epoch"]),
            "expert": plan.expert,
            "threshold": THRESHOLD,
            "test_image_count": len(rows),
            "split_sha256": plan.split_sha256,
            **result,
            "excluded_recalculation": excluded_summary,
            "per_image_metrics": "per_image.csv",
            "per_source_metrics": "per_source.csv",
        }
        _write_csv(temporary / "per_image.csv", rows)
        _write_csv(temporary / "per_source.csv", source_rows)
        write_json(temporary / "metrics.json", record)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del model
        torch.cuda.empty_cache()
    return output / "metrics.json"


def main() -> int:
    args = parse_args()
    print(evaluate_checkpoint(args.run_dir, args.checkpoint, args.exclude_source_group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
