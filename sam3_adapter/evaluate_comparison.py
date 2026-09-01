"""Re-evaluate clean outer tests and build the A/B/C/D comparison summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import random
from pathlib import Path
from typing import Any

import torch

from sam2_adapter.adapter_core import make_binary_target
from sam2_adapter.data import prepare_data_plan
from sam2_adapter.reporting import write_json
from sam2_adapter.runtime import _autocast, _batch_tensor, _make_loader
from sam3_adapter.losses import weighted_bce_dice_loss
from sam3_adapter.train_probe import _build, _counts, parse_args


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
EXPERIMENT = "2026-08-26_sam2-sam3-native-probe-adapter_seed42"
DATASET = WORKSPACE_ROOT / "datasets" / "dataset_clean_v2_merged_craquelure"
GROUPS = {
    "A": ("sam2_probe", "sam2_probe"),
    "B": ("sam3_probe", "sam3_probe"),
    "D": ("sam3_adapter", "sam3_adapter"),
}
C_ROOT = WORKSPACE_ROOT / "sam2_adapter" / "runs" / (
    "2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42"
) / "5fold" / "foreground"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _image_row(name: str, source_group: str, counts: tuple[int, int, int]) -> dict[str, Any]:
    tp, fp, fn = counts
    gt_pixels = tp + fn
    pred_pixels = tp + fp
    return {
        "image": name,
        "source_group": source_group,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("image", "source_group", "tp", "fp", "fn", "gt_pixels", "pred_pixels", "precision", "recall", "f1", "iou")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_source_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("source_group", "image_count", "tp", "fp", "fn", "precision", "recall", "f1", "iou")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in sorted({str(row["source_group"]) for row in rows}):
        selected = [row for row in rows if row["source_group"] == source]
        counts = {key: sum(int(row[key]) for row in selected) for key in ("tp", "fp", "fn")}
        output.append({
            "source_group": source,
            "image_count": len(selected),
            **counts,
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "f1": _ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
            "iou": _ratio(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"]),
        })
    return output


def _reevaluate(group: str, model_group: str, experiment_root: Path, device: torch.device) -> dict[str, Any]:
    rows_all: list[dict[str, Any]] = []
    for fold in range(5):
        fold_root = experiment_root / "5fold" / model_group / f"fold{fold}"
        args = parse_args(["--group", model_group, "--folds", str(fold), "--batch-size", "4", "--accumulation-steps", "1", "--experiment-id", EXPERIMENT])
        plan = prepare_data_plan(DATASET, outer_fold=fold)
        model = _build(args, device, fold=fold)
        payload = torch.load(fold_root / "artifacts" / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        from sam2_adapter.h0_core import load_trainable_state_dict
        load_trainable_state_dict(model, payload["adaptation_state"])
        loader = _make_loader(plan, plan.test, train=False, args=args)
        rows: list[dict[str, Any]] = []
        losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = _batch_tensor(batch, "image", device)
                target = make_binary_target(_batch_tensor(batch, "mask", device).long(), ignore_value=plan.ignore_value)
                with _autocast(device, args.amp):
                    logits = model(images)
                    losses.append(float(weighted_bce_dice_loss(logits, target, ignore_value=plan.ignore_value).cpu()))
                for index, name in enumerate(batch["name"]):
                    rows.append(_image_row(str(name), str(batch["source_group"][index]), _counts(logits[index:index + 1], target[index:index + 1], plan.ignore_value)))
        _write_rows(fold_root / "metrics" / "outer_test_per_image.csv", rows)
        source_rows = _source_summary(rows)
        _write_source_rows(fold_root / "metrics" / "outer_test_per_source.csv", source_rows)
        metric_path = fold_root / "metrics" / "outer_test_metrics.json"
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        metric["per_image_metrics"] = "metrics/outer_test_per_image.csv"
        metric["per_source_metrics"] = "metrics/outer_test_per_source.csv"
        write_json(metric_path, metric)
        rows_all.extend(rows)
        del model
        torch.cuda.empty_cache()
    source_all = _source_summary(rows_all)
    return {"group": group, "image_count": len(rows_all), "source_group": source_all}


def _read_group_metrics(root: Path, group: str, folder: str) -> dict[str, Any]:
    records = [json.loads((root / "5fold" / folder / f"fold{i}" / "metrics" / "outer_test_metrics.json").read_text(encoding="utf-8")) for i in range(5)]
    datasets = [json.loads((root / "5fold" / folder / f"fold{i}" / "config" / "dataset.json").read_text(encoding="utf-8")) for i in range(5)]
    f1 = [float(r["tile_micro"]["mf1"]) for r in records]
    iou = [float(r["tile_micro"]["miou"]) for r in records]
    precision = [float(r["tile_micro"]["mprecision"]) for r in records]
    recall = [float(r["tile_micro"]["mrecall"]) for r in records]
    accuracy = [float(r["tile_micro"]["maccuracy"]) for r in records]
    counts = {key: sum(int(r["tile_micro"][key]) for r in records) for key in ("tp", "fp", "fn")}
    valid_pixels = [
        int(record["pixel_counts"]["outer_test"]["background"])
        + int(record["pixel_counts"]["outer_test"]["foreground"])
        for record in datasets
    ]
    return {
        "group": group,
        "fold_f1": f1,
        "fold_iou": iou,
        "fold_precision": precision,
        "fold_recall": recall,
        "fold_accuracy": accuracy,
        "mean_f1": statistics.mean(f1),
        "std_f1": statistics.stdev(f1),
        "mean_iou": statistics.mean(iou),
        "std_iou": statistics.stdev(iou),
        "mean_precision": statistics.mean(precision),
        "mean_recall": statistics.mean(recall),
        "mean_accuracy": statistics.mean(accuracy),
        "worst_f1": min(f1),
        "best_f1": max(f1),
        "f1_range": max(f1) - min(f1),
        "pooled_f1": _ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
        "pooled_iou": _ratio(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"]),
        "pooled_accuracy": 1.0 - (counts["fp"] + counts["fn"]) / sum(valid_pixels),
    }


def _paired(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    d_f1 = [y - x for x, y in zip(a["fold_f1"], b["fold_f1"], strict=True)]
    d_iou = [y - x for x, y in zip(a["fold_iou"], b["fold_iou"], strict=True)]
    return {"left": a["group"], "right": b["group"], "f1_difference_right_minus_left": d_f1, "iou_difference_right_minus_left": d_iou, "f1_mean_difference": statistics.mean(d_f1), "iou_mean_difference": statistics.mean(d_iou), "f1_wins_right": sum(x > 0 for x in d_f1), "iou_wins_right": sum(x > 0 for x in d_iou)}


def _bootstrap_mean_ci(values: list[float], *, seed: int = 42, draws: int = 10000) -> list[float]:
    """Deterministic percentile bootstrap CI for the mean across folds.

    The interval is descriptive only: five folds are the resampling units and
    are not a substitute for an independent repeated experiment.
    """
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    means.sort()
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def _source_metrics(experiment_root: Path) -> dict[str, Any]:
    """Return per-source pooled metrics for A/B/D (C has no source CSV)."""
    output: dict[str, Any] = {}
    for group, (_, folder) in GROUPS.items():
        pooled: dict[str, dict[str, int]] = {}
        for fold in range(5):
            path = experiment_root / "5fold" / folder / f"fold{fold}" / "metrics" / "outer_test_per_source.csv"
            for row in csv.DictReader(path.open(encoding="utf-8")):
                source = str(row["source_group"])
                item = pooled.setdefault(source, {key: 0 for key in ("image_count", "tp", "fp", "fn")})
                for key in item:
                    item[key] += int(row[key])
        output[group] = []
        for source in sorted(pooled):
            item = pooled[source]
            output[group].append({
                "source_group": source,
                **item,
                "precision": _ratio(item["tp"], item["tp"] + item["fp"]),
                "recall": _ratio(item["tp"], item["tp"] + item["fn"]),
                "f1": _ratio(2 * item["tp"], 2 * item["tp"] + item["fp"] + item["fn"]),
                "iou": _ratio(item["tp"], item["tp"] + item["fp"] + item["fn"]),
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reevaluate", action="store_true", help="recompute A/B/D outer-test per-image and per-source metrics")
    parser.add_argument("--group", choices=tuple(GROUPS), help="only re-evaluate this group; use separate processes for D to avoid module-name collisions")
    parser.add_argument("--experiment-id", default=EXPERIMENT)
    args = parser.parse_args()
    experiment_root = PROJECT_ROOT / "runs" / args.experiment_id
    if args.reevaluate:
        if not torch.cuda.is_available():
            raise RuntimeError("--reevaluate requires CUDA")
        device = torch.device("cuda")
        selected_groups = {args.group: GROUPS[args.group]} if args.group else GROUPS
        for group, pair in selected_groups.items():
            _reevaluate(group, pair[0], experiment_root, device)
    summaries = [_read_group_metrics(experiment_root, group, folder) for group, (_, folder) in GROUPS.items()]
    c_summary = _read_group_metrics(WORKSPACE_ROOT / "sam2_adapter" / "runs" / "2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42", "C", "foreground")
    summaries.append(c_summary)
    by_group = {row["group"]: row for row in summaries}
    for row in summaries:
        row["bootstrap_mean_f1_ci95"] = _bootstrap_mean_ci(row["fold_f1"])
        row["bootstrap_mean_iou_ci95"] = _bootstrap_mean_ci(row["fold_iou"], seed=43)
    source_metrics = _source_metrics(experiment_root)
    info = experiment_root / "info"
    write_json(info / "comparison_summary.json", {"groups": summaries, "paired_A_B": _paired(by_group["A"], by_group["B"]), "paired_C_D": _paired(by_group["C"], by_group["D"]), "source_metrics": source_metrics, "notes": ["A/B compare frozen backbones under common prompt-free probe.", "C/D are complete-system comparisons with different native input resolutions and adapter/decoder scope.", "Bootstrap intervals resample the five outer folds with seed 42 (descriptive, not an independent replication).", "C is a historical run and has no reconstructed outer-test per-image/source CSV; source-level comparison is therefore reported for A/B/D only."]})
    print(json.dumps({row["group"]: {key: row[key] for key in ("mean_f1", "std_f1", "mean_iou", "std_iou", "worst_f1", "f1_range", "pooled_f1", "pooled_iou")} for row in summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
