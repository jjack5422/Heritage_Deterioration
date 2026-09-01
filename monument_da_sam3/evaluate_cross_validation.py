"""Locked-checkpoint outer testing and five-fold DA-SAM3 report generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sam2_adapter.reporting import binary_metric_row

from .concepts import CHANNEL_ORDER, load_concept_registry
from .data import MonumentDeteriorationDataset
from .metrics import MultilabelConfusion
from .model import MonumentDaSam3
from .reporting import RunLayout, save_concept_qualitative, write_json, write_validation_rows
from .sam3_integration import DEFAULT_CHECKPOINT, file_sha256
from .splits import fold_membership, load_split_contract, source_group_index
from .train import (
    DEFAULT_DATASET,
    DEFAULT_EXPERIMENT,
    PROJECT_ROOT,
    _load_adaptation,
    _loader,
    _to_device,
)


REPORT_BUILDER = Path.home() / ".codex" / "skills" / "training-output-reporting" / "scripts" / "build_training_report.py"
OUTER_COLUMNS = (
    "fold",
    "stage",
    "image",
    "source_group",
    "target_class",
    "f1",
    "precision",
    "recall",
    "iou",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
    "gt_pixels",
    "pred_pixels",
    "error_reason",
)


def _finite_binary_row(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    row = binary_metric_row(target, prediction)
    if row["f1"] == "":
        empty_score = 1.0 if int(row["pred_pixels"]) == 0 else 0.0
        row["f1"] = empty_score
        row["iou"] = empty_score
    row["tn"] = int(target.size) - int(row["tp"]) - int(row["fp"]) - int(row["fn"])
    return row


def _checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "stage": checkpoint["stage"],
        "epoch": int(checkpoint["epoch"]),
        "validation_segmentation_loss": float(checkpoint["validation_segmentation_loss"]),
        "prompt_contract_sha256": checkpoint["prompt_contract_sha256"],
        "split_contract_sha256": checkpoint["split_contract_sha256"],
    }


def lock_selected_checkpoints(experiment_root: Path, split_hash: str, prompt_hash: str) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold_index in range(5):
        layout = RunLayout.create(experiment_root / "5fold" / "da_sam3" / f"fold{fold_index}")
        stage1 = _checkpoint_record(layout.checkpoints / "stage1_best.pt")
        stage2 = _checkpoint_record(layout.checkpoints / "stage2_best.pt")
        if stage1["stage"] != "stage1" or stage2["stage"] != "stage2":
            raise RuntimeError(f"fold{fold_index} selected checkpoint stage mismatch")
        for record in (stage1, stage2):
            if record["prompt_contract_sha256"] != prompt_hash or record["split_contract_sha256"] != split_hash:
                raise RuntimeError(f"fold{fold_index} selected checkpoint contract mismatch")
        folds.append({"fold": fold_index, "stage1": stage1, "stage2": stage2})
    payload: dict[str, Any] = {
        "schema_version": 1,
        "selection_boundary": "validation_only; outer-test was not read before this lock",
        "prompt_contract_sha256": prompt_hash,
        "split_contract_sha256": split_hash,
        "folds": folds,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(experiment_root / "info" / "selected_checkpoints.json", payload)
    return payload


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for concept in CHANNEL_ORDER:
        selected = [row for row in rows if row["target_class"] == concept]
        totals = {key: sum(int(row[key]) for row in selected) for key in ("tp", "fp", "fn", "tn")}
        tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
        def ratio(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0
        per_class[concept] = {
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "iou": ratio(tp, tp + fp + fn),
            "accuracy": ratio(tp + tn, tp + fp + fn + tn),
            **totals,
        }
    macro = {
        metric: sum(float(per_class[concept][metric]) for concept in CHANNEL_ORDER) / len(CHANNEL_ORDER)
        for metric in ("precision", "recall", "f1", "iou", "accuracy")
    }
    return {"per_class": per_class, "macro": macro}


def _source_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_group"])].append(row)
    return {name: _metrics_from_rows(group_rows) for name, group_rows in sorted(groups.items())}


@torch.no_grad()
def evaluate_stage(
    model: MonumentDaSam3,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    fold_index: int,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
        predictions = output.logits.sigmoid() >= 0.5
        for sample_index, image_id in enumerate(batch["image_id"]):
            source_group = str(batch["source_group"][sample_index])
            for concept_index, concept in enumerate(CHANNEL_ORDER):
                valid = batch["valid_masks"][sample_index, concept_index].detach().cpu().numpy().astype(bool)
                target = batch["targets"][sample_index, concept_index].detach().cpu().numpy().astype(bool) & valid
                prediction = predictions[sample_index, concept_index].detach().cpu().numpy().astype(bool) & valid
                row = _finite_binary_row(target, prediction)
                row.update({
                    "fold": fold_index,
                    "stage": stage,
                    "image": str(image_id),
                    "source_group": source_group,
                    "target_class": concept,
                })
                rows.append(row)
    metrics = _metrics_from_rows(rows)
    metrics["per_source_group"] = _source_group_metrics(rows)
    metrics["image_concept_rows"] = len(rows)
    return rows, metrics


@torch.no_grad()
def refresh_validation_artifacts(
    model: MonumentDaSam3,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    layout: RunLayout,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch["image"])
        predictions = output.logits.sigmoid() >= 0.5
        for sample_index, image_id in enumerate(batch["image_id"]):
            rgb = (
                batch["image"][sample_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
            for concept_index, concept in enumerate(CHANNEL_ORDER):
                valid = batch["valid_masks"][sample_index, concept_index].detach().cpu().numpy().astype(bool)
                target = batch["targets"][sample_index, concept_index].detach().cpu().numpy().astype(bool) & valid
                prediction = predictions[sample_index, concept_index].detach().cpu().numpy().astype(bool) & valid
                rows.append(save_concept_qualitative(
                    layout,
                    concept=concept,
                    image_id=str(image_id),
                    input_rgb=rgb,
                    target=target,
                    prediction=prediction,
                ))
    return write_validation_rows(layout, rows)


def _write_outer_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_report_builder(layout: RunLayout) -> None:
    result = subprocess.run(
        [sys.executable, str(REPORT_BUILDER), "--run-dir", str(layout.root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"training-output-reporting builder failed: {result.stderr or result.stdout}")


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def build_cross_validation_summary(experiment_root: Path, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in ("stage1", "stage2"):
        stage_summary: dict[str, Any] = {"macro": {}, "per_class": {}}
        for metric in ("precision", "recall", "f1", "iou", "accuracy"):
            values = [float(result[stage]["macro"][metric]) for result in fold_results]
            stage_summary["macro"][metric] = _mean_std(values)
        for concept in CHANNEL_ORDER:
            stage_summary["per_class"][concept] = {
                metric: _mean_std([float(result[stage]["per_class"][concept][metric]) for result in fold_results])
                for metric in ("precision", "recall", "f1", "iou", "accuracy")
            }
        stages[stage] = stage_summary
    paired = {
        metric: {
            "per_fold": [
                float(result["stage2"]["macro"][metric]) - float(result["stage1"]["macro"][metric])
                for result in fold_results
            ]
        }
        for metric in ("precision", "recall", "f1", "iou", "accuracy")
    }
    for value in paired.values():
        value.update(_mean_std(value["per_fold"]))
    summary = {
        "schema_version": 1,
        "fold_count": 5,
        "primary_metric": "stage2 macro foreground F1, arithmetic mean across folds",
        "standard_deviation": "sample standard deviation (N-1)",
        "stages": stages,
        "stage2_minus_stage1": paired,
        "folds": fold_results,
        "outer_test_excluded_from_selection": True,
        "overlap_limitation": "class-index ground truth cannot validate overlapping deterioration labels",
    }
    metrics_dir = experiment_root / "metrics"
    reports_dir = experiment_root / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "cross_validation_summary.json", summary)
    csv_path = metrics_dir / "cross_validation_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("fold", "stage", "target", "precision", "recall", "f1", "iou", "accuracy")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in fold_results:
            for stage in ("stage1", "stage2"):
                for target in (*CHANNEL_ORDER, "macro"):
                    values = result[stage]["macro"] if target == "macro" else result[stage]["per_class"][target]
                    writer.writerow({"fold": result["fold"], "stage": stage, "target": target, **{key: values[key] for key in fields[3:]}})
    stage2_macro = stages["stage2"]["macro"]
    table_rows = "".join(
        "<tr>" + f"<td>{result['fold']}</td>" + "".join(
            f"<td>{float(result['stage2']['macro'][metric]):.4f}</td>"
            for metric in ("precision", "recall", "f1", "iou", "accuracy")
        ) + f'<td><a href="../5fold/da_sam3/fold{result["fold"]}/reports/index.html">Fold report</a></td></tr>'
        for result in fold_results
    )
    html = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>DA-SAM3 five-fold report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef;padding:2px 5px}}</style></head><body>
<h1>Monument DA-SAM3 512 five-fold outer-test report</h1>
<p>Primary Stage2 Macro-F1: <strong>{stage2_macro['f1']['mean']:.4f} ± {stage2_macro['f1']['std_sample']:.4f}</strong> (sample SD).</p>
<p>Outer-test was excluded from checkpoint, prompt, threshold, and hyperparameter selection. Threshold is fixed at 0.5.</p>
<table><thead><tr><th>Fold</th><th>Precision</th><th>Recall</th><th>Macro-F1</th><th>IoU</th><th>Accuracy</th><th>Artifacts</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Stage2 five-fold summary</h2><pre>{json.dumps(stage2_macro, ensure_ascii=False, indent=2)}</pre>
<h2>Stage2 − Stage1 paired differences</h2><pre>{json.dumps(paired, ensure_ascii=False, indent=2)}</pre>
<p>Limitation: the original class-index masks cannot validate overlapping crack/craquelure and loss at the same pixel.</p>
</body></html>"""
    (reports_dir / "cross_validation.html").write_text(html, encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official SAM3 evaluation")
    experiment_root = PROJECT_ROOT / "runs" / args.experiment_id
    registry = load_concept_registry(args.concepts)
    split = load_split_contract(args.splits, dataset_root=args.dataset)
    locked = lock_selected_checkpoints(experiment_root, split.sha256, registry.sha256)
    group_index = source_group_index(split)
    device = torch.device("cuda")
    fold_results: list[dict[str, Any]] = []
    for fold_record in locked["folds"]:
        fold_index = int(fold_record["fold"])
        membership = fold_membership(split, fold_index)
        layout = RunLayout.create(experiment_root / "5fold" / "da_sam3" / f"fold{fold_index}")
        outer_dataset = MonumentDeteriorationDataset(args.dataset, membership.outer_test, group_index, train_augmentation=False)
        validation_dataset = MonumentDeteriorationDataset(args.dataset, membership.validation, group_index, train_augmentation=False)
        outer_loader = _loader(outer_dataset, batch_size=args.batch_size, train=False, workers=args.workers)
        validation_loader = _loader(validation_dataset, batch_size=args.batch_size, train=False, workers=args.workers)
        model = MonumentDaSam3(registry, checkpoint=args.checkpoint, device=device).to(device)
        stage_rows: list[dict[str, Any]] = []
        stage_metrics: dict[str, Any] = {}
        for stage in ("stage1", "stage2"):
            checkpoint_path = Path(fold_record[stage]["path"])
            _load_adaptation(checkpoint_path, model, expected_prompt_hash=registry.sha256, expected_split_hash=split.sha256)
            rows, metrics = evaluate_stage(model, outer_loader, device, fold_index=fold_index, stage=stage)
            stage_rows.extend(rows)
            stage_metrics[stage] = metrics
        _write_outer_rows(layout.metrics / "per_image_outer_test.csv", stage_rows)
        selected_epoch = int(fold_record["stage2"]["epoch"])
        outer_payload = {
            "status": "complete",
            "selected_epoch": selected_epoch,
            "selected_checkpoint_sha256": fold_record["stage2"]["sha256"],
            "selection_source": "stage2 minimum validation segmentation loss",
            "outer_test_excluded_from_selection": True,
            "threshold": 0.5,
            "stage1": stage_metrics["stage1"],
            "stage2": stage_metrics["stage2"],
            "per_class": stage_metrics["stage2"]["per_class"],
            "macro": stage_metrics["stage2"]["macro"],
        }
        write_json(layout.metrics / "outer_test_metrics.json", outer_payload)
        _load_adaptation(Path(fold_record["stage2"]["path"]), model, expected_prompt_hash=registry.sha256, expected_split_hash=split.sha256)
        refresh_validation_artifacts(model, validation_loader, device, layout)
        _run_report_builder(layout)
        fold_results.append({"fold": fold_index, **stage_metrics})
        print(json.dumps({"fold": fold_index, "stage2_macro": stage_metrics["stage2"]["macro"]}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    build_cross_validation_summary(experiment_root, fold_results)
    print(json.dumps({"status": "complete", "report": str(experiment_root / "reports" / "cross_validation.html")}, ensure_ascii=False), flush=True)
    return experiment_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--concepts", type=Path, default=PROJECT_ROOT / "configs" / "concepts.yaml")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "configs" / "splits.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
