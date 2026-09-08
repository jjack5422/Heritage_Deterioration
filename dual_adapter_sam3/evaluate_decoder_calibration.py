"""Evaluate locked DA-SAM3 decoder-calibration checkpoints against their sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_adaptation_checkpoint
from .concepts import CHANNEL_ORDER, load_concept_registry
from .data import MonumentDeteriorationDataset
from .decoder_calibration import (
    CALIBRATION_CHECKPOINT_TYPE,
    CALIBRATION_EXPERT,
    load_decoder_calibration_checkpoint,
)
from .evaluate_cross_validation import (
    _run_report_builder,
    _write_outer_rows,
    evaluate_stage,
    refresh_validation_artifacts,
)
from .model import build_dual_adapter_model
from .reporting import RunLayout, write_json
from .sam3_integration import DEFAULT_CHECKPOINT, file_sha256
from .splits import fold_membership, load_split_contract, source_group_index
from .train import DEFAULT_DATASET, PROJECT_ROOT, _loader
from .train_decoder_calibration import (
    DEFAULT_EXPERIMENT,
    DEFAULT_SOURCE_EXPERIMENT,
    calibration_run_root,
    source_checkpoint_path,
)


STAGES = ("source_stage2", "decoder_calibration")
METRICS = ("precision", "recall", "f1", "iou", "accuracy")


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _pixel_area_ratios(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for concept in CHANNEL_ORDER:
        selected = [row for row in rows if row["target_class"] == concept]
        gt_pixels = sum(int(row["gt_pixels"]) for row in selected)
        pred_pixels = sum(int(row["pred_pixels"]) for row in selected)
        result[concept] = pred_pixels / gt_pixels if gt_pixels else 0.0
    result["macro"] = statistics.fmean(result.values())
    return result


def lock_selected_checkpoints(
    experiment_root: Path,
    source_experiment_root: Path,
    *,
    prompt_hash: str,
    split_hash: str,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in range(5):
        source_path = source_checkpoint_path(source_experiment_root, fold)
        calibration_path = calibration_run_root(experiment_root, fold) / "artifacts" / "checkpoints" / "best.pt"
        if not source_path.is_file():
            raise FileNotFoundError(f"source Stage2 checkpoint is missing: {source_path}")
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"decoder-calibration fold{fold} selected checkpoint is missing: {calibration_path}"
            )
        source_sha256 = file_sha256(source_path)
        checkpoint = torch.load(calibration_path, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_type") != CALIBRATION_CHECKPOINT_TYPE:
            raise RuntimeError(f"fold{fold} selected checkpoint type mismatch")
        if checkpoint.get("stage") != "decoder_calibration":
            raise RuntimeError(f"fold{fold} selected checkpoint stage mismatch")
        if checkpoint.get("prompt_contract_sha256") != prompt_hash:
            raise RuntimeError(f"fold{fold} selected checkpoint prompt contract mismatch")
        if checkpoint.get("split_contract_sha256") != split_hash:
            raise RuntimeError(f"fold{fold} selected checkpoint split contract mismatch")
        if checkpoint.get("source_checkpoint_sha256") != source_sha256:
            raise RuntimeError(f"fold{fold} selected checkpoint source mismatch")
        folds.append(
            {
                "fold": fold,
                "source": {
                    "path": str(source_path.resolve()),
                    "sha256": source_sha256,
                },
                "decoder_calibration": {
                    "path": str(calibration_path.resolve()),
                    "sha256": file_sha256(calibration_path),
                    "epoch": int(checkpoint["epoch"]),
                    "validation_segmentation_loss": float(
                        checkpoint["validation_segmentation_loss"]
                    ),
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "expert": CALIBRATION_EXPERT,
        "selection_boundary": "minimum validation segmentation loss; outer test not read during calibration training or checkpoint lock",
        "outer_test_previously_observed_for_architecture_design": True,
        "prompt_contract_sha256": prompt_hash,
        "split_contract_sha256": split_hash,
        "folds": folds,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(experiment_root / "info" / "selected_checkpoints.json", payload)
    return payload


def build_cross_validation_summary(
    experiment_root: Path,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if [int(row["fold"]) for row in fold_results] != list(range(5)):
        raise ValueError("decoder-calibration summary requires ordered folds 0..4")
    stages: dict[str, Any] = {}
    for stage in STAGES:
        stages[stage] = {
            "macro": {
                metric: _mean_std(
                    [float(row[stage]["macro"][metric]) for row in fold_results]
                )
                for metric in METRICS
            },
            "per_class": {
                concept: {
                    metric: _mean_std(
                        [
                            float(row[stage]["per_class"][concept][metric])
                            for row in fold_results
                        ]
                    )
                    for metric in METRICS
                }
                for concept in CHANNEL_ORDER
            },
            "predicted_to_gt_pixel_ratio": {
                target: _mean_std(
                    [float(row[stage]["predicted_to_gt_pixel_ratio"][target]) for row in fold_results]
                )
                for target in (*CHANNEL_ORDER, "macro")
            },
        }
    paired: dict[str, Any] = {}
    for target in (*CHANNEL_ORDER, "macro"):
        paired[target] = {}
        for metric in METRICS:
            values = [
                float(row["decoder_calibration"]["macro" if target == "macro" else "per_class"][metric] if target == "macro" else row["decoder_calibration"]["per_class"][target][metric])
                - float(row["source_stage2"]["macro" if target == "macro" else "per_class"][metric] if target == "macro" else row["source_stage2"]["per_class"][target][metric])
                for row in fold_results
            ]
            paired[target][metric] = {"per_fold": values, **_mean_std(values)}
        area_values = [
            float(row["decoder_calibration"]["predicted_to_gt_pixel_ratio"][target])
            - float(row["source_stage2"]["predicted_to_gt_pixel_ratio"][target])
            for row in fold_results
        ]
        paired[target]["predicted_to_gt_pixel_ratio"] = {
            "per_fold": area_values,
            **_mean_std(area_values),
        }
    summary = {
        "schema_version": 1,
        "expert": CALIBRATION_EXPERT,
        "fold_count": 5,
        "primary_metric": "decoder-calibration macro foreground F1, arithmetic mean across folds",
        "checkpoint_selection": "minimum validation segmentation loss",
        "threshold": 0.5,
        "outer_test_excluded_from_checkpoint_selection": True,
        "outer_test_previously_observed_for_architecture_design": True,
        "standard_deviation": "sample standard deviation (N-1)",
        "stages": stages,
        "decoder_calibration_minus_source_stage2": paired,
        "folds": fold_results,
    }
    metrics_dir = experiment_root / "metrics"
    reports_dir = experiment_root / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "cross_validation_summary.json", summary)
    with (metrics_dir / "cross_validation_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "fold",
            "stage",
            "target",
            *METRICS,
            "predicted_to_gt_pixel_ratio",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in fold_results:
            for stage in STAGES:
                for target in (*CHANNEL_ORDER, "macro"):
                    values = row[stage]["macro"] if target == "macro" else row[stage]["per_class"][target]
                    writer.writerow(
                        {
                            "fold": row["fold"],
                            "stage": stage,
                            "target": target,
                            **{metric: values[metric] for metric in METRICS},
                            "predicted_to_gt_pixel_ratio": row[stage][
                                "predicted_to_gt_pixel_ratio"
                            ][target],
                        }
                    )
    rows_html = "".join(
        "<tr>"
        f"<td>{row['fold']}</td>"
        f"<td>{row['source_stage2']['macro']['f1']:.4f}</td>"
        f"<td>{row['decoder_calibration']['macro']['f1']:.4f}</td>"
        f"<td>{row['decoder_calibration']['macro']['f1'] - row['source_stage2']['macro']['f1']:+.4f}</td>"
        f"<td>{row['source_stage2']['predicted_to_gt_pixel_ratio']['crack_craquelure']:.3f}</td>"
        f"<td>{row['decoder_calibration']['predicted_to_gt_pixel_ratio']['crack_craquelure']:.3f}</td>"
        f'<td><a href="../5fold/{CALIBRATION_EXPERT}/fold{row["fold"]}/reports/index.html">Fold report</a></td>'
        "</tr>"
        for row in fold_results
    )
    macro = stages["decoder_calibration"]["macro"]
    crack_delta = paired["crack_craquelure"]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DA-SAM3 decoder calibration</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>DA-SAM3 decoder calibration: five-fold diagnostic ablation</h1>
<p>Calibrated Macro-F1: <strong>{macro['f1']['mean']:.4f} ± {macro['f1']['std_sample']:.4f}</strong>.</p>
<p>Craquelure F1 delta: <strong>{crack_delta['f1']['mean']:+.4f}</strong>; predicted/GT pixel-ratio delta: <strong>{crack_delta['predicted_to_gt_pixel_ratio']['mean']:+.4f}</strong>.</p>
<p>Checkpoints were selected only by validation segmentation loss. This is a diagnostic ablation: baseline outer-test results had already been observed before the architecture was selected.</p>
<table><thead><tr><th>Fold</th><th>Source Macro-F1</th><th>Calibrated Macro-F1</th><th>Δ F1</th><th>Source crack area ratio</th><th>Calibrated crack area ratio</th><th>Artifacts</th></tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    (reports_dir / "cross_validation.html").write_text(html, encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official SAM3 evaluation")
    experiment_root = PROJECT_ROOT / "runs" / args.experiment_id
    source_experiment_root = PROJECT_ROOT / "runs" / args.source_experiment_id
    registry = load_concept_registry(args.concepts)
    split = load_split_contract(args.splits, dataset_root=args.dataset)
    locked = lock_selected_checkpoints(
        experiment_root,
        source_experiment_root,
        prompt_hash=registry.sha256,
        split_hash=split.sha256,
    )
    group_index = source_group_index(split)
    device = torch.device("cuda")
    fold_results: list[dict[str, Any]] = []
    for fold_record in locked["folds"]:
        fold = int(fold_record["fold"])
        membership = fold_membership(split, fold)
        layout = RunLayout.create(calibration_run_root(experiment_root, fold))
        outer_dataset = MonumentDeteriorationDataset(
            args.dataset, membership.outer_test, group_index, train_augmentation=False
        )
        validation_dataset = MonumentDeteriorationDataset(
            args.dataset, membership.validation, group_index, train_augmentation=False
        )
        outer_loader = _loader(
            outer_dataset, batch_size=args.batch_size, train=False, workers=args.workers
        )
        validation_loader = _loader(
            validation_dataset,
            batch_size=args.batch_size,
            train=False,
            workers=args.workers,
        )
        model = build_dual_adapter_model(
            registry,
            model_variant="da_sam3",
            checkpoint=args.checkpoint,
            device=device,
        ).to(device)

        source_path = Path(fold_record["source"]["path"])
        source_sha256 = str(fold_record["source"]["sha256"])
        load_adaptation_checkpoint(
            source_path,
            model,
            expected_prompt_hash=registry.sha256,
            expected_split_hash=split.sha256,
        )
        source_rows, source_metrics = evaluate_stage(
            model, outer_loader, device, fold_index=fold, stage="source_stage2"
        )
        source_metrics["predicted_to_gt_pixel_ratio"] = _pixel_area_ratios(source_rows)

        calibration_path = Path(fold_record["decoder_calibration"]["path"])
        selected = load_decoder_calibration_checkpoint(
            calibration_path,
            model,
            expected_prompt_hash=registry.sha256,
            expected_split_hash=split.sha256,
            expected_source_checkpoint_sha256=source_sha256,
        )
        calibrated_rows, calibrated_metrics = evaluate_stage(
            model, outer_loader, device, fold_index=fold, stage="decoder_calibration"
        )
        calibrated_metrics["predicted_to_gt_pixel_ratio"] = _pixel_area_ratios(
            calibrated_rows
        )
        _write_outer_rows(
            layout.metrics / "per_image_outer_test.csv",
            [*source_rows, *calibrated_rows],
        )
        write_json(
            layout.metrics / "outer_test_metrics.json",
            {
                "status": "complete",
                "expert": CALIBRATION_EXPERT,
                "selected_epoch": int(selected["epoch"]),
                "selected_checkpoint_sha256": fold_record["decoder_calibration"]["sha256"],
                "selection_source": "minimum validation segmentation loss",
                "outer_test_excluded_from_checkpoint_selection": True,
                "outer_test_previously_observed_for_architecture_design": True,
                "threshold": 0.5,
                "source_stage2": source_metrics,
                "decoder_calibration": calibrated_metrics,
            },
        )
        refresh_validation_artifacts(model, validation_loader, device, layout)
        _run_report_builder(layout)
        fold_results.append(
            {
                "fold": fold,
                "source_stage2": source_metrics,
                "decoder_calibration": calibrated_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "source_macro_f1": source_metrics["macro"]["f1"],
                    "calibrated_macro_f1": calibrated_metrics["macro"]["f1"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    build_cross_validation_summary(experiment_root, fold_results)
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(experiment_root / "reports" / "cross_validation.html"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return experiment_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source-experiment-id", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--concepts", type=Path, default=PROJECT_ROOT / "configs" / "concepts.yaml"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "configs" / "splits.json"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
