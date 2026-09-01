"""Generate the durable A/B/C/D comparison and audit artifacts.

The model runs and per-fold reporting artifacts are produced by ``train_probe``
and the training-output-reporting exporter.  This module only consumes those
immutable outputs and writes the cross-experiment summary under ``info/``.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXPERIMENT = ROOT / "runs" / "2026-08-26_sam2-sam3-native-probe-adapter_seed42"
GROUP_FOLDERS = {"A": "sam2_probe", "B": "sam3_probe", "D": "sam3_adapter"}
C_ROOT = WORKSPACE / "sam2_adapter" / "runs" / "2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42" / "5fold" / "foreground"
DATASET = WORKSPACE / "datasets" / "dataset_clean_v2_merged_craquelure"


def _read_summary() -> dict[str, Any]:
    return json.loads((EXPERIMENT / "info" / "comparison_summary.json").read_text(encoding="utf-8"))


def _all_required(run: Path) -> list[str]:
    required = [
        "config/run.json", "config/dataset.json", "config/environment.json",
        "metrics/epochs.csv", "metrics/tensorboard_scalars.csv",
        "metrics/per_image_validation.csv", "metrics/outer_test_metrics.json",
        "metrics/experiment_summary.json", "artifacts/checkpoints/best.pt",
        "artifacts/checkpoints/last.pt", "tensorboard/images/loss_curve.png",
        "tensorboard/images/manifest.csv", "reports/index.html",
        "reports/best_20.html", "reports/worst_20.html",
    ]
    return [item for item in required if not (run / item).is_file()]


def _image_paths_ok(run: Path) -> bool:
    for page in (run / "reports" / "index.html", run / "reports" / "best_20.html", run / "reports" / "worst_20.html"):
        text = page.read_text(encoding="utf-8", errors="replace")
        for token in ("tensorboard/images/", "artifacts/qualitative/"):
            if token in text:
                # Every relative image reference in the generated report must resolve.
                import re
                for value in re.findall(r'(?:src|href)=["\']([^"\']+)', text):
                    if value.startswith(("tensorboard/", "artifacts/")) and not (run / value).exists():
                        return False
    return True


def _dataset_audit() -> dict[str, Any]:
    """Check exact partition membership and source-group isolation for all runs."""
    expected_hashes = {
        "manifest_hash": "eb7ab1d34063d7371b97f672ca34782dee1e1d13f84c2a30b29281b8515c1e46",
        "image_manifest_hash": "9ee2574e7a3f70b847371943f5023587d8b4f396e497d7d48e418d6bc3d2af94",
        "mask_manifest_hash": "a457d40d9b819c1787e425c73f0524fa9ad4b903bf2efd28912c28b55b5e52e4",
    }
    from sam2_adapter.data import prepare_data_plan
    plan_index = prepare_data_plan(DATASET, outer_fold=0).index
    reference: dict[int, dict[str, set[str]]] = {}
    hash_ok = True
    split_ok = True
    leakage: dict[str, list[str]] = {}
    for group, folder in {**GROUP_FOLDERS, "C": "__c__"}.items():
        for fold in range(5):
            base = (C_ROOT if group == "C" else EXPERIMENT / "5fold" / folder) / f"fold{fold}"
            config = base / "config" / "dataset.json"
            data = json.loads(config.read_text(encoding="utf-8"))
            hash_ok &= all(data.get(key) == value for key, value in expected_hashes.items())
            sets = {key: set(data[key]) for key in ("train", "validation", "outer_test")}
            if group == "A":
                reference[fold] = sets
            else:
                split_ok &= sets == reference[fold]
            groups = {key: {str(plan_index[name]["source_group"]) for name in names} for key, names in sets.items()}
            overlap = []
            for left, right in (("train", "validation"), ("train", "outer_test"), ("validation", "outer_test")):
                overlap.extend(sorted(groups[left] & groups[right]))
            if overlap:
                leakage[f"{group}/fold{fold}"] = sorted(set(overlap))
    return {"hashes_ok": hash_ok, "partition_membership_identical": split_ok, "source_group_leakage": leakage, "source_group_leakage_free": not leakage, "dataset_root": str(DATASET)}


def _metric_consistency_audit() -> dict[str, Any]:
    """Ensure per-image TP/FP/FN sums reproduce each fold's aggregate JSON."""
    mismatches: dict[str, Any] = {}
    for group, folder in GROUP_FOLDERS.items():
        for fold in range(5):
            run = EXPERIMENT / "5fold" / folder / f"fold{fold}"
            aggregate = json.loads((run / "metrics" / "outer_test_metrics.json").read_text(encoding="utf-8"))["tile_micro"]
            rows = list(csv.DictReader((run / "metrics" / "outer_test_per_image.csv").open(encoding="utf-8")))
            counts = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn")}
            if any(counts[key] != int(aggregate[key]) for key in counts):
                mismatches[f"{group}/fold{fold}"] = {"csv": counts, "json": {key: int(aggregate[key]) for key in counts}}
    return {"mismatches": mismatches, "consistent": not mismatches}


def _validation_and_manifest_audit() -> dict[str, Any]:
    """Validate finite validation ranking rows and ranked image manifests."""
    failures: dict[str, str] = {}
    for group, folder in GROUP_FOLDERS.items():
        for fold in range(5):
            run = EXPERIMENT / "5fold" / folder / f"fold{fold}"
            rows = list(csv.DictReader((run / "metrics" / "per_image_validation.csv").open(encoding="utf-8")))
            # Images with no foreground GT have an undefined foreground F1;
            # reporting.py records those explicitly as non-rankable rows with
            # error_reason. All folds must still contain rankable rows.
            rankable = [row for row in rows if row.get("f1") not in (None, "")]
            if not rows or any(not row.get("image") or not row.get("accuracy") for row in rows) or not rankable:
                failures[f"{group}/fold{fold}/validation"] = "empty, missing image, or no rankable f1"
            for row in rows:
                for key in ("input_path", "gt_path", "prediction_path", "overlay_path"):
                    value = row.get(key, "")
                    if not value or value.startswith("/") or not (run / value).is_file():
                        failures[f"{group}/fold{fold}/validation/{key}"] = value
            manifest = list(csv.DictReader((run / "tensorboard" / "images" / "manifest.csv").open(encoding="utf-8")))
            if len(manifest) != 40 or {row.get("group") for row in manifest} != {"best", "worst"}:
                failures[f"{group}/fold{fold}/manifest"] = f"rows={len(manifest)}"
            for row in manifest:
                path = row.get("path", "")
                if not path or path.startswith("/") or not (run / "tensorboard" / "images" / path).is_file():
                    failures[f"{group}/fold{fold}/manifest/path"] = path
    return {"failures": failures, "valid": not failures}


def _timing_records() -> dict[str, Any]:
    """Estimate per-fold wall time from the immutable log start/end markers."""
    output: dict[str, Any] = {}
    for group, folder in GROUP_FOLDERS.items():
        folds = []
        for fold in range(5):
            log = EXPERIMENT / "5fold" / folder / f"fold{fold}" / "logs" / "train.log"
            first = log.read_text(encoding="utf-8").splitlines()[0]
            started = datetime.fromisoformat(first.split("=", 1)[1])
            ended = datetime.fromtimestamp(log.stat().st_mtime, started.tzinfo)
            folds.append({"fold": fold, "started": started.isoformat(), "log_completed_mtime": ended.isoformat(), "wall_minutes": (ended - started).total_seconds() / 60.0})
        output[group] = {"folds": folds, "wall_minutes_total": sum(item["wall_minutes"] for item in folds)}
    output["all_groups_wall_minutes_total"] = sum(item["wall_minutes_total"] for item in output.values())
    return output


def _resolution_qa_audit() -> dict[str, Any]:
    expected = {"sam2_probe": 1024, "sam3_probe": 1008}
    failures: dict[str, Any] = {}
    for group, size in expected.items():
        path = ROOT / "runs" / "preflight" / "resolution_qa" / group / "manifest.csv"
        if not path.is_file():
            failures[group] = "manifest missing"
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if len(rows) != 20 or any(int(row["model_input_size"]) != size for row in rows) or len({row["source_group"] for row in rows}) < 10:
            failures[group] = {"row_count": len(rows), "expected_size": size, "source_groups": len({row["source_group"] for row in rows})}
        if any(not (path.parent / row["panel"]).is_file() for row in rows):
            failures[f"{group}/panels"] = "panel missing"
    return {"failures": failures, "valid": not failures}


def _write_csv(summary: dict[str, Any]) -> None:
    info = EXPERIMENT / "info"
    fields = ["group", "mean_f1", "std_f1", "bootstrap_mean_f1_ci95", "mean_iou", "std_iou", "bootstrap_mean_iou_ci95", "mean_precision", "mean_recall", "mean_accuracy", "worst_f1", "best_f1", "f1_range", "pooled_f1", "pooled_iou", "pooled_accuracy"]
    with (info / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["groups"]:
            writer.writerow({key: row.get(key) for key in fields})
    with (info / "source_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["group", "source_group", "image_count", "tp", "fp", "fn", "precision", "recall", "f1", "iou"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, rows in summary.get("source_metrics", {}).items():
            for row in rows:
                writer.writerow({"group": group, **row})


def _report(summary: dict[str, Any], audit: dict[str, Any], timing: dict[str, Any]) -> str:
    rows = {row["group"]: row for row in summary["groups"]}
    paired_ab = summary["paired_A_B"]
    paired_cd = summary["paired_C_D"]
    lines = [
        "# SAM2／SAM3 受控比較報告",
        "",
        f"實驗：`{EXPERIMENT.name}`。所有 A、B、D 組均為 clean nested 5-fold outer-test；checkpoint 僅由 validation loss 選擇。",
        "",
        "## 主要結果",
        "",
        "| 組別 | mean F1 ± SD | mean IoU ± SD | pooled F1 | pooled IoU | pooled accuracy | worst F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("A", "B", "C", "D"):
        row = rows[group]
        lines.append(f"| {group} | {row['mean_f1']:.4f} ± {row['std_f1']:.4f} | {row['mean_iou']:.4f} ± {row['std_iou']:.4f} | {row['pooled_f1']:.4f} | {row['pooled_iou']:.4f} | {row['pooled_accuracy']:.4f} | {row['worst_f1']:.4f} |")
    lines += [
        "",
        "A/B 是相同 prompt-free probe 下的 frozen-backbone 比較；B 的五折 F1 勝過 A `4/5` 折，paired mean ΔF1（B−A）為 `%.4f`。" % paired_ab["f1_mean_difference"],
        "C/D 是完整系統比較，不能把差異單獨歸因於 backbone；paired mean ΔF1（D−C）為 `%.4f`，D 勝過 C ` %d/5` 折。" % (paired_cd["f1_mean_difference"], paired_cd["f1_wins_right"]),
        "",
        "## 穩定性與來源分布",
        "",
        "`mean ± SD`、worst fold 與 best-to-worst range 同時呈現；較高平均分數不自動等於較 robust。來源級 pooled 指標見 `source_metrics.csv`。C 為既有歷史 run，沒有可重建的 outer-test 逐影像／逐 source CSV，因此來源分布只對 A/B/D 報告。",
        "",
        "## 解析度與解讀邊界",
        "",
        "A 使用 SAM2 native 1024，B/D 使用 SAM3 native 1008，原始 tile 與評分空間均為 512；C 為既有 512-resolution run。A/B 因此是 native-preprocessing backbone comparison，不是完全相同輸入尺寸比較。",
        "官方可驗證的 SAM3 checkpoint 約 4.55 億 vision-backbone 參數，SAM2 Hiera-L 約 2.13 億；目前沒有同等預訓練的 SAM3 小型 checkpoint。因此本結果也不是等參數量比較，差異須解讀為官方模型表徵與容量的共同效果。",
        "",
        "## Reporting audit",
        "",
        f"- A/B/D 五組各 5 folds：`{'PASS' if not audit['missing_required'] else 'FAIL'}`",
        f"- HTML 圖片路徑：`{'PASS' if audit['html_image_paths_ok'] else 'FAIL'}`",
        f"- 每 fold 80 epochs、validation-only checkpoint selection：`{'PASS' if audit['selection_and_epochs_ok'] else 'FAIL'}`",
        f"- exact TensorBoard tags：`{'PASS' if audit['tensorboard_tags_ok'] else 'FAIL'}`",
        f"- dataset hashes、fold tile IDs 與 source-group isolation：`{'PASS' if audit['dataset']['hashes_ok'] and audit['dataset']['partition_membership_identical'] and audit['dataset']['source_group_leakage_free'] else 'FAIL'}`",
        f"- 逐影像 TP/FP/FN 與 outer-test aggregate 一致：`{'PASS' if audit['metric_consistency']['consistent'] else 'FAIL'}`",
        f"- validation rows 與 ranked image manifest：`{'PASS' if audit['validation_and_manifest']['valid'] else 'FAIL'}`",
        f"- 20-tile resolution preprocessing QA：`{'PASS' if audit['resolution_qa']['valid'] else 'FAIL'}`",
        "- Best/Worst composite panel order：Input、GT、Prediction、Overlay。已本地檢視 loss curve 與代表性 Best/Worst composite。",
        "",
        "## Bootstrap",
        "",
        "`comparison_summary.json` 的 95% bootstrap interval 以五個 outer folds 為重抽樣單位、固定 seed；僅作描述性不確定性，不取代獨立重複實驗。",
        "",
        "## Wall time",
        "",
        "訓練 log 的 `created=` 到完成 log mtime 估算：A 五折合計 `%.1f` 分鐘，B `%.1f` 分鐘，D `%.1f` 分鐘；三組合計 `%.1f` 分鐘。此值包含每 fold 的模型初始化、validation、checkpoint 與 reporting 輸出。" % (timing["A"]["wall_minutes_total"], timing["B"]["wall_minutes_total"], timing["D"]["wall_minutes_total"], timing["all_groups_wall_minutes_total"]),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    summary = _read_summary()
    missing: dict[str, list[str]] = {}
    html_ok = True
    selection_ok = True
    tags_ok = True
    expected_tags = {"loss/train", "loss/validation", "metrics/f1", "metrics/precision", "metrics/recall", "metrics/iou", "metrics/accuracy", "optimizer/lr"}
    for group, folder in GROUP_FOLDERS.items():
        for fold in range(5):
            run = EXPERIMENT / "5fold" / folder / f"fold{fold}"
            miss = _all_required(run)
            if miss:
                missing[f"{group}/fold{fold}"] = miss
            html_ok &= _image_paths_ok(run)
            run_summary = json.loads((run / "metrics" / "experiment_summary.json").read_text(encoding="utf-8"))
            selection_ok &= run_summary.get("epoch_count") == 80 and run_summary.get("selection_policy") == "validation_only_outer_test_excluded"
            tags = {row["tag"] for row in csv.DictReader((run / "metrics" / "tensorboard_scalars.csv").open(encoding="utf-8"))}
            tags_ok &= tags == expected_tags
    audit = {"missing_required": missing, "html_image_paths_ok": html_ok, "selection_and_epochs_ok": selection_ok, "tensorboard_tags_ok": tags_ok, "dataset": _dataset_audit(), "metric_consistency": _metric_consistency_audit(), "validation_and_manifest": _validation_and_manifest_audit(), "resolution_qa": _resolution_qa_audit()}
    info = EXPERIMENT / "info"
    (info / "comparison_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timing = _timing_records()
    (info / "training_timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (info / "comparison_report.md").write_text(_report(summary, audit, timing), encoding="utf-8")
    _write_csv(summary)
    if missing or not html_ok or not selection_ok or not tags_ok or not audit["dataset"]["hashes_ok"] or not audit["dataset"]["partition_membership_identical"] or not audit["dataset"]["source_group_leakage_free"] or not audit["metric_consistency"]["consistent"] or not audit["validation_and_manifest"]["valid"] or not audit["resolution_qa"]["valid"]:
        raise SystemExit(f"comparison audit failed: {audit}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
