from __future__ import annotations

import json
from pathlib import Path

import pytest

import dual_adapter_sam3.evaluate_cross_validation as evaluation
from dual_adapter_sam3.evaluate_cross_validation import (
    build_cross_validation_summary,
    build_parser,
    lock_selected_checkpoints,
)


def test_evaluation_parser_defaults_to_legacy_and_accepts_visual() -> None:
    assert build_parser().parse_args([]).model_variant == "da_sam3"
    assert (
        build_parser().parse_args(["--model-variant", "visual_da_sam3"]).model_variant
        == "visual_da_sam3"
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model-variant", "unknown"])


def _selected_record(path: Path, model_variant: str) -> dict[str, object]:
    stage = "stage1" if "stage1" in path.name else "stage2"
    return {
        "path": str(path.resolve()),
        "sha256": f"sha-{path.parent.parent.parent.name}-{stage}",
        "stage": stage,
        "epoch": 1 if stage == "stage1" else 2,
        "validation_segmentation_loss": 0.2,
        "prompt_contract_sha256": "prompt",
        "split_contract_sha256": "split",
        "model_variant": model_variant,
    }


def test_checkpoint_lock_reads_only_requested_variant_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_paths: list[Path] = []
    for fold in range(5):
        checkpoint_dir = tmp_path / "5fold" / "visual_da_sam3" / f"fold{fold}" / "artifacts" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "stage1_best.pt").touch()
        (checkpoint_dir / "stage2_best.pt").touch()

    def fake_record(path: Path) -> dict[str, object]:
        requested_paths.append(path)
        return _selected_record(path, "visual_da_sam3")

    monkeypatch.setattr(evaluation, "_checkpoint_record", fake_record)
    payload = lock_selected_checkpoints(
        tmp_path,
        "split",
        "prompt",
        model_variant="visual_da_sam3",
    )

    assert payload["model_variant"] == "visual_da_sam3"
    assert len(requested_paths) == 10
    assert all("/5fold/visual_da_sam3/" in path.as_posix() for path in requested_paths)
    assert all("/5fold/da_sam3/" not in path.as_posix() for path in requested_paths)


def test_checkpoint_lock_reports_precise_missing_variant_fold(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"visual_da_sam3.*fold0.*stage1_best\.pt"):
        lock_selected_checkpoints(
            tmp_path,
            "split",
            "prompt",
            model_variant="visual_da_sam3",
        )


def _fold_results() -> list[dict[str, object]]:
    metrics = {
        "macro": {name: 0.5 for name in ("precision", "recall", "f1", "iou", "accuracy")},
        "per_class": {
            concept: {name: 0.5 for name in ("precision", "recall", "f1", "iou", "accuracy")}
            for concept in ("crack_craquelure", "loss")
        },
    }
    return [
        {"fold": fold, "model_variant": "visual_da_sam3", "stage1": metrics, "stage2": metrics}
        for fold in range(5)
    ]


def test_cross_validation_summary_uses_variant_specific_links(tmp_path: Path) -> None:
    summary = build_cross_validation_summary(
        tmp_path,
        _fold_results(),
        model_variant="visual_da_sam3",
    )
    html = (tmp_path / "reports" / "cross_validation.html").read_text()
    payload = json.loads((tmp_path / "metrics" / "cross_validation_summary.json").read_text())

    assert summary["model_variant"] == payload["model_variant"] == "visual_da_sam3"
    assert "../5fold/visual_da_sam3/fold0/reports/index.html" in html
    assert "../5fold/da_sam3/" not in html
