from __future__ import annotations

import json
from pathlib import Path

import pytest

from dual_adapter_sam3.train import (
    _variant_run_root,
    _write_or_validate_model_contract,
    build_parser,
)


def test_training_parser_defaults_to_legacy_variant() -> None:
    args = build_parser().parse_args(["--fold", "0"])
    assert args.model_variant == "da_sam3"


@pytest.mark.parametrize("model_variant", ["da_sam3", "visual_da_sam3"])
def test_training_parser_accepts_supported_variants(model_variant: str) -> None:
    args = build_parser().parse_args(["--fold", "2", "--model-variant", model_variant])
    assert args.model_variant == model_variant


def test_training_parser_rejects_unknown_variant() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--fold", "0", "--model-variant", "unknown"])


def test_variant_run_root_is_pure_and_variant_specific(tmp_path: Path) -> None:
    legacy = _variant_run_root(tmp_path, "da_sam3", 1)
    visual = _variant_run_root(tmp_path, "visual_da_sam3", 4)

    assert legacy == tmp_path / "5fold" / "da_sam3" / "fold1"
    assert visual == tmp_path / "5fold" / "visual_da_sam3" / "fold4"
    assert not (tmp_path / "5fold").exists()

    with pytest.raises(ValueError, match="model variant"):
        _variant_run_root(tmp_path, "unknown", 0)
    with pytest.raises(ValueError, match="fold"):
        _variant_run_root(tmp_path, "da_sam3", 5)


def test_model_contract_guard_writes_once_and_reuses_identical_contract(tmp_path: Path) -> None:
    path = tmp_path / "info" / "model_contract.json"
    contract = {"model_variant": "visual_da_sam3", "rank": 8}

    _write_or_validate_model_contract(path, contract)
    first_bytes = path.read_bytes()
    _write_or_validate_model_contract(path, dict(contract))

    assert path.read_bytes() == first_bytes
    assert json.loads(path.read_text()) == contract


def test_model_contract_guard_rejects_mismatch_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "info" / "model_contract.json"
    legacy = {"model_variant": "da_sam3", "rank": 8}
    _write_or_validate_model_contract(path, legacy)
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="existing model contract"):
        _write_or_validate_model_contract(
            path,
            {"model_variant": "visual_da_sam3", "rank": 8},
        )
    assert path.read_bytes() == before


def test_model_contract_guard_accepts_legacy_contract_missing_only_variant(tmp_path: Path) -> None:
    path = tmp_path / "info" / "model_contract.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rank": 8}) + "\n", encoding="utf-8")
    before = path.read_bytes()

    _write_or_validate_model_contract(
        path,
        {"model_variant": "da_sam3", "rank": 8},
    )
    assert path.read_bytes() == before

    with pytest.raises(RuntimeError, match="existing model contract"):
        _write_or_validate_model_contract(
            path,
            {"model_variant": "visual_da_sam3", "rank": 8},
        )
