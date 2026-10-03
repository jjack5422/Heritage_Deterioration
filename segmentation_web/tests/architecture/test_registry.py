from pathlib import Path

import pytest

from registry import (
    InvalidWeightError,
    RegistryError,
    UnknownModelError,
    get_deterioration_classes,
    get_models,
    get_weight_path,
    get_weights,
    resolve_deterioration_class,
)


def test_get_models_lists_six_experts_and_retains_existing_models() -> None:
    models = get_models()

    ids = {model["id"] for model in models}
    assert ids == {
        "dummy", "da_sam3", "resunet50", "convnext_unet",
        "sam3_adapter_craquelure", "sam3_adapter_loss", "sam3_adapter_crack",
        "sam2_adapter_craquelure", "sam2_adapter_loss", "sam2_adapter_crack",
    }
    assert {item["id"] for item in get_deterioration_classes("resunet50")} == {"craquelure"}
    assert {item["id"] for item in get_deterioration_classes("convnext_unet")} == {"craquelure"}


@pytest.mark.parametrize("architecture", ["sam2", "sam3"])
@pytest.mark.parametrize("category", ["craquelure", "loss", "crack"])
def test_experts_infer_their_only_class_and_reject_other_classes(architecture, category):
    model_id = f"{architecture}_adapter_{category}"
    assert resolve_deterioration_class(model_id, None) == category
    assert resolve_deterioration_class(model_id, category) == category
    with pytest.raises(RegistryError, match="Invalid"):
        resolve_deterioration_class(model_id, "crack_craquelure")


def test_da_sam3_exposes_and_validates_fixed_deterioration_classes() -> None:
    assert get_deterioration_classes("da_sam3") == (
        {"id": "crack_craquelure", "label": "裂縫／龜裂"},
        {"id": "loss", "label": "缺失"},
    )
    assert resolve_deterioration_class("da_sam3", "loss") == "loss"
    with pytest.raises(RegistryError, match="required"):
        resolve_deterioration_class("da_sam3", None)
    with pytest.raises(RegistryError, match="Invalid"):
        resolve_deterioration_class("da_sam3", "other")
    with pytest.raises(RegistryError, match="not supported"):
        resolve_deterioration_class("dummy", "loss")


def test_get_weights_filters_and_sorts_compatible_checkpoints(tmp_path: Path) -> None:
    model_dir = tmp_path / "sam2_adapter_craquelure"
    model_dir.mkdir()
    for name in ("epoch_100.pth", "best.pt", "model.ckpt", "notes.txt"):
        (model_dir / name).write_bytes(b"checkpoint")
    (model_dir / "nested.pth").mkdir()

    assert get_weights("sam2_adapter_craquelure", model_root=tmp_path) == [
        "best.pt",
        "epoch_100.pth",
        "model.ckpt",
    ]


def test_model_specific_weight_root_can_use_completed_run_directory(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "fold0" / "artifacts" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "best.pt").write_bytes(b"checkpoint")

    assert get_weights(
        "sam3_adapter_craquelure",
        weight_roots={"sam3_adapter_craquelure": checkpoint_root},
    ) == ["best.pt"]
    assert get_weight_path(
        "sam3_adapter_craquelure",
        "best.pt",
        weight_roots={"sam3_adapter_craquelure": checkpoint_root},
    ) == (checkpoint_root / "best.pt").resolve()


def test_da_sam3_prefers_stage2_best_checkpoint(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "da_sam3"
    checkpoint_root.mkdir()
    for name in ("stage1_best.pt", "stage2_last.pt", "stage2_best.pt"):
        (checkpoint_root / name).write_bytes(b"checkpoint")

    assert get_weights("da_sam3", model_root=tmp_path) == [
        "stage2_best.pt",
        "stage1_best.pt",
        "stage2_last.pt",
    ]


def test_dummy_uses_virtual_built_in_weight(tmp_path: Path) -> None:
    assert get_weights("dummy", model_root=tmp_path) == ["built-in"]
    assert get_weight_path("dummy", "built-in", model_root=tmp_path) is None


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnknownModelError, match="Unknown model"):
        get_weights("unknown", model_root=tmp_path)


@pytest.mark.parametrize(
    "weight_name",
    ["../../secret.pth", "/home/user/secret.pth", "nested/model.pth", "model.txt"],
)
def test_get_weight_path_blocks_unsafe_or_incompatible_names(
    tmp_path: Path, weight_name: str
) -> None:
    model_dir = tmp_path / "sam2_adapter_craquelure"
    model_dir.mkdir()

    with pytest.raises(InvalidWeightError):
        get_weight_path("sam2_adapter_craquelure", weight_name, model_root=tmp_path)


def test_get_weight_path_returns_existing_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "resunet50"
    model_dir.mkdir()
    checkpoint = model_dir / "best.pth"
    checkpoint.write_bytes(b"checkpoint")

    assert (
        get_weight_path("resunet50", "best.pth", model_root=tmp_path)
        == checkpoint.resolve()
    )


def test_checkpoint_symlink_cannot_escape_model_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "sam2_adapter_craquelure"
    model_dir.mkdir()
    outside = tmp_path / "secret.pth"
    outside.write_bytes(b"secret")
    (model_dir / "linked.pth").symlink_to(outside)

    assert get_weights("sam2_adapter_craquelure", model_root=tmp_path) == []
    with pytest.raises(InvalidWeightError, match="escapes"):
        get_weight_path("sam2_adapter_craquelure", "linked.pth", model_root=tmp_path)
