from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
MODEL_PROJECTS = WORKSPACE / "model_projects"


def test_unet_and_segformer_have_independent_project_folders() -> None:
    assert not (WORKSPACE / "crack_detection_unet").exists()
    assert (MODEL_PROJECTS / "unet" / "src" / "train.py").is_file()
    assert (MODEL_PROJECTS / "unet" / "src" / "unet_model.py").is_file()
    assert (MODEL_PROJECTS / "segformer" / "src" / "train.py").is_file()
    assert (MODEL_PROJECTS / "segformer" / "src" / "segformer_model.py").is_file()


def test_project_source_trees_do_not_contain_the_other_model_family() -> None:
    unet_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (MODEL_PROJECTS / "unet" / "src").glob("*.py")
    ).lower()
    segformer_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (MODEL_PROJECTS / "segformer" / "src").glob("*.py")
    ).lower()

    assert "segformer" not in unet_source
    assert "resnet" not in segformer_source
    assert "convnext" not in segformer_source

