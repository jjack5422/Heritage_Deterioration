from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]


def test_unet_and_segformer_have_independent_project_folders() -> None:
    assert not (WORKSPACE / "crack_detection_unet").exists()
    assert (WORKSPACE / "unet" / "src" / "train.py").is_file()
    assert (WORKSPACE / "unet" / "src" / "unet_model.py").is_file()
    assert (WORKSPACE / "segformer" / "src" / "train.py").is_file()
    assert (WORKSPACE / "segformer" / "src" / "segformer_model.py").is_file()


def test_project_source_trees_do_not_contain_the_other_model_family() -> None:
    unet_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (WORKSPACE / "unet" / "src").glob("*.py")
    ).lower()
    segformer_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (WORKSPACE / "segformer" / "src").glob("*.py")
    ).lower()

    assert "segformer" not in unet_source
    assert "resnet" not in segformer_source
    assert "convnext" not in segformer_source


def test_each_project_owns_only_its_model_runs() -> None:
    unet_runs = {path.name.lower() for path in (WORKSPACE / "unet" / "runs").iterdir()}
    segformer_runs = {
        path.name.lower() for path in (WORKSPACE / "segformer" / "runs").iterdir()
    }

    assert unet_runs
    assert segformer_runs
    assert all("segformer" not in name for name in unet_runs)
    assert all("segformer" in name for name in segformer_runs)
