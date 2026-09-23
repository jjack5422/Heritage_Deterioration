from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ACTIVE_ENVIRONMENT_REFERENCES = (
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "model_projects" / "unet" / "README.md",
    WORKSPACE / "model_projects" / "unet" / "src" / "threshold_preview.py",
    WORKSPACE / "model_projects" / "unet" / "reporting" / "backfill.py",
    WORKSPACE / "model_projects" / "segformer" / "README.md",
    WORKSPACE / "model_projects" / "segformer" / "src" / "train.py",
    WORKSPACE / "model_projects" / "_shared" / "crackseg_common" / "reporting" / "outputs.py",
    WORKSPACE / "model_projects" / "sam2_sac" / "README.md",
    WORKSPACE / "model_projects" / "sam2_sac" / "train_h0.py",
    WORKSPACE / "model_projects" / "sam2_adapter" / "README.md",
)


def test_active_commands_use_architecture_neutral_environment_name() -> None:
    references = "\n".join(
        path.read_text(encoding="utf-8") for path in ACTIVE_ENVIRONMENT_REFERENCES
    )

    assert "sam2_env" not in references
    assert "crackseg_env" in references
