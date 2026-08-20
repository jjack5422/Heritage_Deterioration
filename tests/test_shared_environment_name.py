from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
ACTIVE_ENVIRONMENT_REFERENCES = (
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "unet" / "README.md",
    WORKSPACE / "unet" / "src" / "threshold_preview.py",
    WORKSPACE / "unet" / "reporting" / "backfill.py",
    WORKSPACE / "segformer" / "README.md",
    WORKSPACE / "segformer" / "src" / "train.py",
    WORKSPACE / "_lib" / "crackseg_common" / "reporting" / "outputs.py",
    WORKSPACE / "sam2_sac" / "README.md",
    WORKSPACE / "sam2_sac" / "train_h0.py",
    WORKSPACE / "sam2_adapter" / "README.md",
)


def test_active_commands_use_architecture_neutral_environment_name() -> None:
    references = "\n".join(
        path.read_text(encoding="utf-8") for path in ACTIVE_ENVIRONMENT_REFERENCES
    )

    assert "sam2_env" not in references
    assert "crackseg_env" in references
