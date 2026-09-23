from __future__ import annotations

import inspect
from pathlib import Path


def test_sac_and_adapter_are_independent_project_packages() -> None:
    import model_projects.sam2_adapter.train_adapter as adapter_training
    import model_projects.sam2_sac.train_h0 as sac_training

    workspace = Path(__file__).resolve().parents[2]
    assert Path(adapter_training.__file__).resolve().parent == workspace / "model_projects" / "sam2_adapter"
    assert Path(sac_training.__file__).resolve().parent == workspace / "model_projects" / "sam2_sac"
    assert adapter_training.PROJECT_ROOT == workspace / "model_projects" / "sam2_adapter"
    assert sac_training.PROJECT_ROOT == workspace / "model_projects" / "sam2_sac"
    assert "sam2_sac" not in inspect.getsource(adapter_training)


