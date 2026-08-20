from __future__ import annotations

import inspect
from pathlib import Path


def test_sac_and_adapter_are_independent_project_packages() -> None:
    import sam2_adapter.train_adapter as adapter_training
    import sam2_sac.train_h0 as sac_training

    workspace = Path(__file__).resolve().parents[1]
    assert Path(adapter_training.__file__).resolve().parent == workspace / "sam2_adapter"
    assert Path(sac_training.__file__).resolve().parent == workspace / "sam2_sac"
    assert adapter_training.PROJECT_ROOT == workspace / "sam2_adapter"
    assert sac_training.PROJECT_ROOT == workspace / "sam2_sac"
    assert "sam2_sac" not in inspect.getsource(adapter_training)


def test_each_project_owns_only_its_experiment_runs() -> None:
    workspace = Path(__file__).resolve().parents[1]
    sac_runs = {path.name for path in (workspace / "sam2_sac" / "runs").iterdir()}
    adapter_runs = {path.name for path in (workspace / "sam2_adapter" / "runs").iterdir()}

    assert sac_runs
    assert adapter_runs
    assert all("adapter" not in name.lower() for name in sac_runs)
    assert all("adapter" in name.lower() for name in adapter_runs)
