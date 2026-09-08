from __future__ import annotations

import sys
from types import ModuleType

from adapters.sam3_runtime import (
    OFFICIAL_SAM3_ROOT,
    VENDOR_MODELS_ROOT,
    VENDOR_RUNTIME_ROOT,
    activate_official_runtime,
    activate_vendor_runtime,
)


def _install_conflicting_modules(monkeypatch) -> None:
    for name in ("sam3", "sam3.model", "models", "models.encoder"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))


def test_official_runtime_purges_conflicts_and_wins_import_order(
    monkeypatch,
) -> None:
    _install_conflicting_modules(monkeypatch)
    monkeypatch.setattr(
        sys,
        "path",
        [
            "baseline",
            str(VENDOR_RUNTIME_ROOT),
            str(OFFICIAL_SAM3_ROOT),
            str(VENDOR_MODELS_ROOT),
        ],
    )

    activate_official_runtime()

    assert not any(
        name == "sam3" or name.startswith("sam3.") for name in sys.modules
    )
    assert not any(
        name == "models" or name.startswith("models.") for name in sys.modules
    )
    assert sys.path[:2] == [str(OFFICIAL_SAM3_ROOT), "baseline"]


def test_vendor_runtime_purges_conflicts_and_wins_import_order(
    monkeypatch,
) -> None:
    _install_conflicting_modules(monkeypatch)
    monkeypatch.setattr(
        sys,
        "path",
        [
            "baseline",
            str(OFFICIAL_SAM3_ROOT),
            str(VENDOR_MODELS_ROOT),
            str(VENDOR_RUNTIME_ROOT),
        ],
    )

    activate_vendor_runtime()

    assert not any(
        name == "sam3" or name.startswith("sam3.") for name in sys.modules
    )
    assert not any(
        name == "models" or name.startswith("models.") for name in sys.modules
    )
    assert sys.path[:3] == [
        str(VENDOR_RUNTIME_ROOT),
        str(VENDOR_MODELS_ROOT),
        "baseline",
    ]
