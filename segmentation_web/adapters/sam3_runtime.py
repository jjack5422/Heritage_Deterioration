"""Isolate the incompatible official and vendor SAM3 Python runtimes."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from config import WORKSPACE_ROOT


OFFICIAL_SAM3_ROOT = WORKSPACE_ROOT / "segment-anything-3"
VENDOR_RUNTIME_ROOT = WORKSPACE_ROOT / "sam3_adapter/vendor_upstream_runtime"
VENDOR_MODELS_ROOT = VENDOR_RUNTIME_ROOT / "models"
_RUNTIME_PATHS = (
    OFFICIAL_SAM3_ROOT,
    VENDOR_RUNTIME_ROOT,
    VENDOR_MODELS_ROOT,
)
_CONFLICTING_MODULE_FAMILIES = ("sam3", "models")


def _purge_conflicting_modules() -> None:
    for name in tuple(sys.modules):
        if any(
            name == family or name.startswith(f"{family}.")
            for family in _CONFLICTING_MODULE_FAMILIES
        ):
            sys.modules.pop(name, None)


def _activate(paths: tuple[Path, ...]) -> None:
    _purge_conflicting_modules()
    known_paths = {str(path) for path in _RUNTIME_PATHS}
    sys.path[:] = [path for path in sys.path if path not in known_paths]
    sys.path[:0] = [str(path) for path in paths]
    importlib.invalidate_caches()


def activate_official_runtime() -> None:
    """Make the pristine Meta SAM3 package win the next import."""

    _activate((OFFICIAL_SAM3_ROOT,))


def activate_vendor_runtime() -> None:
    """Make the SAM3-Adapter author's modified runtime win the next import."""

    _activate((VENDOR_RUNTIME_ROOT, VENDOR_MODELS_ROOT))
