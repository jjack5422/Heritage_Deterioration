"""Shared pytest configuration for the web application tests."""

from __future__ import annotations

import sys
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = WEB_ROOT.parent
for path in (WEB_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
