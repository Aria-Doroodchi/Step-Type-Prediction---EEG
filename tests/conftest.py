"""Pytest fixtures shared across smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `eeg_steptype` importable when running tests without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def smoke_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "smoke.yaml"
