"""Shared fixtures: everything runs on CPU on a tiny synthetic release (tests/synthetic.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synthetic import make_release  # noqa: E402


@pytest.fixture(scope="session")
def release(tmp_path_factory) -> Path:
    return make_release(tmp_path_factory.mktemp("release"))
