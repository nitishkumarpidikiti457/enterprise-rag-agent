"""Smoke test: proves the package is installed and importable."""

import app


def test_package_imports():
    # If `pip install -e .` worked, the `app` package is importable from anywhere.
    assert app is not None