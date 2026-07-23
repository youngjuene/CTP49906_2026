from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "digital_storytelling" / "CTP49906_storytelling_molab.py"


def test_notebook_compiles() -> None:
    subprocess.run([sys.executable, "-m", "py_compile", str(NOTEBOOK)], check=True)


def test_notebook_passes_strict_marimo_when_checker_is_available() -> None:
    if importlib.util.find_spec("marimo") is None:
        pytest.skip("marimo is verified separately with the pinned uvx checker")
    subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict", str(NOTEBOOK)],
        cwd=ROOT,
        check=True,
    )
