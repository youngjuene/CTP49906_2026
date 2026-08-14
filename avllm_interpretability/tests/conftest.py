"""Make `from src.x import y` resolve however pytest is invoked.

The tests are written as `from src.teacher_forcing import ...`, which only
works with avllm_interpretability/ on sys.path. There is no installable package
here (the project is a marimo notebook plus a src/ folder), so rootdir-relative
imports are the whole mechanism. pytest's own sys.path insertion depends on the
invocation directory, so pin it here instead: conftest.py is imported before any
test module, from a path pytest computes itself.

Prepended, not appended, so the repo's `src` wins over any same-named package
that happens to be installed in the interpreter.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> avllm_interpretability/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# find_spec, not `import anywidget`: importing it pulls in ipywidgets/comm and is
# pure cost when the answer is "yes, it is installed, carry on".
HAVE_ANYWIDGET = importlib.util.find_spec("anywidget") is not None


def pytest_collection_modifyitems(items):
    # Tests marked `needs_anywidget` skip rather than error on a bare
    # interpreter. The marker is a promise the *module* still imports without
    # anywidget — keep widget classes out of module-level imports.
    if HAVE_ANYWIDGET:
        return
    skip = pytest.mark.skip(reason="anywidget is not installed")
    for item in items:
        if "needs_anywidget" in item.keywords:
            item.add_marker(skip)
