"""Make `from src.x import y` resolve however pytest is invoked.

The tests are written as `from src.roster import ...`, which only works with
feedback-atlas/ on sys.path. There is no installable package here (this is a
server plus a src/ folder, matching avllm_interpretability/), so rootdir-relative
imports are the whole mechanism. pytest's own sys.path insertion depends on the
invocation directory, so pin it here instead: conftest.py is imported before any
test module, from a path pytest computes itself.

Prepended, not appended, so the repo's `src` wins over any same-named package
that happens to be installed in the interpreter.

The optional-dependency gate below is what keeps this suite honest about the
project's one hard testing rule, inherited from avllm_interpretability/tests:
CPU only, no model weights, no network. numpy is the only hard requirement.
umap-learn is heavy (it drags in numba and llvmlite) and every test that needs it
is marked, so a bare interpreter skips rather than errors. sentence-transformers is
deliberately absent from this list -- no test may need it, and
test_embedder_contract.py asserts that importing src.embedder does not drag it in.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> feedback-atlas/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# find_spec, not a real import: importing sklearn or umap costs seconds and
# pulls in numba, and the answer we want is only "is it installed".
_OPTIONAL = {
    "needs_umap": "umap",
    "needs_fastapi": "fastapi",
    "needs_httpx": "httpx",
    # The viewer half. duckdb and pyarrow are what the Mosaic relation is made
    # of, and embedding_atlas supplies both the analyser and the Arrow IPC
    # writer the query endpoint answers with. All three are optional in exactly
    # the same sense as umap: absent, the server still runs the class on the
    # hand-written map, so their tests skip rather than fail.
    "needs_duckdb": "duckdb",
    "needs_pyarrow": "pyarrow",
    "needs_embedding_atlas": "embedding_atlas",
}

MISSING = {
    marker: module
    for marker, module in _OPTIONAL.items()
    if importlib.util.find_spec(module) is None
}


def pytest_collection_modifyitems(items):
    # A marker is a promise that the *module* still imports without the
    # dependency -- keep sklearn/umap/scipy imports inside the functions that
    # use them, never at module scope.
    if not MISSING:
        return
    for marker, module in MISSING.items():
        skip = pytest.mark.skip(reason=f"{module} is not installed")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
