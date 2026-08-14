"""Static checks on both marimo notebooks for constructs that break at runtime.

These are not style rules. Each one below stands for a defect that actually
shipped and that neither the dataflow check nor the execution smoke tests catch
reliably, because whether it fires depends on incidental naming in *other* cells.

Run:  python -m pytest tests/test_notebook_lint.py
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "avllm_interpretability" / "CTP49906_avllm_molab.py",
    ROOT / "jacobian-lens" / "CTP49906_jlens_molab.py",
]


def _cells(path):
    """Each `@app.cell`-decorated function in the notebook."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_no_lambda_default_captures_a_cell_local_name(notebook):
    """A lambda default over an `_`-prefixed name is a live NameError risk.

    marimo rewrites underscore-prefixed names to per-cell mangled ones. A lambda
    *default* that captures such a name can resolve against a different cell's
    prefix -- observed as
    `NameError: name '_cell_NCOB_p' is not defined. Did you mean '_cell_pHFh_p'?`
    from an `mo.download(data=lambda _f=_p: _f.read_bytes())` inside a list
    comprehension. Whether it fires depends on what other cells name their
    locals, so the construct is banned rather than the specific collision.

    Use `functools.partial(Type.method, value)` instead: no default, no closure,
    nothing for the rewriter to disagree about.
    """
    offenders = []
    for cell in _cells(notebook):
        for node in ast.walk(cell):
            if not isinstance(node, ast.Lambda):
                continue
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in defaults:
                if isinstance(default, ast.Name) and default.id.startswith("_"):
                    offenders.append(f"{notebook.name}:{default.lineno} lambda …={default.id}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_downloads_are_lazy(notebook):
    """`mo.download(data=...)` must not receive eager bytes.

    marimo materialises an eager payload into a virtual file *and* a shared
    memory segment at render time. These notebooks offer ~400 MB lens files and
    multi-megabyte slice pages, for buttons a student may never press.

    A `.read_bytes()` call passed directly as `data=` is the eager form; a
    `partial`/callable is not.
    """
    offenders = []
    for cell in _cells(notebook):
        for node in ast.walk(cell):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func).endswith("download")):
                continue
            for kw in node.keywords:
                if kw.arg != "data":
                    continue
                rendered = ast.unparse(kw.value)
                if rendered.endswith(".read_bytes()"):
                    offenders.append(f"{notebook.name}:{kw.value.lineno} data={rendered}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_form_validators_read_their_payload_defensively(notebook):
    """A batch's value is a partial dict until the frontend pushes every child.

    Indexing it directly (`value["layers"]`) raises `KeyError` on the first
    render instead of validating, so every validator must use `.get`.
    """
    offenders = []
    for cell in _cells(notebook):
        for node in ast.walk(cell):
            if not (isinstance(node, ast.FunctionDef) and "validate" in node.name):
                continue
            param = node.args.args[0].arg if node.args.args else None
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == param
                ):
                    offenders.append(
                        f"{notebook.name}:{sub.lineno} {node.name} indexes {ast.unparse(sub)}"
                    )
    assert not offenders, "\n".join(offenders)
