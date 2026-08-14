"""Dataflow invariants for the molab notebook. Parses cells; executes none.

The widgets in this notebook sit next to cells that load 8 GB of weights and run
GPU generations, and marimo's reactivity is what keeps them apart. There is no
per-trait subscription: *any* synced trait changing from the browser re-runs
*every* cell that references the element. So the rules below are not style —
breaking one of them means a student dragging a slider triggers a fresh
generation, or a form silently resets and blanks a result that cost a minute.

`marimo export script` (see tests/README.md) already proves the graph builds at
all — no cycles, no name defined twice. This file checks the properties specific
to the widgets.

Run:  python -m pytest avllm_interpretability/tests/test_notebook_graph.py
"""
import sys
from pathlib import Path

import pytest

marimo = pytest.importorskip("marimo", reason="graph check needs marimo installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NOTEBOOK = Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab.py"

# A cell is "expensive" if it loads weights, generates, or scores. These are the
# cells no widget interaction may ever reach.
EXPENSIVE_MARKERS = (
    ".generate(",
    "thinker(**",
    "from_pretrained",
    "teacher_forced_delta(",
    "_tfd(",
)


@pytest.fixture(scope="module")
def cells():
    from marimo._ast.load import load_app

    app = load_app(str(NOTEBOOK))
    out = []
    for index, data in enumerate(app._cell_manager.cell_data()):
        if data.cell is None:
            continue
        out.append(
            {
                "index": index,
                "defs": set(data.cell.defs),
                "refs": set(data.cell.refs),
                "code": data.code,
            }
        )
    return out


def _names(cell):
    """A cell's refs minus Python builtins.

    marimo's `refs` include builtins (`print`, `len`, ...), which never appear in
    a cell's generated signature and are irrelevant to the dataflow graph.
    """
    import builtins

    return set(cell["refs"]) - (set(dir(builtins)) | {"__file__", "__name__"})


def _defining(cells, name):
    return [c for c in cells if name in c["defs"]]


def _referring(cells, name):
    return [c for c in cells if name in c["refs"]]


def _expensive(cells):
    return [c for c in cells if any(m in c["code"] for m in EXPENSIVE_MARKERS)]


def test_the_notebook_still_has_expensive_cells_to_protect(cells):
    # Guards the guard: if the markers stop matching, every assertion below
    # would pass vacuously.
    assert len(_expensive(cells)) >= 4


def test_probe_grid_has_no_readers(cells):
    # The whole reason ProbeGrid declares zero js->py traits. If any cell ever
    # reads `probe_grid.value`, dragging the grid re-runs that cell — and the
    # grid sits upstream of the knockout experiment.
    assert len(_defining(cells, "probe_grid")) == 1
    assert _referring(cells, "probe_grid") == []


def test_no_expensive_cell_reads_the_run_ledger(cells):
    # A getter-referencing cell re-runs on every append. If a GPU cell read the
    # ledger, recording a run would re-run the generation that produced it.
    bad = [c["index"] for c in _referring(cells, "get_runs") if c in _expensive(cells)]
    assert bad == [], f"expensive cells referencing get_runs: {bad}"


def test_no_form_cell_reads_the_run_ledger(cells):
    # Re-running a form cell mints a fresh element id and resets `.value` to
    # None, which fires the downstream `mo.stop` and blanks the result. A form
    # that re-ran on every append would erase the run being recorded.
    forms = [c for c in cells if ".form(" in c["code"]]
    assert forms, "expected the notebook to still use forms"
    bad = [c["index"] for c in _referring(cells, "get_runs") if c in forms]
    assert bad == [], f"form cells referencing get_runs: {bad}"


def test_the_state_cell_depends_only_on_run_once_constants(cells):
    # marimo mints a new SetFunctor when a state cell re-runs, and every cell
    # holding the old setter re-runs with it. Since three GPU cells call
    # `set_runs`, a *reactive* dependency here re-fires all three at once.
    # `mo` and `RESULTS_DIR` are both computed once and never change, so they are
    # the only refs this cell may hold.
    state = _defining(cells, "get_runs")
    assert len(state) == 1
    assert _names(state[0]) <= {"mo", "RESULTS_DIR"}, _names(state[0])

    # And RESULTS_DIR must itself be defined by a cell that no widget or form can
    # invalidate — otherwise "computed once" is only true today.
    results_dir = _defining(cells, "RESULTS_DIR")
    assert len(results_dir) == 1
    assert _names(results_dir[0]) <= {"PROJECT_DIR"}, _names(results_dir[0])


def test_the_ledger_writers_do_not_read_the_ledger(cells):
    # The GPU cells import run_record/append_run from this cell. If it also
    # referenced the state, it would re-run on every append — and take every
    # cell that imported from it along.
    writers = _defining(cells, "run_record")
    assert len(writers) == 1
    assert "get_runs" not in writers[0]["refs"]
    assert _defining(cells, "append_run") == writers


def test_no_cell_defines_and_reads_the_same_ui_element(cells):
    # marimo forbids reading `.value` in the cell that constructs the element.
    for name in ("probe_grid", "band_controls", "ko_controls", "tf_controls",
                 "verdict_form", "get_runs", "set_runs"):
        defining = {c["index"] for c in _defining(cells, name)}
        referring = {c["index"] for c in _referring(cells, name)}
        assert not (defining & referring), f"{name} is defined and read in {defining & referring}"


def test_knob_edits_do_not_invalidate_the_model_loads(cells):
    # The three-way parameter split exists so tweaking a prompt re-runs the
    # experiments and not ~8 GB of weights. Assert it rather than trusting the
    # comment that claims it.
    loaders = [c for c in cells if "from_pretrained" in c["code"] or "load_model_and_processor(" in c["code"]]
    loaders = [c for c in loaders if "load_model_and_processor(" in c["code"]]
    assert loaders, "expected dedicated model-loader cells"
    knobs = {"NFRAMES", "LOGIT_PROMPT", "ATTENTION_PROMPT", "KNOCKOUT_RULES",
             "MAX_NEW_TOKENS", "ATTENTION_CAPTURE_LAYERS"}
    for cell in loaders:
        assert not (cell["refs"] & knobs), (
            f"loader cell {cell['index']} depends on knobs {cell['refs'] & knobs} — "
            "editing a prompt would reload the model"
        )
