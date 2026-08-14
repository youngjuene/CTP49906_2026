"""CPU logic tests for F3 (the logit-lens probe grid). No GPU / no widget.

Pins the claims the widget is built to make, against the *committed* CSV rather
than a synthetic fixture: if a re-run of the sweep changes the numbers, these
tests fail and the classroom narrative gets re-checked instead of quietly
drifting. The anchors are the degeneracy itself (`Layer_34` decodes to 6
distinct tokens, 245 of them junk, across 248 audio positions), the wire format
(uint16 little-endian, row-major `[position][layer]` -- an off-by-one in the
stride would still render a plausible-looking grid), the payload budget, and the
CSP rule that the ESM imports nothing remote.

`ProbeGrid` itself is deliberately never imported: the prep functions must stay
usable without anywidget installed.

Run:  python -m pytest avllm_interpretability/tests/test_probe_grid.py
  or:  python avllm_interpretability/tests/test_probe_grid.py
"""
import base64
import csv
import json
import struct
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.probe_grid import (  # noqa: E402
    CONTENT as CONTENT_CLASS,
    build_probe_grid_pack, classify_probe_token, probe_grid_layer_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "precomputed" / "logit_lens_audio_token_analysis.csv"
JS_PATH = ROOT / "src" / "probe_grid.js"
CSS_PATH = ROOT / "src" / "probe_grid.css"

PACK = build_probe_grid_pack(CSV_PATH)


def _raw_rows():
    """The CSV re-parsed independently of the module under test."""
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], [r for r in rows[1:] if r]


def test_pack_has_the_measured_shape():
    assert len(PACK["positions"]) == 248
    assert len(PACK["layer_names"]) == 36
    assert PACK["layer_names"][0] == "Layer_0" and PACK["layer_names"][-1] == "Layer_35"
    assert len(PACK["vocab"]) == 911         # 8,928 cells collapse to 911 strings
    assert len(PACK["vocab_class"]) == len(PACK["vocab"])
    assert len(PACK["token_index"]) == 2 * 248 * 36 == 17856
    assert PACK["seconds_per_position"] == 0.04


def test_positions_are_absolute_and_carry_the_interleave_gap():
    # The ordinal->time mapping is only defensible because these are NOT
    # contiguous: two runs with a 599 gap, 248 x 0.04 s = 9.92 s of audio.
    pos = PACK["positions"]
    assert pos[0] == 621 and pos[-1] == 1466
    gaps = [(pos[i], pos[i + 1]) for i in range(len(pos) - 1) if pos[i + 1] - pos[i] != 1]
    assert gaps == [(670, 1269)]
    assert abs(len(pos) * PACK["seconds_per_position"] - 9.92) < 1e-9


def test_token_index_is_uint16_le_row_major_over_position_then_layer():
    _, rows = _raw_rows()
    n_layers = len(PACK["layer_names"])
    vocab = PACK["vocab"]
    buf = PACK["token_index"]
    samples = [(0, 0), (0, 35), (1, 17), (123, 7), (200, 34), (247, 0), (247, 35)]
    for p, k in samples:
        (slot,) = struct.unpack_from("<H", buf, 2 * (p * n_layers + k))
        assert slot < len(vocab)
        assert vocab[slot] == rows[p][2 + k], (p, k, vocab[slot], rows[p][2 + k])
    # ...and prove the check discriminates. The transposed (layer-major)
    # packing shares the two corners `(0, 0)` and `(247, 35)` with this one, so
    # only the interior samples can tell the strides apart -- every one of them
    # must decode to a different string under the wrong stride.
    n_pos = len(rows)
    interior = [(p, k) for p, k in samples if (p, k) not in {(0, 0), (247, 35)}]
    for p, k in interior:
        (slot,) = struct.unpack_from("<H", buf, 2 * (k * n_pos + p))
        assert vocab[slot] != rows[p][2 + k], (p, k)


def test_layer_summary_pins_the_degeneracy():
    summary = probe_grid_layer_summary(PACK)
    assert len(summary) == 36
    by_name = {row["name"]: row for row in summary}

    l34 = by_name["Layer_34"]
    assert l34["layer"] == 34
    assert l34["unique"] == 6
    assert l34["junk"] == 245
    assert l34["matches_final"] == 92
    assert l34["modal_token"] == " " and l34["modal_count"] == 236

    l35 = by_name["Layer_35"]
    assert l35["unique"] == 42
    assert l35["junk"] == 193
    assert l35["matches_final"] == 248  # the final layer trivially equals itself

    # The headline: the last layer is mostly not-words, so "the prediction
    # crystallizes" is not what this surface shows.
    assert l35["junk"] / len(PACK["positions"]) > 0.7


def test_classify_probe_token_rules():
    assert classify_probe_token(" ") == 1        # whitespace-only -> junk
    assert classify_probe_token("\n") == 1
    assert classify_probe_token(",") == 1        # category Po
    assert classify_probe_token("") == 1
    assert classify_probe_token("=-=-=-=-") == 1  # all symbol/punctuation
    assert classify_probe_token("the") == 0
    assert classify_probe_token("制约") == 0      # CJK is Lo, not junk
    assert classify_probe_token(" piano") == 0   # mixed -> content
    assert classify_probe_token("a�b") == 2  # undecodable beats content
    assert classify_probe_token("�") == 2   # ...and beats the junk rule


def test_vocab_classes_cover_all_three_outcomes():
    # No per-slot loop against `classify_probe_token`: `build_probe_grid_pack`
    # computes `vocab_class` by calling exactly that function, so restating it
    # would pass for any classifier, right or wrong. The classifier itself is
    # pinned by `test_classify_probe_token_*`; what is worth asserting here is
    # that the real CSV exercises all three outcomes and that the array lines up
    # with the vocab it indexes.
    assert set(PACK["vocab_class"]) == {0, 1, 2}
    assert len(PACK["vocab_class"]) == len(PACK["vocab"])
    # And spot-check the alignment against the CSV's own strings, independently
    # of how the pack was built.
    for slot, token in enumerate(PACK["vocab"]):
        if token.strip() and any(ch.isalnum() for ch in token):
            assert PACK["vocab_class"][slot] == CONTENT_CLASS, token
            break
    else:
        raise AssertionError("no content token in the vocab — pack looks wrong")


def test_empty_and_headers_only_csv_return_none():
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.csv"
        empty.write_text("", encoding="utf-8")
        assert build_probe_grid_pack(empty) is None

        headers = Path(tmp) / "headers.csv"
        headers.write_text("Token_Position,Token_Type,Layer_0,Layer_1\n", encoding="utf-8")
        assert build_probe_grid_pack(headers) is None

        blank_rows = Path(tmp) / "blank.csv"
        blank_rows.write_text("Token_Position,Token_Type,Layer_0\n\n\n", encoding="utf-8")
        assert build_probe_grid_pack(blank_rows) is None


def test_short_row_fails_loud_instead_of_shifting_the_grid():
    with tempfile.TemporaryDirectory() as tmp:
        ragged = Path(tmp) / "ragged.csv"
        ragged.write_text(
            "Token_Position,Token_Type,Layer_0,Layer_1\n621,audio,a,b\n622,audio,c\n",
            encoding="utf-8",
        )
        try:
            build_probe_grid_pack(ragged)
        except ValueError as exc:
            assert "columns" in str(exc)
        else:
            raise AssertionError("a ragged row must raise, not silently shift cells")


def test_pack_stays_inside_the_payload_budget():
    # Measures the *data* pack, not the wire: the two Bytes traits travel as
    # buffers that marimo's JSON channel base64-encodes, so the transmitted size
    # is ~4/3 of the blob half. Both are checked, because the budget that matters
    # is the one that crosses the socket once per logit-lens run into a notebook
    # that is also holding a model on the GPU.
    lists = sum(
        len(json.dumps(PACK[key]).encode("utf-8"))
        for key in ("layer_names", "positions", "vocab")
    )
    blobs = len(PACK["vocab_class"]) + len(PACK["token_index"])
    assert lists + blobs < 40 * 1024, lists + blobs

    wire = lists + len(base64.b64encode(PACK["vocab_class"])) + len(
        base64.b64encode(PACK["token_index"])
    )
    assert wire < 56 * 1024, wire
    assert len(PACK["token_index"]) == 17856


def test_ui_text_states_the_rule_and_the_caveat():
    # Both strings are rendered in the widget: the junk rule so students can
    # contest the classification, the caveat so nobody reads these probes as
    # calibrated predictions.
    assert "P*, Z*, C*, S*" in PACK["junk_definition"]
    assert "U+FFFD" in PACK["junk_definition"]
    assert "RMSNorm" in PACK["caveat"]
    assert "entropy" in PACK["caveat"]


def test_esm_imports_nothing_remote():
    source = JS_PATH.read_text(encoding="utf-8")
    assert "http://" not in source
    assert "https://" not in source
    assert "cdn." not in source
    assert "export function render" in source
    # py -> js only: a trait write from the ESM would re-run every marimo cell
    # that references the widget, and the GPU cells are next door. Checked
    # against the code only -- the comments name the forbidden calls on purpose.
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("//")
    )
    assert "save_changes" not in code
    assert "model.set(" not in code
    assert CSS_PATH.exists()  # _css points here


def _unscoped_selectors(css, root=".ctp-probe-grid"):
    """Selectors in `css` that are not confined to `root`.

    `_css` is injected into the host page, so one unscoped selector restyles the
    whole notebook. Splitting on `{` rather than on lines is what makes this see
    a group written on one line (`div, .ctp-probe-grid .pg-x {`) as two
    selectors; block comments are stripped first so no comment-state tracking is
    needed, which is what previously forced a blanket skip of `*` -- the single
    most destructive selector that can leak.
    """
    import re

    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    bad = []
    for chunk in css.split("{")[:-1]:
        prelude = chunk.rsplit("}", 1)[-1].strip()
        if not prelude or prelude.startswith("@"):
            continue  # at-rule prelude; the rules inside it are their own chunks
        for selector in prelude.split(","):
            selector = " ".join(selector.split())
            if not selector or selector in ("from", "to"):
                continue
            if re.fullmatch(r"\d+(\.\d+)?%", selector):
                continue  # keyframe stop
            if root not in selector:
                bad.append(selector)
    return bad


def test_css_is_scoped_to_the_widget_root():
    assert _unscoped_selectors(CSS_PATH.read_text(encoding="utf-8")) == []


def test_the_scoping_check_catches_every_way_of_hiding_an_unscoped_selector():
    # Guards the guard, by calling the *same* helper the assertion above uses --
    # a duplicated parser could drift from the real one and silently stop
    # catching anything. Each case below is a real way to write the defect.
    ok = ".ctp-probe-grid .pg-chip {\n  color: red;\n}"
    assert _unscoped_selectors(ok) == []

    cases = {
        "bare rule": "div { color: red; }",
        "group across lines": "div,\n.ctp-probe-grid {\n color: red;\n}",
        "group on one line": "div, .ctp-probe-grid .pg-x { color: red; }",
        "universal": "* { box-sizing: border-box; }",
        "universal in a group": "*,\n.ctp-probe-grid .pg-x { margin: 0; }",
        "inside a media query": "@media (min-width: 40em) {\n  div { color: red; }\n}",
    }
    for label, css in cases.items():
        assert _unscoped_selectors(css), f"missed: {label}"

    # ...and things that are scoped, or are not selectors at all, stay quiet.
    assert _unscoped_selectors("@media (min-width: 40em) {\n"
                               "  .ctp-probe-grid .pg-x { color: red; }\n}") == []
    assert _unscoped_selectors("/* div { color: red; } */\n"
                               ".ctp-probe-grid { color: blue; }") == []
    assert _unscoped_selectors("@keyframes pg-fade {\n  from { opacity: 0; }\n"
                               "  to { opacity: 1; }\n}") == []


@pytest.mark.needs_anywidget
def test_the_pack_survives_construction_and_the_blobs_stay_binary():
    # The widget class carries the whole GPU-safety contract, and until now the
    # suite never constructed it. Imported locally on purpose: the module must
    # still import on a bare interpreter with `ProbeGrid` simply absent.
    from ipywidgets.widgets.widget import _remove_buffers

    from src.probe_grid import ProbeGrid

    widget = ProbeGrid(**PACK)
    state = widget.get_state()

    # traitlets only warns on an unknown kwarg, so a pack key the class forgot to
    # declare would be silently dropped and the grid would render empty.
    assert set(PACK) <= set(state), set(PACK) - set(state)
    assert state["token_index"] == PACK["token_index"]
    assert state["positions"] == PACK["positions"]

    # The two big traits must leave as out-of-band buffers, not inline JSON.
    _stripped, buffer_paths, buffers = _remove_buffers(state)
    carried = {tuple(path) for path in buffer_paths}
    assert ("token_index",) in carried, carried
    assert ("vocab_class",) in carried, carried
    assert sum(len(b) for b in buffers) == len(PACK["token_index"]) + len(
        PACK["vocab_class"]
    )


@pytest.mark.needs_anywidget
def test_the_widget_declares_no_trait_the_esm_writes_back():
    # The invariant that keeps a widget interaction from reaching the GPU: every
    # trait is py->js. marimo re-runs *every* cell referencing an element when any
    # synced trait changes from the browser, and this widget sits upstream of the
    # knockout experiment.
    import re

    from src.probe_grid import ProbeGrid

    raw = JS_PATH.read_text(encoding="utf-8")
    # Strip comments first: the ESM *documents* why it never calls
    # `model.save_changes()`, and a naive substring search matches that prose.
    code = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    code = re.sub(r"(?m)//.*$", "", code)

    written = set(re.findall(r"model\.set\(\s*[\"']([A-Za-z_][\w]*)[\"']", code))
    assert not written, f"ESM writes back: {written}"
    assert "save_changes" not in code
    # And the explanation is still there — this is a rule future edits must know.
    assert "save_changes" in raw

    ours = set(PACK)
    assert ours <= set(ProbeGrid(**PACK).traits(sync=True))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
