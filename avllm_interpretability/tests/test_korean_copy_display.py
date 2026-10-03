"""Keep a scored model-token slot identifiable when its printable piece is empty.

Evaluate the notebook's actual stat expression without loading the GPU model.
The byte-continuation and special-token cases otherwise both appear as a dot;
a completed Korean character can also carry only one constituent token's score.
"""

import ast
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

mo = pytest.importorskip("marimo")
NOTEBOOK = Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab_kr.py"


class _StatAttributes(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "marimo-stat":
            self.values = {
                key.removeprefix("data-"): json.loads(value)
                for key, value in attrs
                if key.startswith("data-")
            }


def _minimum_card(pieces, kinds, deltas):
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    panel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tf_result_panel")
    card = next(
        node for node in ast.walk(panel)
        if isinstance(node, ast.Call)
        and any(
            kw.arg == "label" and isinstance(kw.value, ast.Constant)
            and str(kw.value.value).startswith("Δ가 가장 작은")
            for kw in node.keywords
        )
    )
    result = eval(
        compile(ast.Expression(card), str(NOTEBOOK), "eval"),
        {"mo": mo, "_tf_toks": pieces, "_tf_delta": deltas,
         "_tf_worst": min(range(len(deltas)), key=deltas.__getitem__) if deltas else 0,
         "_tf_res": {"caption_token_kinds": kinds}},
    )
    parser = _StatAttributes()
    parser.feed(result.text)
    return parser.values


@pytest.mark.parametrize(
    ("deltas", "position", "kind", "value"),
    [
        ([-5.0, 1.0, -0.2], 0, "문자 이어짐", "·"),
        ([1.0, -5.0, -0.2], 1, "텍스트 조각", "가"),
        ([1.0, 2.0, -5.0], 2, "특수 토큰", "·"),
    ],
)
def test_minimum_card_identifies_scored_slot_and_keeps_individual_delta(deltas, position, kind, value):
    card = _minimum_card(["", "가", ""], ["byte_continuation", "text", "special"], deltas)
    assert card["value"] == value
    assert f"위치 {position}" in card["caption"]
    assert kind in card["caption"]
    assert "-5.00 nats" in card["caption"]
    # For the text case, the visible group's delta is -4, not the model slot's -5.
    assert "-4.00 nats" not in card["caption"]


def test_minimum_card_handles_an_empty_caption_without_indexing_token_metadata():
    card = _minimum_card([], [], [])
    assert card["value"] == "—"
    assert card["caption"] == ""
