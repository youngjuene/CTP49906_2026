# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Script-free slice rendering.

These fragments exist because the interactive page cannot be displayed in a
host that refuses to run an embedded page's scripts. A fragment that smuggled a
``<script>`` back in, or that leaked an unescaped vocabulary string into the
host document, would defeat the reason for having them -- so that is what these
tests pin, alongside the shape of the grid and the meaning of its colours.
"""

from __future__ import annotations

from html.parser import HTMLParser

import numpy as np

from jlens.vis import (
    SliceData,
    compute_slice,
    position_top_k_rows,
    slice_grid_html,
    token_rank_grid_html,
)

SEQ_LEN, N_LAYERS, TOP_N = 5, 3, 4


class _parse(HTMLParser):
    """The tag names, attribute names and human-visible text of a fragment,
    read the way a browser reads it rather than by string matching."""

    def __init__(self, markup: str) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: set[str] = set()
        self.attrs: set[str] = set()
        self._text: list[str] = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for name, value in attrs:
            self.attrs.add(name)
            self._text.append(value or "")

    def handle_data(self, data):
        self._text.append(data)

    @property
    def content(self) -> str:
        return "\n".join(self._text)


def _slice(**overrides) -> SliceData:
    """A slice whose final-layer top-1 is token 100 at every position, with
    layer 1 agreeing at position 0 and only ranking it fourth at position 1."""
    top_ids = np.full((SEQ_LEN, N_LAYERS, TOP_N), 900, dtype=np.int32)
    top_ids[:, -1, 0] = 100  # final layer's answer, every position
    top_ids[0, 1, 0] = 100  # agrees outright
    top_ids[1, 1, 3] = 100  # present, but not the cell's top-1
    fields = {
        "seq_len": SEQ_LEN,
        "layers": [0, 4, 9],
        "context_token_ids": list(range(SEQ_LEN + 2)),
        "context_token_strs": [f" w{i}" for i in range(SEQ_LEN + 2)],
        "top_ids": top_ids,
        "top_ranks": np.tile(
            np.arange(TOP_N, dtype=np.int32), (SEQ_LEN, N_LAYERS, 1)
        ),
        "tracked_token_ids": [100, 900],
        "rank_tensor": np.stack(
            [
                np.full((SEQ_LEN, N_LAYERS), 7, dtype=np.int32),
                np.full((SEQ_LEN, N_LAYERS), 4321, dtype=np.int32),
            ],
            axis=-1,
        ),
        "vocab_fragment": {100: " euro", 900: " lira", **{i: f" w{i}" for i in range(7)}},
        "vocab_size": 150_000,
        "ctx_offset": 2,
    }
    fields.update(overrides)
    return SliceData(**fields)


def test_grid_carries_no_script_and_no_external_asset():
    out = slice_grid_html(_slice())
    assert "<script" not in out.lower()
    assert "<iframe" not in out.lower()
    assert "http://" not in out and "https://" not in out


def test_grid_has_one_cell_per_position_and_layer():
    out = slice_grid_html(_slice())
    assert out.count("<td ") == SEQ_LEN * N_LAYERS
    # Layer headers name the fitted layers; the last is the model's own output.
    assert ">L0<" in out and ">L4<" in out and ">L9★<" in out


def test_colour_separates_agreement_from_presence_from_absence():
    out = slice_grid_html(_slice())
    # `;color:` so the legend's own swatches (background only) are not counted.
    dark, mid, pale = "#08519c;color:", "#9ecae1;color:", "#f7fbff;color:"
    # position 0/layer 1 agrees outright, position 1/layer 1 only has it in
    # top-K, and the whole first column never mentions it.
    assert out.count(f"background:{dark}") == SEQ_LEN + 1  # final column + one
    assert out.count(f"background:{mid}") == 1
    assert out.count(f"background:{pale}") == SEQ_LEN * N_LAYERS - SEQ_LEN - 2


def test_vocabulary_strings_cannot_escape_into_the_host_document():
    hostile = '<img src=x onerror="alert(1)">'
    sd = _slice()
    sd.vocab_fragment[900] = hostile
    out = slice_grid_html(sd)
    # The payload may survive as inert text; what must not survive is the tag
    # or the attribute boundary it would need to become markup again. Parse
    # rather than string-match, so this asks the question a browser asks.
    parsed = _parse(out)
    assert "img" not in parsed.tags
    assert "onerror" not in parsed.attrs
    assert hostile in parsed.content  # inert, in a tooltip


def test_absolute_positions_are_labelled_through_the_context_offset():
    out = slice_grid_html(_slice())
    # ctx_offset=2, so slice row 0 is prompt position 2, not 0.
    assert "prompt position 2" in out
    assert "prompt position 6" in out
    assert "prompt position 0" not in out


def test_windowing_says_what_it_dropped():
    out = slice_grid_html(_slice(), max_positions=2)
    assert out.count("<td ") == 2 * N_LAYERS
    assert "3 earlier position(s) are not drawn" in out


def test_rank_map_shades_by_rank_and_reports_the_best_cell():
    out = token_rank_grid_html(_slice(), [100])
    assert out.count("<td ") == SEQ_LEN * N_LAYERS
    assert "best rank 7" in out
    assert ">7<" in out  # the rank itself, not a colour-only encoding


def test_rank_map_summary_describes_the_rows_it_actually_drew():
    """A minimum taken over the whole slice would caption a windowed grid with
    a cell the reader cannot find in it."""
    sd = _slice()
    sd.rank_tensor[:, :, 0] = 5
    sd.rank_tensor[0, 0, 0] = 1  # best overall, in a row the window drops
    out = token_rank_grid_html(sd, [100], max_positions=2)
    assert "best rank 5" in out
    assert "best rank 1" not in out
    assert "prompt position 5" in out  # ctx_offset 2 + first drawn row


def test_rank_columns_stay_distinct_when_two_tokens_share_a_label():
    sd = _slice()
    sd.vocab_fragment[900] = " euro"  # decodes identically to token 100
    row = position_top_k_rows(sd, 0, token_ids=[100, 900])[0]
    assert "rank of ' euro'" in row
    assert "rank of ' euro' · id 900" in row


def test_rank_map_says_when_a_token_was_never_tracked():
    out = token_rank_grid_html(_slice(), [12345])
    assert "never tracked" in out
    assert "<td " not in out


def test_rank_map_with_no_selection_is_a_prompt_not_an_empty_table():
    out = token_rank_grid_html(_slice(), [])
    assert "<table" not in out
    assert "No token selected" in out


def test_renders_a_slice_that_compute_slice_actually_produced():
    """The synthetic fixtures above pin behaviour; this pins the contract with
    :func:`compute_slice`, whose windowing sets ``ctx_offset`` and whose final
    layer is appended rather than fitted."""
    from jlens.fitting import fit

    from .tiny import TinyDecoder

    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(model, ["abcdefghij " * 4], source_layers=[0, 1, 2], dim_batch=4)
    sd = compute_slice(model, lens, "the quick brown fox", last_n_tokens=6, top_n=3)

    out = slice_grid_html(sd, tooltip_k=3)
    assert out.count("<td ") == sd.seq_len * len(sd.layers)
    assert f"L{sd.layers[-1]}★" in out
    ranks = token_rank_grid_html(sd, sd.tracked_token_ids[:1])
    assert ranks.count("<td ") == sd.seq_len * len(sd.layers)
    assert len(position_top_k_rows(sd, sd.seq_len - 1)) == len(sd.layers)


def test_position_rows_cover_every_layer_and_the_asked_for_tokens():
    rows = position_top_k_rows(_slice(), 0, top_k=2, token_ids=[100, 12345])
    assert len(rows) == N_LAYERS
    assert rows[-1]["Layer"].endswith("★ model output")
    assert "' euro' (0)" in rows[-1]["Top 2 candidates (rank)"]
    assert rows[0]["rank of ' euro'"] == "7"
    assert rows[0]["rank of '<id 12345>'"] == "not tracked"
