"""Probe grid for the logit-lens audio sweep (F3) -- replaces the layer scrubber.

The scrubber invites students to watch "early-layer noise crystallize into the
final prediction". On this clip that narrative is false, and the committed CSV
says so out loud: in `precomputed/logit_lens_audio_token_analysis.csv` the
final-layer column is ~78% whitespace and punctuation, and `Layer_34` decodes to
**6** distinct tokens across 248 audio positions, 245 of them junk. A scrubber
shows one layer at a time, so the degeneracy is exactly the thing it hides. This
widget puts all 36 layers x 248 positions on screen at once and makes the
degeneracy the default reading.

Two choices carry the argument and must not be "improved" away:

* The colour channel is junk-vs-content, **not** entropy. `summarize_logits`
  log-softmaxes raw `lm_head` output of *un-RMSNormed* hidden states (see the
  caveat in `logitlens_experiment.py:242-248`). Residual norm grows with depth,
  so normalized entropy falls with depth for scale reasons that are independent
  of any decision the model makes -- an entropy surface would restate the
  crystallization myth in a new coordinate and look like evidence. "Is this
  probe token junk" is scale-invariant, so it is the honest channel.

* Times key off the **ordinal** index of the audio positions (0..N-1), never the
  absolute token position. The 248 positions are two runs -- 621-670 and
  1269-1466, a 599 gap -- because of the `use_audio_in_video` interleave.
  248 x 0.04 s = 9.92 s matches the clip's 9.915 s audio stream, so the 25 Hz
  ordinal mapping is the right one; an absolute-position mapping would be off by
  ~24 s in the middle of the strip. The absolute position still rides along in
  the tooltip, because the knockout experiment speaks in absolute positions.

The prep functions are pure stdlib -- no torch, no marimo, no anywidget at
module scope -- so they import and unit-test in a plain 3.10 venv. `anywidget`
is imported behind a guard and `ProbeGrid` only exists when it is installed.
"""

import csv
import unicodedata
from array import array
from collections import Counter
from pathlib import Path
from sys import byteorder

try:
    import anywidget
    import traitlets
except ImportError:  # pure functions must still import in a bare venv
    anywidget = None

# unicodedata.category() prefixes that count as "not a word": P punctuation,
# Z separator, C control/format/surrogate/private-use, S symbol. Deliberately
# excludes L (letters, incl. CJK Lo) and N (digits), so "制约" and "3" are
# content. Exposed as a constant because the rule is contestable and the UI
# prints it in words for exactly that reason.
JUNK_CATEGORY_PREFIXES = ("P", "Z", "C", "S")

CONTENT, JUNK, UNDECODABLE = 0, 1, 2

JUNK_DEFINITION = (
    "Junk = a probe token whose characters are ALL Unicode punctuation, "
    "separator, control/format or symbol (categories P*, Z*, C*, S*), or that "
    "is empty / whitespace-only. Everything else -- letters (including CJK), "
    "digits, and any mixed string -- counts as content. Tokens containing the "
    "replacement character U+FFFD are counted separately as undecodable. "
    "Contest this rule: a lone '%' or an arrow can carry meaning, and a "
    "content-classed token can still be gibberish. The claim it supports is "
    "narrow -- a layer whose column is almost all junk is not 'a prediction "
    "crystallizing'."
)

PROBE_CAVEAT = (
    "These probes apply lm_head directly to raw layer outputs, skipping the "
    "thinker's final RMSNorm, and audio positions carry no calibrated "
    "next-token objective -- they are probes, not intermediate predictions. "
    "That is also why this grid is not coloured by entropy: residual norm grows "
    "with depth, so an un-normed entropy falls with depth for scale reasons "
    "alone, which would look like 'crystallization' no matter what the model "
    "did. Junk-vs-content is scale-invariant."
)


def classify_probe_token(token):
    """0 = content, 1 = junk, 2 = undecodable.

    Order matters: a token that is *only* whitespace is junk even though U+0020
    is category Zs (the whitespace test just gets there first and is cheaper),
    and the U+FFFD test precedes the category test because a replacement
    character means the byte-level BPE piece never formed a valid string -- that
    is a different failure from "the model predicted a comma" and gets its own
    colour rather than being folded into junk.
    """
    text = "" if token is None else str(token)
    if not text.strip():
        return JUNK
    if "�" in text:
        return UNDECODABLE
    if all(unicodedata.category(ch)[0] in JUNK_CATEGORY_PREFIXES for ch in text):
        return JUNK
    return CONTENT


def _pack_uint16_le(values):
    """uint16 little-endian buffer. `array` is native-order, so swap on BE."""
    buf = array("H", values)
    if byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _unpack_uint16_le(payload):
    buf = array("H")
    buf.frombytes(payload)
    if byteorder == "big":
        buf.byteswap()
    return buf


def build_probe_grid_pack(csv_path, *, seconds_per_position=0.04):
    """Parse the logit-lens CSV into the widget's trait kwargs (or None).

    Column 0 is the absolute token position, column 1 the token type, columns
    2.. one per layer. Returns a dict whose keys are *exactly* the `ProbeGrid`
    traits, so `ProbeGrid(**build_probe_grid_pack(path))` is the whole call
    site; returns None when the file holds no data rows (a headers-only CSV is
    what a half-finished GPU run leaves behind, and the notebook should render
    nothing rather than an empty grid).

    The 8,928 cells collapse to 911 distinct strings, so the payload is a
    de-duplicated `vocab` plus a uint16 index grid: ~30 KiB instead of the
    ~120 KiB the raw strings would cost, and classification runs 911 times
    instead of 8,928. Row-major `[position][layer]` because every reader --
    tooltip, trajectory panel, per-layer stats -- walks one position's 36 layers
    or strides one layer's 248 positions, and the former wants contiguity.
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return None

    header, data = rows[0], [r for r in rows[1:] if r]
    if not data:
        return None
    layer_names = [str(name) for name in header[2:]]
    n_layers = len(layer_names)
    if n_layers == 0:
        return None

    positions = []
    flat = []
    vocab = []
    index_of = {}
    for row_number, row in enumerate(data, start=2):
        if len(row) != n_layers + 2:
            # Fail loud: a short row would silently shift every later cell one
            # layer to the left and the grid would still look plausible.
            raise ValueError(
                f"{csv_path}:{row_number} has {len(row)} columns, "
                f"expected {n_layers + 2}"
            )
        positions.append(int(row[0]))
        for cell in row[2:]:
            slot = index_of.get(cell)
            if slot is None:
                slot = len(vocab)
                index_of[cell] = slot
                vocab.append(cell)
            flat.append(slot)

    if len(vocab) > 0xFFFF:
        raise ValueError(f"{len(vocab)} distinct tokens overflows the uint16 grid")

    return {
        "layer_names": layer_names,
        "positions": positions,
        "vocab": vocab,
        "vocab_class": bytes(classify_probe_token(tok) for tok in vocab),
        "token_index": _pack_uint16_le(flat),
        "seconds_per_position": float(seconds_per_position),
        "junk_definition": JUNK_DEFINITION,
        "caveat": PROBE_CAVEAT,
    }


def probe_grid_layer_summary(pack):
    """Per-layer degeneracy stats, computed from the pack alone.

    `matches_final` counts positions whose probe token already equals that
    position's own last-layer token -- the honest version of "the prediction has
    settled". It is compared per position, not against a global modal token, so
    a layer that is 245/248 junk does not get credit for agreeing with a final
    layer that is itself mostly junk unless it agrees *cell by cell*.
    """
    layer_names = pack["layer_names"]
    vocab = pack["vocab"]
    vocab_class = pack["vocab_class"]
    n_layers = len(layer_names)
    n_pos = len(pack["positions"])
    grid = _unpack_uint16_le(pack["token_index"])

    summary = []
    for k, name in enumerate(layer_names):
        column = [grid[p * n_layers + k] for p in range(n_pos)]
        finals = [grid[p * n_layers + (n_layers - 1)] for p in range(n_pos)]
        counts = Counter(column)
        # Break count ties by vocab index so the reported modal token is stable
        # across runs rather than depending on dict insertion order.
        modal = min(counts, key=lambda i: (-counts[i], i)) if counts else None
        summary.append(
            {
                "layer": k,
                "name": name,
                "unique": len(counts),
                "junk": sum(1 for i in column if vocab_class[i] == JUNK),
                "undecodable": sum(1 for i in column if vocab_class[i] == UNDECODABLE),
                # strict=: a length mismatch here would silently truncate the
                # count and report a plausible-looking number instead of failing.
                "matches_final": sum(
                    1 for a, b in zip(column, finals, strict=True) if a == b
                ),
                "modal_token": vocab[modal] if modal is not None else "",
                "modal_count": counts[modal] if modal is not None else 0,
            }
        )
    return summary


if anywidget is not None:

    class ProbeGrid(anywidget.AnyWidget):
        """36 layers x 248 audio positions, coloured by junk-vs-content.

        **Every trait is py->js only.** Nothing in `probe_grid.js` ever calls
        `model.set`/`save_changes`: a js->py trait write would mark the widget
        dirty and re-run every marimo cell that references it, and the cells
        next door hold a multi-GPU model. Layer scrubbing, pinning and the junk
        toggle are therefore all client-side state, deliberately not persisted.

        `_esm`/`_css` are real `Path`s -- anywidget coerces them to FileContents
        and picks up edits on reload, which matters while iterating in class.
        """

        _esm = Path(__file__).parent / "probe_grid.js"
        _css = Path(__file__).parent / "probe_grid.css"

        layer_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
        positions = traitlets.List(traitlets.Int()).tag(sync=True)
        vocab = traitlets.List(traitlets.Unicode()).tag(sync=True)
        # Bytes is pulled out of the model state into a separate buffer by
        # ipywidgets' `_remove_buffers`, and lands in JS as a DataView. It is
        # still base64 on the wire -- marimo's JSON channel encodes the buffer
        # list that way -- so budget ~4/3 of the byte counts below, not 1x:
        # uint8 class per vocab entry, uint16 LE row-major grid.
        vocab_class = traitlets.Bytes(b"").tag(sync=True)
        token_index = traitlets.Bytes(b"").tag(sync=True)
        seconds_per_position = traitlets.Float(0.04).tag(sync=True)
        junk_definition = traitlets.Unicode(JUNK_DEFINITION).tag(sync=True)
        caveat = traitlets.Unicode(PROBE_CAVEAT).tag(sync=True)
