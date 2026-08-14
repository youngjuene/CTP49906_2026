"""Execute the jlens notebook against a stub model — no GPU, no weights.

Why this exists: running the notebook is how the A.2 download cell was caught
raising ``NameError: name '_cell_XXXX_p' is not defined`` as soon as
``fitted_lens_files`` was non-empty. The file parses and the dataflow graph
builds; the branch is simply unreachable until a student has fitted a lens, so
the first time it could ever fire was in a classroom, on the button they need.

So the fixture deliberately puts a fitted lens on disk before running — that is
the state the crash needed, and several assertions below only mean anything in
it.

Note on coverage: this suite does **not** pin that specific NameError. Whether it
fires depends on incidental underscore naming in other cells, so the durable
guard is the static one in ``test_notebook_lint.py``, which bans the construct
outright. What this file guarantees is broader and duller: every cell body
actually executes, in dependency order, in the states a student will be in.

The stubs stand in for a 4B model and its reference lens. Everything the course
actually changed — the lens-source form and its four branches, the vocabulary
warm-up, the playground, the fit re-entry guard, the metadata-only validation
table — runs its real body.

Run:  python -m pytest tests/test_jlens_notebook_smoke.py
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("marimo", reason="notebook smoke test needs marimo")
torch = pytest.importorskip("torch")

JLENS_DIR = Path(__file__).resolve().parents[1] / "jacobian-lens"
sys.path.insert(0, str(JLENS_DIR))

D_MODEL, N_LAYERS, VOCAB = 8, 4, 300


class _Tokenizer:
    def __call__(self, text, **_):
        # Uncapped on purpose: the notebook compares this against the *truncated*
        # `encode()` length to decide whether to warn about truncation.
        return SimpleNamespace(input_ids=list(range(max(len(text.split()), 2))))

    def decode(self, ids, **_):
        index = int(ids[0]) if isinstance(ids, (list, tuple)) else int(ids)
        return f" tok{index}" if index % 5 else " ,"

    def __len__(self):
        # Deliberately != the unembedding width, which is the mismatch the
        # notebook's §1.4 note exists to point out.
        return VOCAB - 3


class _Model:
    def __init__(self):
        self.n_layers, self.d_model = N_LAYERS, D_MODEL
        self.layers = [torch.nn.Linear(D_MODEL, D_MODEL) for _ in range(N_LAYERS)]
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL)
        self._unembed = torch.nn.Linear(D_MODEL, VOCAB, bias=False)
        self.tokenizer = _Tokenizer()

    def encode(self, text, max_length=512):
        n = min(max(len(text.split()), 2), max_length)
        return torch.arange(n).unsqueeze(0) % VOCAB

    def forward(self, input_ids):
        # Run the stack so jlens' activation hooks on `self.layers` fire.
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return hidden

    def unembed(self, residual):
        return self._unembed(residual)


class _Form:
    """Stand-in for a submitted marimo form."""

    def __init__(self, value):
        self.value = value


def _make_lens(seed=0):
    import jlens

    generator = torch.Generator().manual_seed(seed)
    return jlens.JacobianLens(
        {i: torch.randn(D_MODEL, D_MODEL, generator=generator) for i in range(N_LAYERS)},
        n_prompts=1000,
        d_model=D_MODEL,
    )


@pytest.fixture(scope="module")
def notebook():
    import importlib
    import subprocess

    import jlens

    out = Path(tempfile.mkdtemp(prefix="jlens-smoke-"))
    # The state that used to crash the download cell.
    fitted = _make_lens(seed=3)
    fitted.fitted_for_model = "Qwen/Qwen3.5-4B"
    fitted.save(str(out / "jacobian_lens_n25.pt"))

    cwd = os.getcwd()
    os.chdir(JLENS_DIR)
    try:
        import CTP49906_jlens_molab as nb
    finally:
        os.chdir(cwd)

    base = dict(
        JLENS_DIR=JLENS_DIR, Path=Path, REPO_DIR=JLENS_DIR.parent, REPO_REF="local",
        importlib=importlib, subprocess=subprocess, sys=sys,
        OUTPUT_ROOT=out, device=torch.device("cpu"), jlens=jlens, os=os, torch=torch,
        hf_model=None, model=_Model(), tokenizer=_Tokenizer(), transformers=None,
        lens=_make_lens(),
        demo_layers=[0, 2], jlens_logits=None, logit_lens_out=None,
        model_logits=None, prompt_compare="Fact: the currency is the",
    )

    def run(**overrides):
        prev = os.getcwd()
        os.chdir(JLENS_DIR)
        try:
            outputs, defs = nb.app.run(defs={**base, **overrides})
        finally:
            os.chdir(prev)
        blob = "\n".join(_text(o) for o in outputs)
        return outputs, defs, blob

    return SimpleNamespace(run=run, out=out)


def _text(obj):
    if obj is None:
        return ""
    for attr in ("text", "_repr_html_"):
        try:
            value = getattr(obj, attr)
            rendered = value() if callable(value) else value
            if isinstance(rendered, str):
                return rendered
        except Exception:  # noqa: BLE001
            pass
    try:
        return str(obj)
    except Exception:  # noqa: BLE001
        return ""


def test_the_notebook_runs_with_a_fitted_lens_on_disk(notebook):
    # The regression: this raised NameError inside the A.2 download cell.
    outputs, defs, _ = notebook.run()
    assert not [o for o in outputs if "Error" in type(o).__name__]
    assert [p.name for p in defs["fitted_lens_files"]] == ["jacobian_lens_n25.pt"]


def test_the_vocabulary_width_comes_from_the_unembedding_not_the_tokenizer(notebook):
    # Ranks are computed against the unembedding width; using len(tokenizer)
    # would put the stated chance rank in the wrong place.
    _, defs, blob = notebook.run()
    assert defs["VOCAB_SIZE"] == VOCAB != len(_Tokenizer())
    assert "different number" in blob


def test_the_scrambled_control_really_scrambles(notebook):
    _, defs, _ = notebook.run(
        lens_form=_Form({"source": "scrambled", "fit_size": 0, "upload": []})
    )
    active, reference = defs["active_lens"], _make_lens()
    assert active.source_layers == reference.source_layers
    unchanged = [
        i for i in reference.source_layers
        if torch.equal(active.jacobians[i], reference.jacobians[i])
    ]
    assert not unchanged, unchanged
    assert "CONTROL" in defs["active_lens_label"]


def test_a_missing_student_fit_warns_instead_of_cancelling_the_section(notebook):
    # This used to `mo.stop`, which cancelled the form, the results and the slice
    # download together — with the recovery action 250 lines further down.
    _, defs, blob = notebook.run(
        lens_form=_Form({"source": "fitted", "fit_size": 999, "upload": []})
    )
    assert "No 999-prompt fit" in blob
    assert defs["active_lens"] is not None
    assert "playground_controls" in defs


def test_a_lens_fitted_for_another_model_is_refused(notebook):
    wrong = _make_lens(seed=2)
    wrong.fitted_for_model = "meta-llama/Llama-3-8B"
    path = notebook.out / "wrong.pt"
    wrong.save(str(path))

    class _Upload:
        name = "wrong.pt"
        contents = path.read_bytes()

    _, _, blob = notebook.run(
        lens_form=_Form({"source": "uploaded", "fit_size": 0, "upload": [_Upload()]})
    )
    assert "was fitted for" in blob and "Llama-3-8B" in blob


def test_resubmitting_a_size_that_already_exists_does_not_refit(notebook):
    # A form keeps its submitted value, so any upstream re-run re-enters this
    # cell. Without the artifact guard that silently restarts a 20-minute fit —
    # and the first thing it does is pip-install, so a fast clean return is the
    # assertion.
    _, defs, blob = notebook.run(fit_controls=_Form({"n_prompts": 25}))
    assert "already exists" in blob
    assert "Fitting a" not in blob


def test_the_validation_table_reports_recorded_model_identity(notebook):
    _, _, blob = notebook.run(fit_controls=_Form({"n_prompts": 25}))
    assert "jacobian_lens_n25.pt" in blob
    assert "Qwen/Qwen3.5-4B" in blob


PLAYGROUND = {
    "hypothesis": "J-lens wins early", "prompt": "Fact: the currency is the",
    "position_from_end": 1, "layers": [0, 2], "top_k": 5, "make_slice": False,
    "slice_stride": 2, "slice_window": 64, "mask_display": True,
}


def test_the_playground_produces_a_verdict_table(notebook):
    _, _, blob = notebook.run(playground_controls=_Form(PLAYGROUND))
    assert "Result and verdict" in blob
    assert "J-lens wins early" in blob


def test_the_prediction_gate_holds_even_when_validate_is_bypassed(notebook):
    # marimo runs a form's `validate=` only in the submit-button handler;
    # Ctrl/Cmd+Enter sets the value directly and skips it.
    _, _, blob = notebook.run(
        playground_controls=_Form({**PLAYGROUND, "hypothesis": "   "})
    )
    assert "Write a falsifiable prediction first" in blob


def test_a_truncated_prompt_says_so_and_names_both_lengths(notebook):
    long_prompt = " ".join(f"w{i}" for i in range(900))
    _, _, blob = notebook.run(
        playground_controls=_Form({**PLAYGROUND, "prompt": long_prompt})
    )
    assert "was truncated" in blob
    assert "900" in blob and "512" in blob


def test_layers_the_active_lens_cannot_fit_are_explained_not_crashed(notebook):
    import jlens

    narrow = jlens.JacobianLens(
        {i: torch.eye(D_MODEL) for i in (0, 1)}, n_prompts=7, d_model=D_MODEL
    )
    path = notebook.out / "narrow.pt"
    narrow.save(str(path))

    class _Upload:
        name = "narrow.pt"
        contents = path.read_bytes()

    lens_form = _Form({"source": "uploaded", "fit_size": 0, "upload": [_Upload()]})

    _, _, blob = notebook.run(
        lens_form=lens_form,
        playground_controls=_Form({**PLAYGROUND, "layers": [0, 2]}),
    )
    assert "Dropped layer" in blob and "Result and verdict" in blob

    _, _, blob = notebook.run(
        lens_form=lens_form,
        playground_controls=_Form({**PLAYGROUND, "layers": [2, 3]}),
    )
    assert "None of the selected layers are fitted" in blob


def test_the_slice_filename_records_the_settings_that_produced_it(notebook):
    import re

    _, _, blob = notebook.run(playground_controls=_Form({
        **PLAYGROUND, "make_slice": True, "slice_stride": 3,
        "slice_window": 32, "mask_display": False,
    }))
    names = re.findall(r"slice_\d+stride_\d+win_off\d+_\w+\.html", blob)
    assert names, "no settings-bearing slice filename in the output"
    assert "3stride" in names[0] and "32win" in names[0] and "raw" in names[0]
