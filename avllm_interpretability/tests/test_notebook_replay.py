"""Execute the notebook end-to-end in `USE_PRECOMPUTED` replay mode.

`test_notebook_graph.py` proves the dataflow graph is well-formed; it never runs
a cell body. That gap is not academic -- a sibling notebook shipped a cell that
raised `NameError` the moment a file existed on disk, and no static check saw it,
because the crash lived inside a list comprehension in a branch that is empty
until a student has done something.

Replay mode is the one path that needs no GPU and no model weights, and it still
executes the setup, the parameters, the logit-lens replay, the diversity plot,
the probe grid, both attention panels, the ledger, and every form constructor.

`app.run(defs=...)` *replaces* the cells that define the given names, so the two
overrides below skip the pip-install/git-clone cell and the replay switch
entirely: no network, no subprocess, no weights.

Run:  python -m pytest avllm_interpretability/tests/test_notebook_replay.py
"""
import sys
from pathlib import Path

import pytest

marimo = pytest.importorskip("marimo", reason="notebook smoke test needs marimo")
pytest.importorskip("qwen_omni_utils", reason="notebook imports qwen_omni_utils")
pytest.importorskip("wigglystuff", reason="notebook imports wigglystuff")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


@pytest.fixture(scope="module")
def replay():
    import importlib
    import os
    import subprocess

    cwd = os.getcwd()
    os.chdir(PROJECT)
    try:
        import CTP49906_avllm_molab as nb

        outputs, defs = nb.app.run(defs={
            # the setup cell: every name it defines, so it does not run at all
            "PROJECT_DIR": PROJECT,
            "Path": Path,
            "REPO_DIR": PROJECT.parent,
            "REPO_REF": "local-checkout",
            "importlib": importlib,
            "subprocess": subprocess,
            "sys": sys,
            # the replay switch
            "USE_PRECOMPUTED": True,
            "PRECOMPUTED_DIR": PROJECT / "precomputed",
        })
    finally:
        os.chdir(cwd)
    return outputs, defs


def _text(obj):
    if obj is None:
        return ""
    for attr in ("text", "_repr_html_"):
        try:
            value = getattr(obj, attr)
            rendered = value() if callable(value) else value
            if isinstance(rendered, str):
                return rendered
        except Exception:  # noqa: BLE001 — a renderer that fails is not the test
            pass
    try:
        return str(obj)
    except Exception:  # noqa: BLE001
        return ""


def test_every_cell_runs_without_raising(replay):
    outputs, defs = replay
    assert len(outputs) > 40, len(outputs)
    assert not [o for o in outputs if "Error" in type(o).__name__]


def test_replay_never_touches_a_gpu(replay):
    import torch

    _, defs = replay
    assert not torch.cuda.is_initialized()
    assert str(defs["DEVICE"]) == "cpu"


def test_the_attention_panels_are_baseline_knockout_and_delta(replay):
    # The defect this notebook shipped for a while: one panel, measured under the
    # knockout, captioned as if it described normal behaviour.
    outputs, _ = replay
    figures = [o for o in outputs if type(o).__module__.startswith("matplotlib")]
    heatmap = next(
        (f for f in figures if any(ax.get_title() == "Knockout run" for ax in f.axes)),
        None,
    )
    assert heatmap is not None, "no attention heatmap figure"
    titles = {ax.get_title() for ax in heatmap.axes}
    assert {"Baseline (no knockout)", "Knockout run", "Δ = knockout − baseline"} <= titles


def test_the_two_mass_panels_share_one_color_scale(replay):
    # Independently normalised panels look alike while describing different
    # numbers -- the same defect as a per-run color scale on the Δ strip, in the
    # one place whose entire purpose is a side-by-side comparison.
    outputs, _ = replay
    figures = [o for o in outputs if type(o).__module__.startswith("matplotlib")]
    heatmap = next(f for f in figures if any(ax.get_title() == "Knockout run" for ax in f.axes))
    clims = [
        ax.images[0].get_clim()
        for ax in heatmap.axes
        if ax.get_title() in ("Baseline (no knockout)", "Knockout run") and ax.images
    ]
    assert len(clims) == 2 and clims[0] == clims[1], clims


def test_the_blocked_modality_differs_between_baseline_and_knockout(replay):
    _, defs = replay
    baseline, knockout = defs["baseline_attention_summary"], defs["attention_summary"]
    video = baseline[1].index("video")
    assert baseline[2][0][video] != knockout[2][0][video]
    assert knockout[2][0][video] == 0.0  # blocked by construction


def test_the_probe_grid_is_built_from_the_replayed_csv(replay):
    _, defs = replay
    summary = {row["name"]: row for row in defs["probe_summary"]}
    assert len(summary) == 36
    layer34 = summary["Layer_34"]
    assert (layer34["unique"], layer34["junk"], layer34["matches_final"]) == (6, 245, 92)


def test_the_clip_chooser_resolves_all_three_branches(replay):
    _, defs = replay
    resolve = defs["resolve_clip"]

    default_clip, is_control, err = resolve("Default clip", [])
    assert err is None and not is_control and default_clip.name == "02321.mp4"

    silent, is_control, err = resolve("Silent control", [])
    assert err is None and is_control and silent.name == "02321_silent.mp4"

    # An empty upload must be an ERROR, not a silent fall back to the real clip:
    # that fallback is how a "silent control" run became a duplicate of the
    # experiment it was meant to falsify.
    path, _, err = resolve("Upload", [])
    assert path is None and err


def test_an_upload_cannot_escape_the_results_directory(replay):
    _, defs = replay
    resolve = defs["resolve_clip"]
    blob = (PROJECT / "assets" / "02321_silent.mp4").read_bytes()

    class _Upload:
        name = "../../../../tmp/pwned.mp4"
        contents = blob

    path, _, err = resolve("Upload", [_Upload()])
    assert err is None, err
    try:
        assert PROJECT.resolve() in path.resolve().parents
        assert "/" not in path.name and ".." not in path.name.split("_", 2)[-1]
    finally:
        if path and path.exists():
            path.unlink()


@pytest.mark.parametrize("form_name", ["band_controls", "ko_controls", "tf_controls"])
def test_the_forms_start_unsubmitted(replay, form_name):
    _, defs = replay
    assert defs[form_name].value is None
