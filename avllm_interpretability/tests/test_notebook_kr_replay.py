"""Execute the Korean notebook end-to-end in `USE_PRECOMPUTED` replay mode.

`test_notebook_replay.py` does this for `CTP49906_avllm_molab.py`, but its
assertions name the English panel titles and that pack's probe numbers, so the
Korean twin needs its own. Beyond re-checking that every cell runs, this file
guards the two things localization can quietly break:

* the clip radio and the verdict dropdown now carry **Korean labels over English
  values**, because `resolve_clip` and `run_ledger.VERDICTS` are shared with the
  English notebook. A label that stops mapping to a known value fails open — the
  form still submits, and the run is silently mislabelled.
* `PRECOMPUTED_DIR` points at `precomputed_kr/`. Pointed at the English pack, the
  prompts in `meta.json` no longer match and replay must refuse rather than
  relabel a run that used different prompts.

Run:  python -m pytest avllm_interpretability/tests/test_notebook_kr_replay.py
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

NOTEBOOK = PROJECT / "CTP49906_avllm_molab_kr.py"
PACK = PROJECT / "precomputed_kr"

LOGIT_PROMPT = "영상에서 들리는 소리를 설명해 주세요"
ATTENTION_PROMPT = "영상에서 보이는 것과 들리는 소리를 설명해 주세요"


@pytest.fixture(scope="module")
def replay():
    import importlib
    import os
    import subprocess

    from marimo._ast.load import load_app

    cwd = os.getcwd()
    os.chdir(PROJECT)
    try:
        app = load_app(str(NOTEBOOK))
        outputs, defs = app.run(defs={
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
            "PRECOMPUTED_DIR": PACK,
        })
    finally:
        os.chdir(cwd)
    return outputs, defs


def test_every_cell_runs_without_raising(replay):
    outputs, _ = replay
    assert len(outputs) > 40, len(outputs)
    assert not [o for o in outputs if "Error" in type(o).__name__]


def test_replay_never_touches_a_gpu(replay):
    import torch

    _, defs = replay
    assert not torch.cuda.is_initialized()
    assert str(defs["DEVICE"]) == "cpu"


def test_the_prompts_are_korean(replay):
    _, defs = replay
    assert defs["LOGIT_PROMPT"] == LOGIT_PROMPT
    assert defs["ATTENTION_PROMPT"] == ATTENTION_PROMPT


def test_the_korean_pack_is_the_one_being_replayed(replay):
    # Not the English pack: same clip and rules, different prompts.
    import json

    _, defs = replay
    meta = json.loads((PACK / "meta.json").read_text(encoding="utf-8"))
    assert meta["logit_prompt"] == LOGIT_PROMPT
    assert meta["attention_prompt"] == ATTENTION_PROMPT
    assert defs["PRECOMPUTED_DIR"].name == "precomputed_kr"


def test_replay_refuses_the_english_pack():
    # The guard that stops a Korean run being labelled with English-prompt
    # artifacts. `validate_precompute_meta` must raise, not warn.
    import json

    from src.precompute import validate_precompute_meta

    english = json.loads((PROJECT / "precomputed" / "meta.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        validate_precompute_meta(english, logit_prompt=LOGIT_PROMPT)


def test_the_attention_panels_are_baseline_knockout_and_delta(replay):
    outputs, _ = replay
    figures = [o for o in outputs if type(o).__module__.startswith("matplotlib")]
    heatmap = next(
        (f for f in figures if any(ax.get_title() == "녹아웃 실행" for ax in f.axes)),
        None,
    )
    assert heatmap is not None, "no attention heatmap figure"
    titles = {ax.get_title() for ax in heatmap.axes}
    assert {"기준선 (녹아웃 없음)", "녹아웃 실행", "Δ = 녹아웃 − 기준선"} <= titles


def test_the_two_mass_panels_share_one_color_scale(replay):
    outputs, _ = replay
    figures = [o for o in outputs if type(o).__module__.startswith("matplotlib")]
    heatmap = next(f for f in figures if any(ax.get_title() == "녹아웃 실행" for ax in f.axes))
    clims = [
        ax.images[0].get_clim()
        for ax in heatmap.axes
        if ax.get_title() in ("기준선 (녹아웃 없음)", "녹아웃 실행") and ax.images
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
    summary = defs["probe_summary"]
    assert len(summary) == 36
    # The probe is uncalibrated at audio positions -- that degeneracy is the
    # week's result, so assert it is visible rather than pinning exact counts.
    assert any(row["junk"] > 0 for row in summary)


def test_korean_clip_labels_map_onto_the_values_resolve_clip_knows(replay):
    # The regression this file exists for: the radio shows Korean, submits
    # English. A label mapped to anything `resolve_clip` does not branch on would
    # silently resolve to the "Upload with no file" error -- or worse, succeed.
    _, defs = replay
    choices, resolve = defs["CLIP_CHOICES"], defs["resolve_clip"]

    assert set(choices.values()) == {"Default clip", "Silent control", "Upload"}
    assert defs["CLIP_DEFAULT"] in choices
    assert choices[defs["CLIP_UPLOAD"]] == "Upload"

    default_clip, is_control, err = resolve(choices[defs["CLIP_DEFAULT"]], [])
    assert err is None and not is_control and default_clip.name == "02321.mp4"

    silent, is_control, err = resolve(choices["무음 대조군"], [])
    assert err is None and is_control and silent.name == "02321_silent.mp4"

    path, _, err = resolve(choices[defs["CLIP_UPLOAD"]], [])
    assert path is None and err  # an empty upload is an error, never a fallback


def test_the_upload_gate_fires_for_the_korean_label_and_the_english_value(replay):
    # `validate` sees the *frontend* value, which for a radio is the label. A
    # check written against the English value alone never matches and fails open.
    _, defs = replay
    base = {
        "prediction": "가설", "video": None, "nframes": 8, "prompt": "p",
        "ko_enable": False, "ko_source": ["audio"], "ko_target": ["video"],
        "ko_layers": (0, 36), "ko_rules_text": "", "compare": True,
        "target": ["audio"], "layers": (0, 36),
    }
    for form in ("ko_controls", "tf_controls"):
        validate = defs[form].validate
        assert validate({**base, "clip": defs["CLIP_DEFAULT"]}) is None
        assert validate({**base, "clip": defs["CLIP_UPLOAD"]}) is not None
        assert validate({**base, "clip": "Upload"}) is not None
        assert validate({**base, "clip": defs["CLIP_UPLOAD"], "video": [object()]}) is None


def test_the_verdict_dropdown_submits_values_the_ledger_accepts(replay):
    # Korean labels, English values -- `apply_verdict` is a silent no-op on an
    # unknown verdict, so a translated *value* would drop the verdict entirely.
    from src.run_ledger import VERDICTS

    _, defs = replay
    options = defs["verdict_form"].element.elements["verdict"].options
    assert set(options.values()) == set(VERDICTS)
    assert all(any("가" <= ch <= "힣" for ch in label) for label in options)


def test_the_ledger_chrome_is_localized_and_still_matches_the_renderer(replay):
    # `ledger_view` rewrites `render_ledger_html`'s fixed English, because that
    # renderer is shared with the English notebook. If its vocabulary drifts, the
    # rewrite silently stops applying -- so assert the mapping still covers every
    # header the renderer actually emits.
    import re

    from src.run_ledger import render_ledger_html, run_record

    _, defs = replay
    record = run_record(
        kind="band_sweep", condition="generated→video [0,12)",
        metric_name="caption_similarity", metric_value=0.83, metric_unit="ratio",
        config={"clip": "02321.mp4"}, prediction="가설", is_control=False,
    )
    raw = render_ledger_html([record])
    emitted = set(re.findall(r"<th[^>]*>([^<]*)</th>", raw))
    assert emitted, "renderer emitted no <th> row"
    assert emitted <= set(defs["LEDGER_HEADINGS"]), (
        f"untranslated ledger headers: {emitted - set(defs['LEDGER_HEADINGS'])}"
    )
    assert '<span style="opacity:0.55">unresolved</span>' in raw

    # And the view actually applies it. Seed the state explicitly: `ledger_view`
    # reads whatever `load_log` found, and notebook_results/ is gitignored -- so
    # on a fresh clone the ledger is empty and there are no headers to assert on.
    defs["set_runs"](lambda _prev: [record])
    html = defs["ledger_view"]().text
    assert "판정 없음" in html
    for korean in ("종류", "조건", "지표", "대조군", "가설", "판정"):
        assert f">{korean}</th>" in html, korean


def test_the_ledger_summary_and_verdicts_reach_the_student_in_korean(replay):
    # The "N claims with NO control" clause is the number the ledger section tells
    # students to drive to zero, and the verdict cell is what they change it with.
    # Both are rendered by shared `src/` code, so both are rewritten in the view.
    from src.run_ledger import apply_verdict, run_record

    from src.run_ledger import render_ledger_html

    _, defs = replay
    runs = [
        run_record(kind="band_sweep", condition="generated→video [0,12)",
                   metric_name="caption_similarity", metric_value=0.83,
                   metric_unit="ratio", config={"clip": "02321.mp4"},
                   prediction="가설", is_control=False),
        run_record(kind="diversity", condition="audio→video [0,36)",
                   metric_name="mean_delta_diversity", metric_value=-2.1,
                   metric_unit="unique preds", config={"clip": "02321_silent.mp4"},
                   prediction="가설", is_control=True),
    ]
    runs = apply_verdict(runs, runs[0]["run_id"], "supported", rival="캡션이 짧아졌을 뿐")

    raw = render_ledger_html(runs)
    # The renderer still emits the English this view is responsible for rewriting;
    # if it stops, these anchors fail rather than the rewrite silently no-opping.
    assert "claim(s) with NO control" in raw
    assert 'title="rival: 캡션이 짧아졌을 뿐">supported</span>' in raw

    defs["set_runs"](lambda _prev: runs)
    html = defs["ledger_view"]().text
    assert "대조군 없는 주장" in html and "with NO control" not in html
    assert ">지지됨</span>" in html and ">supported</span>" not in html
    assert set(defs["LEDGER_VERDICTS"]) == {"supported", "refuted", "untested"}
    # The control chip is display-only and localized; `kind` and `metric` are the
    # keys that also appear in lab_log.jsonl, and deliberately stay English so a
    # student can match a table row against the file.
    assert ';font-weight:600">대조군</span>' in html
    assert "band_sweep" in html and "caption_similarity" in html


@pytest.mark.parametrize("form_name", ["band_controls", "ko_controls", "tf_controls"])
def test_the_forms_start_unsubmitted(replay, form_name):
    _, defs = replay
    assert defs[form_name].value is None
