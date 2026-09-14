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
        "prediction": "가설", "video": None, "prompt_preset": ["직접 작성"], "prompt": "p",
        "ko_enable": False, "ko_source": ["audio"], "ko_target": ["video"],
        "ko_layers": (0, 36), "ko_rules_text": "", "compare": True,
        "target": ["audio"], "layers": ["전체"],
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
    # The view selects the shared renderer's Korean locale. Fixed chrome is
    # translated while student text, including words that resemble chrome,
    # remains intact and HTML-escaped.
    import re
    from html import escape

    from src.run_ledger import render_ledger_html, run_record

    _, defs = replay
    prediction = 'supported · unresolved · control <b>직접 쓴 주장</b>'
    record = run_record(
        kind="band_sweep", condition="generated→video [0,12)",
        metric_name="caption_similarity", metric_value=0.83, metric_unit="ratio",
        config={"clip": "02321.mp4"}, prediction=prediction, is_control=False,
    )
    english = render_ledger_html([record])
    assert ">prediction</th>" in english and ">save status</th>" in english

    localized = render_ledger_html([record], lang="ko")
    emitted = set(re.findall(r"<th[^>]*>([^<]*)</th>", localized))
    assert emitted and all(any("가" <= ch <= "힣" for ch in heading) for heading in emitted)

    # Seed state instead of depending on a local, gitignored log file.
    defs["set_runs"](lambda _prev: [record])
    html = defs["ledger_view"]().text
    assert set(re.findall(r"<th[^>]*>([^<]*)</th>", html)) == emitted
    assert ">미판정</span>" in html
    assert escape(prediction, quote=True) in html
    assert "<b>직접 쓴 주장</b>" not in html
    assert record["prediction"] == prediction


def test_the_ledger_summary_and_verdicts_reach_the_student_in_korean(replay):
    # A declared control from another experiment does not imply matched settings.
    # The view must communicate that distinction and localize the verdict value.
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
    # The default locale remains usable by the English notebook.
    assert "claim(s) with NO control" in raw
    assert 'title="rival: 캡션이 짧아졌을 뿐">supported</span>' in raw

    defs["set_runs"](lambda _prev: runs)
    html = defs["ledger_view"]().text
    assert "설정이 일치하는 대조군 없는 실행 1개" in html
    assert "대조군과 설정이 일치하는 실행 0개" in html
    assert "with NO control" not in html
    assert ">지지됨</span>" in html and ">supported</span>" not in html
    # The control chip is display-only and localized; `kind` and `metric` are the
    # keys that also appear in lab_log.jsonl, and deliberately stay English so a
    # student can match a table row against the file.
    assert ';font-weight:600">대조군으로 지정</span>' in html
    assert "band_sweep" in html and "caption_similarity" in html


def test_no_form_asks_for_a_hypothesis_any_more(replay):
    # The lab was retuned for tweaking over typing: the prediction text areas and
    # the gates that refused a run without one are gone. Assert both halves --
    # the widget is absent, and a submit carrying no prediction is accepted.
    _, defs = replay
    payloads = {
        "band_controls": {"target": ["video"], "layers": (0, 12), "null_band": False},
        "ko_controls": {"clip": defs["CLIP_DEFAULT"], "video": None,
                        "prompt_preset": ["직접 작성"], "prompt": "p", "ko_enable": False, "ko_source": ["audio"],
                        "ko_target": ["video"], "ko_layers": (0, 36),
                        "ko_rules_text": "", "compare": True},
        "tf_controls": {"clip": defs["CLIP_DEFAULT"], "video": None,
                        "prompt_preset": ["직접 작성"], "prompt": "p",
                        "target": ["audio"], "layers": ["전체"]},
    }
    for name, payload in payloads.items():
        form = defs[name]
        assert "prediction" not in form.element.elements, name
        assert form.validate(payload) is None, (name, form.validate(payload))


def test_the_ledger_drops_the_prediction_column_when_nothing_carries_one(replay):
    # Without this, every row would render the red "none written" alarm -- marking
    # as a defect the field the notebook deliberately stopped collecting.
    import re

    from src.run_ledger import render_ledger_html, run_record

    def rec(pred):
        return run_record(
            kind="diversity", condition="audio→video [0,36)",
            metric_name="mean_delta_diversity", metric_value=-2.1,
            metric_unit="unique preds", config={"clip": "02321.mp4"},
            prediction=pred, is_control=False,
        )

    without = render_ledger_html([rec("")])
    assert "none written" not in without
    without_headers = set(re.findall(r"<th[^>]*>([^<]*)</th>", without))
    assert "prediction" not in without_headers

    # ...and keeps it the moment a run has one, so the English notebook and any
    # ledger reloaded from an older lab_log.jsonl are unaffected.
    with_pred = render_ledger_html([rec("앞쪽 레이어가 더 크게 바꿀 것이다")])
    with_headers = set(re.findall(r"<th[^>]*>([^<]*)</th>", with_pred))
    assert "prediction" in with_headers
    assert with_headers - {"prediction"} == without_headers


@pytest.mark.parametrize("form_name", ["band_controls", "ko_controls", "tf_controls"])
def test_the_forms_start_unsubmitted(replay, form_name):
    _, defs = replay
    assert defs[form_name].value is None


@pytest.mark.parametrize("form_name", ["ko_controls", "tf_controls"])
def test_classroom_forms_validate_the_selected_prompt_with_fixed_frames(replay, form_name):
    _, defs = replay
    form = defs[form_name]
    valid = {"clip": defs["CLIP_DEFAULT"], "video": None,
             "prompt_preset": ["직접 작성"], "prompt": "소리를 설명해 주세요", "ko_enable": False,
             "target": ["audio"], "layers": ["전체"], "max_new_tokens": 32}
    assert form.validate(valid) is None
    assert "nframes" not in form.element.elements
    assert form.validate({**valid, "prompt_preset": ["소리 설명"], "prompt": "  "}) is None
    assert form.validate({**valid, "prompt": "  "}) is not None


def test_tf_caption_limit_is_a_submitted_form_setting(replay):
    _, defs = replay
    form = defs["tf_controls"]
    assert "max_new_tokens" in form.element.elements
    valid = {"clip": defs["CLIP_DEFAULT"], "video": None,
             "prompt_preset": ["소리 설명"], "prompt": "", "target": ["audio"],
             "layers": ["전체"], "max_new_tokens": 64}
    assert form.validate(valid) is None
    assert form.validate({**valid, "max_new_tokens": 1024}) is not None


def test_korean_figures_render_without_missing_glyphs(replay):
    import io
    import warnings

    outputs, _ = replay
    figures = [o for o in outputs if type(o).__module__.startswith("matplotlib")]
    assert figures
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for figure in figures:
            figure.savefig(io.BytesIO(), format="png")
    assert not [str(w.message) for w in caught
                if "Glyph" in str(w.message) and "missing" in str(w.message)]


def test_export_copy_previews_preserve_literal_fences_and_html(replay):
    """Execute the notebook's export cell with hostile-looking caption text."""
    import ast
    from html.parser import HTMLParser

    from src.run_ledger import append_run, build_evidence_json, build_worksheet_md, run_record

    _, defs = replay
    caption = '```caption```\n<script>alert("caption")</script><img src="x">'
    runs = append_run([], run_record(
        kind="teacher_forcing", condition="answer→audio", metric_name="delta_per_token",
        metric_value=-0.2, metric_unit="nats/token", config={"clip": "02321.mp4"},
        prediction=caption, extra={"caption_text": caption, "delta": [-0.2]},
    ))
    # Isolate only this cheap display cell; its actual mo.Html/accordion stack
    # still renders, without executing any live model branches or changing state.
    tree = ast.parse(NOTEBOOK.read_text())
    cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and {arg.arg for arg in node.args.args}
                == {"get_runs", "mo", "run_provenance", "worksheet_md"})
    cell.decorator_list = []
    cell.name = "render_export_cell"
    for index, statement in enumerate(cell.body):
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            cell.body[index] = ast.Return(value=statement.value)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[])),
                 str(NOTEBOOK), "exec"), namespace)
    rendered = namespace["render_export_cell"](
        lambda: runs, defs["mo"], defs["run_provenance"], build_worksheet_md,
    ).text

    class PreText(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.blocks, self.tags, self.inside = [], [], False

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)
            if tag == "pre":
                self.blocks.append("")
                self.inside = True

        def handle_endtag(self, tag):
            if tag == "pre":
                self.inside = False

        def handle_data(self, data):
            if self.inside:
                self.blocks[-1] += data

    parsed = PreText()
    parsed.feed(rendered)
    assert parsed.blocks == [
        build_worksheet_md(runs),
        build_evidence_json(runs, provenance=defs["run_provenance"]),
    ]
    assert "script" not in parsed.tags and "img" not in parsed.tags


@pytest.mark.parametrize("changed_name", ["02321.mp4", "02321_silent.mp4"])
def test_builtin_control_pair_requires_shipped_content_at_the_known_paths(replay, changed_name):
    import ast
    import hashlib
    import importlib.metadata
    import platform

    from src.run_ledger import matched_control_ids, run_record

    _, defs = replay
    actual_notebook_sha256 = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    assert defs["run_provenance"]["notebook_sha256"] == actual_notebook_sha256
    expected_packages = {
        name: importlib.metadata.version(name) for name in
        ("torch", "torchvision", "transformers", "qwen-omni-utils", "marimo",
         "numpy", "matplotlib", "av", "wigglystuff", "anywidget", "accelerate", "librosa", "audioread")
    }
    assert defs["run_provenance"]["packages"] == expected_packages
    tree = ast.parse(NOTEBOOK.read_text())
    cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and any(isinstance(child, ast.FunctionDef) and child.name == "experiment_config"
                        for child in node.body))
    cell.decorator_list = []
    cell.name = "build_experiment_config"
    namespace = {"__file__": str(NOTEBOOK)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[])),
                 str(NOTEBOOK), "exec"), namespace)
    config_for, _ = namespace["build_experiment_config"](
        defs["MODEL_PATH"], defs["MODEL_REVISION"], PROJECT,
    )
    settings = {"nframes": 8, "prompt": "소리를 설명해 주세요", "target": "audio",
                "start": 0, "end": 36, "max_new_tokens": 32}
    paths = [PROJECT / "assets" / name for name in ("02321.mp4", "02321_silent.mp4")]
    configs = [config_for(path, **settings) for path in paths]
    for config in configs:
        assert config["notebook_sha256"] == actual_notebook_sha256
        assert config["runtime_packages"] == expected_packages
        assert config["model_id"] == defs["MODEL_PATH"]
        assert config["model_revision"] == defs["MODEL_REVISION"]
        assert config["repo_revision"] == defs["run_provenance"]["repo_revision"]
        assert config["python"] == platform.python_version()
        assert all(config[key] == value for key, value in settings.items())

    def record(config, is_control):
        return run_record(kind="teacher_forcing", condition="answer→audio [0,36)",
                          metric_name="delta_per_token", metric_value=-0.2,
                          metric_unit="nats/token", config=config, is_control=is_control)

    original, control = record(configs[0], False), record(configs[1], True)
    assert matched_control_ids(original, [original, control]) == [control["run_id"]]

    class ReplacedClip:
        # Same resolved location, different content; real repository assets
        # remain untouched while exercising the actual notebook config function.
        name = changed_name

        def resolve(self):
            return (PROJECT / "assets" / self.name).resolve()

        def read_bytes(self):
            return self.resolve().read_bytes() + b"changed content"

    altered = config_for(ReplacedClip(), **settings)
    assert "comparison_key" not in altered
    if changed_name == "02321.mp4":
        assert matched_control_ids(record(altered, False), [control]) == []
    else:
        assert matched_control_ids(original, [record(altered, True)]) == []


@pytest.mark.parametrize("label,prompt", [
    ("소리 설명", "영상에서 들리는 소리를 설명해 주세요"),
    ("보이는 내용 설명", "영상에서 보이는 내용을 설명해 주세요"),
    ("시청각 함께 설명", "영상에서 보이는 것과 들리는 소리를 설명해 주세요"),
])
def test_prompt_presets_resolve_frontend_labels_and_converted_values(replay, label, prompt):
    _, defs = replay
    choices = defs["PROMPT_CHOICES"]
    resolve = defs["resolve_classroom_prompt"]
    assert choices[label] == prompt
    # Frontend validation receives a singleton label list; the experiment gets
    # the converted Python value. Both must execute the same question, regardless
    # of an unrelated custom-text field left over from another selection.
    assert resolve([label], "이 텍스트는 선택하지 않았습니다") == prompt
    assert resolve(label, "") == prompt
    assert resolve(choices[label], "") == prompt


def test_custom_prompt_is_trimmed_and_blank_or_unknown_selection_is_rejected(replay):
    _, defs = replay
    resolve = defs["resolve_classroom_prompt"]
    assert defs["PROMPT_CHOICES"]["직접 작성"] == "custom"
    for selection in (["직접 작성"], "직접 작성", "custom"):
        assert resolve(selection, "  보이는 인물을 묘사해 주세요  ") == "보이는 인물을 묘사해 주세요"
        with pytest.raises(ValueError):
            resolve(selection, " \n ")
    for selection in ([], None, "unknown", ["소리 설명", "직접 작성"]):
        with pytest.raises(ValueError):
            resolve(selection, "질문")


@pytest.mark.parametrize("label,value,expected", [
    ("전체", "all", (0, 36)), ("초반", "early", (0, 12)),
    ("중반", "middle", (12, 24)), ("후반", "late", (24, 36)),
])
def test_layer_presets_resolve_the_same_exclusive_interval_in_form_and_execution(
    replay, label, value, expected,
):
    _, defs = replay
    choices = defs["classroom_layer_choices"](36)
    resolve = defs["resolve_classroom_layers"]
    assert choices[label] == value
    assert resolve([label], 36) == expected
    assert resolve(label, 36) == expected
    assert resolve(value, 36) == expected


def test_layer_presets_reject_empty_or_unknown_choices(replay):
    _, defs = replay
    resolve = defs["resolve_classroom_layers"]
    for selection in ([], None, "unknown", ["초반", "후반"], (0, 0)):
        with pytest.raises(ValueError):
            resolve(selection, 36)


@pytest.mark.parametrize("form_name", ["tf_controls", "ko_controls"])
def test_experiment_forms_offer_prompt_tasks_and_preserve_custom_uploads(replay, form_name):
    _, defs = replay
    elements = defs[form_name].element.elements
    assert "nframes" not in elements
    assert elements["prompt_preset"].options == defs["PROMPT_CHOICES"]
    assert {"clip", "video", "prompt"} <= elements.keys()
    assert elements["clip"].options[defs["CLIP_UPLOAD"]] == "Upload"
    assert defs["NFRAMES"] == 8
    if form_name == "tf_controls":
        assert elements["layers"].options == defs["classroom_layer_choices"](36)
        assert set(elements["target"].options.values()) >= {"audio", "video"}
    else:
        assert {"ko_enable", "ko_source", "ko_target", "ko_layers", "ko_rules_text"} <= elements.keys()


def _execute_experiment_setup(defs, form_name, payload):
    """Run the notebook's actual submit-to-validation statements without a GPU.

    Finish immediately after setup validation (and, for TF, after building its
    knockout rule), before either cell encodes a clip or invokes a model.
    """
    import ast
    from types import SimpleNamespace

    tree = ast.parse(NOTEBOOK.read_text())
    cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and form_name in {arg.arg for arg in node.args.args}
                and "validate_experiment" in {arg.arg for arg in node.args.args})
    cell.decorator_list = []
    cell.name = "submit_setup"
    for index, stmt in enumerate(cell.body):
        if form_name == "tf_controls":
            stop_here = isinstance(stmt, ast.FunctionDef) and stmt.name == "_tf_prep"
        else:
            stop_here = (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                         and isinstance(stmt.value.func, ast.Name)
                         and stmt.value.func.id == "validate_experiment")
            if stop_here:
                index += 1
        if stop_here:
            cell.body = cell.body[:index] + [ast.Return(value=ast.Call(
                func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))]
            break
    else:
        raise AssertionError("Cannot locate pre-model experiment setup boundary")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[])),
                 str(NOTEBOOK), "exec"), namespace)
    args = {arg.arg: defs[arg.arg] for arg in cell.args.args}
    args.update({form_name: SimpleNamespace(value=payload), "USE_PRECOMPUTED": False})
    return namespace["submit_setup"](**args)


@pytest.mark.parametrize("form_name", ["tf_controls", "ko_controls"])
def test_submitted_experiment_uses_fixed_frames_and_the_selected_prompt(replay, form_name):
    _, defs = replay
    payload = {"clip": "Default clip", "video": [], "nframes": 2,
               "prompt_preset": "영상에서 보이는 내용을 설명해 주세요",
               "prompt": "무시되어야 하는 이전 질문", "target": "audio",
               "layers": "middle", "max_new_tokens": 32}
    actual = _execute_experiment_setup(defs, form_name, payload)
    prefix = "_tf_" if form_name == "tf_controls" else "_"
    assert actual[prefix + "nframes"] == 8
    assert actual[prefix + "prompt"] == "영상에서 보이는 내용을 설명해 주세요"
    if form_name == "tf_controls":
        assert actual["_tf_rules"] == [("answer", "audio", 12, 24)]


def test_core_answer_measurement_precedes_advanced_diversity_in_the_notebook():
    import ast
    tree = ast.parse(NOTEBOOK.read_text())
    locations = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                if child.id in ("tf_controls", "ko_controls"):
                    locations[child.id] = node.lineno
    assert locations["tf_controls"] < locations["ko_controls"]


@pytest.mark.parametrize("n_layers", [0, 1, 2, -1])
def test_layer_presets_refuse_models_too_small_for_three_nonempty_bands(replay, n_layers):
    _, defs = replay
    with pytest.raises(ValueError):
        defs["classroom_layer_choices"](n_layers)
    with pytest.raises(ValueError):
        defs["resolve_classroom_layers"]("all", n_layers)


@pytest.mark.parametrize("invalid_layers", [[], ["없는 구간"], "unknown"])
def test_tf_form_reports_invalid_layer_selection_instead_of_submitting(replay, invalid_layers):
    _, defs = replay
    payload = {"clip": defs["CLIP_DEFAULT"], "video": None,
               "prompt_preset": ["소리 설명"], "prompt": "", "target": ["audio"],
               "layers": invalid_layers, "max_new_tokens": 32}
    assert defs["tf_controls"].validate(payload) is not None



def test_core_tf_results_have_a_run_ledger_before_advanced_activity_controls():
    """Students need the first four run IDs before moving into advanced work.

    Require a cheap, independent ledger display immediately in the core lesson
    region, after all TF result displays and before the diversity controls.
    """
    import ast

    tree = ast.parse(NOTEBOOK.read_text())
    cells = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    tf_result_cells = [node for node in cells
                       if "tf_result" in {arg.arg for arg in node.args.args}]
    ko_form = next(node for node in cells if any(
        isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        and child.id == "ko_controls" for child in ast.walk(node)))
    last_tf_display_line = max(node.end_lineno for node in tf_result_cells)
    ledger_displays = [node for node in cells if any(
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "ledger_view" for stmt in node.body)]
    core_ledgers = [node for node in ledger_displays
                    if last_tf_display_line < node.lineno < ko_form.lineno]
    assert core_ledgers, "Core TF runs have no nearby ledger/IDs before advanced activity 8"
    assert any({arg.arg for arg in node.args.args} <= {"ledger_view", "mo"}
               for node in core_ledgers), "The core ledger must remain an independent display"
