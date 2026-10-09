"""Native Molab HTML delivery contracts; these do not claim live GPU QA."""

from __future__ import annotations

import ast
import copy
import inspect
import re
import subprocess
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 helper-only CI
    tomllib = None


PROJECT = Path(__file__).resolve().parents[1]
CANONICAL = PROJECT / "CTP49906_avllm_molab_kr.py"
COMPANION = PROJECT / "CTP49906_avllm_molab_html_kr.py"
GENERATOR = PROJECT / "scripts" / "build_molab_html.py"
FORM_CONTROLS = {
    "band_form": "band_controls",
    "diversity_form": "ko_controls",
    "tf_form": "tf_controls",
}
FORM_ANCHORS = {"band_form": "ctp-band", "diversity_form": "ctp-diversity", "tf_form": "ctp-tf"}
DELIVERY_CELLS = {"studio_files", "lab_intro", "molab_setup_note", "studio_status"}
NEW_CELLS = {"molab_html_navigation", "molab_html_evidence_anchor"}


def _tree(path=COMPANION):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _cells(tree):
    result = defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name].append(node)
    return result


def _cell(name, path=COMPANION):
    matches = _cells(_tree(path))[name]
    assert len(matches) == 1, name
    return matches[0]


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _calls(cell, name):
    return [node for node in ast.walk(cell) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == name]


def _generation(*args):
    return subprocess.run([sys.executable, str(GENERATOR), *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_tracked_companion_is_current_and_check_changes_neither_notebook():
    before = {path: path.read_bytes() for path in (CANONICAL, COMPANION)}
    result = _generation("--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path: path.read_bytes() for path in before} == before


def test_generation_preserves_source_and_detects_stale_output(tmp_path):
    source, output = tmp_path / "lesson.py", tmp_path / "lesson_html.py"
    source_bytes = CANONICAL.read_bytes().replace(b"NFRAMES = 8", b"NFRAMES = 6")
    source.write_bytes(source_bytes)
    result = _generation("--source", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert source.read_bytes() == source_bytes
    assert b"NFRAMES = 6" in output.read_bytes()
    assert _generation("--source", source, "--output", output, "--check").returncode == 0
    stale = output.read_bytes() + b"\n# stale generated output\n"
    output.write_bytes(stale)
    assert _generation("--source", source, "--output", output, "--check").returncode != 0
    assert output.read_bytes() == stale
    assert source.read_bytes() == source_bytes


def test_all_scientific_and_result_cells_are_preserved():
    original, native = _cells(_tree(CANONICAL)), _cells(_tree())
    assert set(native) == set(original) | NEW_CELLS
    for name, cells in original.items():
        assert len(native[name]) == len(cells), name
        for before, after in zip(cells, native[name]):
            if name not in DELIVERY_CELLS:
                assert _dump(before.args) == _dump(after.args), name
            if name not in DELIVERY_CELLS | FORM_CONTROLS.keys():
                assert [_dump(node) for node in before.body] == [
                    _dump(node) for node in after.body
                ], f"Scientific cell changed: {name} at canonical line {before.lineno}"


def _normalized_form(cell):
    cell = copy.deepcopy(cell)
    cell.decorator_list = []
    # The final output may place a unique navigation anchor beside the form.
    # Keep computation, controls, validation and submit options under comparison.
    if isinstance(cell.body[-2], ast.Expr):
        cell.body[-2] = ast.Expr(value=ast.Name(id=FORM_CONTROLS[cell.name], ctx=ast.Load()))
    for node in ast.walk(cell):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"_template", "_tf_template"}
            for target in node.targets
        ):
            node.value = ast.Constant(value="<presentation template>")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "batch":
            node.func.value = ast.Constant(value="<presentation template>")
    return cell


@pytest.mark.parametrize("name", FORM_CONTROLS)
def test_form_widgets_validators_submission_and_graph_stay_identical(name):
    before, after = _cell(name, CANONICAL), _cell(name)
    assert _dump(_normalized_form(before)) == _dump(_normalized_form(after))
    before_batch, after_batch = _calls(before, "batch"), _calls(after, "batch")
    assert len(before_batch) == len(after_batch) == 1
    assert [_dump(arg) for arg in before_batch[0].keywords] == [
        _dump(arg) for arg in after_batch[0].keywords
    ]
    before_form, after_form = _calls(before, "form"), _calls(after, "form")
    assert len(before_form) == len(after_form) == 1
    assert [_dump(arg) for arg in before_form[0].keywords] == [
        _dump(arg) for arg in after_form[0].keywords
    ]
    template_constructor = after_batch[0].func.value
    assert isinstance(template_constructor, ast.Call)
    assert _dump(template_constructor.func) == _dump(ast.parse("mo.Html", mode="eval").body)
    displays = [node for node in ast.walk(_tree()) if isinstance(node, ast.Expr)
                and any(isinstance(child, ast.Name) and child.id == FORM_CONTROLS[name]
                        for child in ast.walk(node.value))]
    assert len(displays) == 1, "A form must have exactly one native display"
    assert sum(isinstance(node, ast.Name) and node.id == FORM_CONTROLS[name]
               for node in ast.walk(displays[0])) == 1


def _construct_form(name, **overrides):
    marimo = pytest.importorskip("marimo", reason="HTML form construction needs marimo")
    sys.path.insert(0, str(PROJECT))
    from src.classroom_safety import validate_experiment
    from src.classroom_media import builtin_clip_choices

    cell = copy.deepcopy(_cell(name))
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[])),
                 str(COMPANION), "exec"), namespace)  # trusted form only; no notebook/model execution
    kwargs = {
        "ATTENTION_PROMPT": "영상의 소리를 설명해 주세요",
        "LOGIT_PROMPT": "영상의 소리를 설명해 주세요",
        "KNOCKOUT_RULES": [("generated", "video", 0, 36)],
        "MAX_NEW_TOKENS": 32, "NFRAMES": 8, "USE_PRECOMPUTED": False,
        "VIDEO_PATH": Path("02321.mp4"),
        "attention_model": SimpleNamespace(thinker=SimpleNamespace(model=SimpleNamespace(layers=[None] * 36))),
        "mo": marimo, "validate_experiment": validate_experiment,
        "CLIP_CHOICES": builtin_clip_choices(),
        "CLIP_DEFAULT": next(iter(builtin_clip_choices())), "CLIP_UPLOAD": "업로드",
    }
    kwargs.update(overrides)
    fn = namespace[name]
    return fn(**{key: kwargs[key] for key in inspect.signature(fn).parameters})[0]


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.styles = []
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.styles.append(data)


@pytest.mark.parametrize("name", FORM_CONTROLS)
def test_html_form_places_each_native_widget_once_and_waits_for_submission(name):
    form = _construct_form(name)
    batch = form.element
    assert form.value is None
    template = batch._html.text
    placeholders = Counter(re.findall(r"(?<!\{)\{([a-z_]+)\}(?!\})", template))
    assert placeholders == Counter({key: 1 for key in batch.elements})
    parser = _HTML()
    parser.feed(batch.text)
    for widget in batch.elements.values():
        hosts = [attrs for tag, attrs in parser.tags
                 if tag == "marimo-ui-element" and attrs.get("object-id") == widget._id]
        assert len(hosts) == 1
    assert not any(tag in {"script", "iframe", "marimo-cell"} for tag, _ in parser.tags)
    assert not any(key.startswith("on") for _, attrs in parser.tags for key in attrs)


@pytest.mark.parametrize("name", FORM_CONTROLS)
def test_navigation_anchor_is_outside_cloned_batch_templates(name):
    """Batch/form internals clone HTML; an embedded ID creates hidden targets."""
    anchor = FORM_ANCHORS[name]
    form = _construct_form(name)
    parser = _HTML()
    parser.feed(form.text)
    assert not [attrs for _, attrs in parser.tags if attrs.get("id") == anchor], (
        f"{anchor} must not appear inside the form's cloned HTML templates"
    )
    output = _cell(name).body[-2]
    assert isinstance(output, ast.Expr)
    emitted = _HTML()
    for node in ast.walk(output.value):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            emitted.feed(node.value)
    assert sum(attrs.get("id") == anchor for _, attrs in emitted.tags) == 1
    assert sum(isinstance(node, ast.Name) and node.id == FORM_CONTROLS[name]
               for node in ast.walk(output)) == 1


def _local_styles(parser, attrs, class_name):
    """Read simple authored local rules or inline declarations, not page CSS."""
    blocks = []
    for selectors, declarations in re.findall(r"([^{}]+)\{([^{}]*)\}", "".join(parser.styles)):
        if f".{class_name}" in [selector.strip() for selector in selectors.split(",")]:
            blocks.append(declarations)
    blocks.append(attrs.get("style", ""))
    declarations = {}
    for block in blocks:
        for declaration in block.split(";"):
            if ":" in declaration:
                key, value = declaration.split(":", 1)
                declarations[key.strip().lower()] = value.strip().lower()
    return declarations


@pytest.mark.parametrize("name", FORM_CONTROLS)
def test_card_and_grid_styles_travel_inside_the_form_shadow_root(name):
    """Global navigation CSS cannot style the live MARIMO-DICT shadow root."""
    form = _construct_form(name)
    parser = _HTML()
    parser.feed(form.element.text)
    cards = [attrs for _, attrs in parser.tags
             if "ctp-html-card" in attrs.get("class", "").split()]
    fields = [attrs for _, attrs in parser.tags
              if "ctp-html-fields" in attrs.get("class", "").split()]
    assert len(cards) == 1 and fields
    card_style = _local_styles(parser, cards[0], "ctp-html-card")
    assert "solid" in card_style.get("border", ""), "Card border needs form-local styling"
    assert card_style.get("padding"), "Card padding needs form-local styling"
    for field in fields:
        style = _local_styles(parser, field, "ctp-html-fields")
        assert style.get("display") == "grid", "The form grid cannot rely on page-level CSS"
        assert "minmax(" in style.get("grid-template-columns", "")
        assert style.get("gap")


@pytest.mark.parametrize("name", FORM_CONTROLS)
def test_native_html_form_rejects_empty_band_and_recovers(name):
    form = _construct_form(name)
    value = dict(form.element._initial_value_frontend)
    key = "ko_layers" if name == "diversity_form" else "layers"
    if name == "diversity_form":
        value.update(ko_enable=True, ko_rules_text="")
    value[key] = [12, 12]
    assert "0개 레이어" in form.validate(value)
    assert form.value is None
    value[key] = [12, 24]
    assert form.validate(value) is None
    assert form.value is None


@pytest.mark.parametrize("name", ["diversity_form", "tf_form"])
def test_native_html_form_keeps_blank_prompt_and_missing_upload_guards(name):
    form = _construct_form(name)
    value = dict(form.element._initial_value_frontend)
    value["prompt"] = "   "
    assert "공백" in form.validate(value)
    value["prompt"] = "영상의 소리를 설명해 주세요"
    value.update(clip=["업로드"], video=None)
    assert "파일" in form.validate(value)
    value["clip"] = ["장면 1 · 원본 (10초)"]
    assert form.validate(value) is None
    assert form.value is None


def test_band_template_does_not_treat_prompt_braces_as_batch_inputs_or_html():
    form = _construct_form("band_form", ATTENTION_PROMPT="질문 {audio} <b>직접 입력</b>",
                           VIDEO_PATH=Path("clip-{original}.mp4"))
    assert "&lt;b&gt;직접 입력&lt;/b&gt;" in form.element.text
    assert "직접 입력" in form.element.text
    parser = _HTML()
    parser.feed(form.element.text)
    assert not any(tag == "b" for tag, _ in parser.tags)


def test_companion_has_no_studio_dependency_configuration_or_bundle_execution():
    if tomllib is None:
        pytest.skip("PEP 723 parsing requires Python 3.11")
    source = COMPANION.read_text(encoding="utf-8")
    lines = source.splitlines()
    start = lines.index("# /// script")
    end = lines.index("# ///", start + 1)
    config = tomllib.loads("\n".join(line[1:].removeprefix(" ") for line in lines[start + 1:end]))
    assert not any(re.match(r"marimo[-_]studio(?:[<>=!~\[; ]|$)", dep)
                   for dep in config["dependencies"])
    assert "marimo-studio" not in config.get("tool", {})
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            assert all(not item.name.startswith("marimo_studio") for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("marimo_studio")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert "restore_studio_bundle" not in node.func.id
    assert "_payload" not in {node.id for node in ast.walk(_cell("studio_files")) if isinstance(node, ast.Name)}


def test_navigation_uses_unique_static_anchors_and_no_javascript():
    navigation = _cell("molab_html_navigation")
    parser = _HTML()
    for node in ast.walk(navigation):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parser.feed(node.value)
    targets = [attrs["href"] for tag, attrs in parser.tags if tag == "a"]
    assert set(targets) == {"#ctp-clips", "#ctp-guide", "#ctp-diversity", "#ctp-band", "#ctp-tf", "#ctp-evidence"}
    assert len(targets) == len(set(targets))
    assert not any(tag in {"script", "iframe"} for tag, _ in parser.tags)
    assert not any(key.startswith("on") for _, attrs in parser.tags for key in attrs)
    page = _HTML()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            page.feed(node.value)
    ids = Counter(attrs["id"] for _, attrs in page.tags if "id" in attrs)
    assert all(ids[target[1:]] == 1 for target in targets)


def test_companion_app_runs_in_replay_and_keeps_gpu_forms_disabled(tmp_path, monkeypatch):
    """Run the real notebook graph; replacing only bootstrap and GPU mode."""
    pytest.importorskip("marimo", reason="notebook smoke test needs marimo")
    pytest.importorskip("qwen_omni_utils", reason="notebook imports qwen_omni_utils")
    pytest.importorskip("wigglystuff", reason="notebook imports wigglystuff")
    import importlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marimo._ast.load import load_app

    monkeypatch.chdir(PROJECT)
    monkeypatch.syspath_prepend(str(PROJECT))
    prior_figures = set(plt.get_fignums())
    try:
        outputs, defs = load_app(str(COMPANION)).run(defs={
            "PROJECT_DIR": PROJECT, "Path": Path, "REPO_DIR": PROJECT.parent,
            "REPO_REF": "local-checkout", "importlib": importlib,
            "subprocess": subprocess, "sys": sys,
            "USE_PRECOMPUTED": True, "PRECOMPUTED_DIR": PROJECT / "precomputed_kr",
            "RESULTS_DIR": tmp_path, "LOGIT_CSV_PATH": tmp_path / "logits.csv",
            "VIDEO_PATH": PROJECT / "assets" / "02321.mp4",
            "SILENT_VIDEO_PATH": PROJECT / "assets" / "02321_silent.mp4",
        })
        assert len(outputs) > 40
        assert not [output for output in outputs if "Error" in type(output).__name__]
        assert str(defs["DEVICE"]) == "cpu"
        assert defs["tf_result"] is None
        for name, controls in FORM_CONTROLS.items():
            form = defs[controls]
            assert form.value is None
            assert form._component_args["submit-button-disabled"] is True
            assert "ctp-html-card" in form.element.text
            assert set(form.element.elements) == set(_construct_form(name).element.elements)
    finally:
        for number in set(plt.get_fignums()) - prior_figures:
            plt.close(number)


@pytest.mark.parametrize("name", ["diversity_form", "tf_form"])
def test_each_video_dropdown_value_survives_real_marimo_form_submission(name):
    from src.classroom_media import builtin_clip_choices
    for label, choice in builtin_clip_choices(include_upload=False).items():
        form = _construct_form(name)
        frontend = dict(form.element._initial_value_frontend)
        assert isinstance(frontend["clip"], list), "clip must be a dropdown, not a radio"
        frontend["clip"] = [label]
        assert form.validate(frontend) is None
        assert form.value is None, "editing draft must not submit"
        form._update(frontend)
        assert form.value["clip"] == choice
        assert form.value["video"] == () or not form.value["video"]
