"""Idle/replay contracts for Studio-projected notebook cells."""

import ast
import sys
from pathlib import Path

import pytest

marimo = pytest.importorskip("marimo", reason="idle replay checks need marimo")
pytest.importorskip("qwen_omni_utils", reason="notebook imports qwen_omni_utils")
pytest.importorskip("wigglystuff", reason="notebook imports wigglystuff")

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "CTP49906_avllm_molab_kr.py"
PACK = PROJECT / "precomputed_kr"
sys.path.insert(0, str(PROJECT))


def _source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(_source(), filename=str(NOTEBOOK))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _function_source(name: str) -> str:
    return ast.get_source_segment(_source(), _function(name)) or ""


def _calls_mo_stop(function_name: str) -> bool:
    for node in ast.walk(_function(function_name)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "stop" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mo"):
            return True
    return False


def test_idle_studio_cells_do_not_use_mo_stop() -> None:
    for function_name in (
        "guided_tf_threshold",
        "guided_tf_tokens",
        "tf_result_panel",
        "tf_threshold_panel",
        "tf_tokens_panel",
    ):
        assert not _calls_mo_stop(function_name), function_name


def test_teacher_forcing_cells_define_idle_values_in_replay() -> None:
    import importlib
    import os
    import subprocess

    import matplotlib
    from marimo._ast.load import load_app

    matplotlib.use("Agg")
    cwd = os.getcwd()
    os.chdir(PROJECT)
    try:
        app = load_app(str(NOTEBOOK))
        outputs, defs = app.run(defs={
            "PROJECT_DIR": PROJECT,
            "Path": Path,
            "REPO_DIR": PROJECT.parent,
            "REPO_REF": "local-checkout",
            "importlib": importlib,
            "subprocess": subprocess,
            "sys": sys,
            "USE_PRECOMPUTED": True,
            "PRECOMPUTED_DIR": PACK,
        })
    finally:
        os.chdir(cwd)

    assert len(outputs) > 40
    assert not [o for o in outputs if "Error" in type(o).__name__]
    assert "tf_result" in defs and defs["tf_result"] is None
    assert "tf_threshold" in defs and defs["tf_threshold"] is None
    assert "w9_tf_result" in defs and defs["w9_tf_result"] is None
    assert "w9_threshold" in defs and defs["w9_threshold"] is None


def test_replay_inference_forms_are_disabled_but_keep_fields() -> None:
    for function_name, control_name in (
        ("band_form", "band_controls"),
        ("diversity_form", "ko_controls"),
        ("tf_form", "tf_controls"),
    ):
        source = _function_source(function_name)
        assert "USE_PRECOMPUTED" in source, function_name
        assert "submit_button_disabled=USE_PRECOMPUTED" in source, function_name
        assert "submit_button_tooltip=" in source, function_name
        assert f"return ({control_name}," in source, function_name


def test_studio_status_is_plain_one_line_status() -> None:
    source = _function_source("studio_status")
    assert "mo.md(" in source
    assert "mo.callout" not in source
    assert "USE_PRECOMPUTED=True" not in source
    assert "precomputed_kr" not in source


def test_idle_verdict_handler_has_visible_output_without_writing(tmp_path):
    from types import SimpleNamespace

    import marimo as mo

    import CTP49906_avllm_molab_kr as notebook

    def no_write(*args, **kwargs):
        raise AssertionError("An unsubmitted verdict must not write state")

    output, _ = notebook.verdict_submit_handler.run(
        LEDGER_LOG=tmp_path / "log.jsonl", mo=mo, set_runs=no_write,
        verdict_form=SimpleNamespace(value=None),
    )
    assert "판정 기록" in output.text


def test_worksheet_download_callbacks_capture_exact_utf8_evidence(monkeypatch):
    import asyncio
    import base64
    import json
    import CTP49906_avllm_molab_kr as notebook
    from src.run_ledger import build_worksheet_md, run_record
    from marimo._plugins.stateless.download import EmptyArgs

    run = run_record(kind="teacher_forcing", condition="answer→audio [0,36)",
        metric_name="delta_per_token", metric_value=-0.5, metric_unit="nats/token",
        config={"prompt": "<script>한글</script>", "clip": "02321.mp4"})
    runs = [run]
    provenance = {"repo_revision": "qa-example"}
    downloads = []
    original_download = marimo.download
    def capture_download(*args, **kwargs):
        widget = original_download(*args, **kwargs)
        downloads.append(widget)
        return widget
    monkeypatch.setattr(marimo, "download", capture_download)
    output, _ = notebook.worksheet_panel.run(
        get_runs=lambda: runs, mo=marimo, run_provenance=provenance,
        worksheet_md=build_worksheet_md,
    )
    # Exercise the real lazy callback used by the native download control.
    payloads = {}
    for widget in downloads:
        response = asyncio.run(widget._load(EmptyArgs()))
        payloads[response.filename] = base64.b64decode(response.data.split(",", 1)[1]).decode("utf-8")
    assert set(payloads) == {"lab_log.md", "lab_evidence.json"}
    assert payloads["lab_log.md"] == build_worksheet_md(runs)
    evidence = json.loads(payloads["lab_evidence.json"])
    assert evidence["runs"] == [dict(run, matched_control_ids=[])]
    assert evidence["provenance"] == provenance
    assert "<script>한글</script>" not in output.text
