"""Static Studio-facing contracts for the Korean marimo notebook.

These tests avoid importing marimo or loading the model. They pin the notebook
surface that the Studio side pane projects: stable cell names, native form
field sets/validators, and cache identity for teacher-forcing captions.
"""

import ast
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "CTP49906_avllm_molab_kr.py"


def _source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(NOTEBOOK))


def _function(name: str) -> ast.FunctionDef:
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _function_source(name: str) -> str:
    return ast.get_source_segment(_source(), _function(name)) or ""


def test_studio_projection_cells_have_stable_names() -> None:
    expected = {
        "lab_intro",
        "method_guide",
        "clip_preview",
        "token_census",
        "probe_grid_panel",
        "probe_summary_panel",
        "guided_captions",
        "attention_mass_panel",
        "guided_tf_panel",
        "guided_tf_threshold",
        "guided_tf_tokens",
        "band_form",
        "band_result_panel",
        "diversity_form",
        "diversity_result_panel",
        "tf_form",
        "tf_result_panel",
        "tf_threshold_panel",
        "tf_tokens_panel",
        "ledger_panel",
        "verdict_panel",
        "verdict_submit_handler",
        "worksheet_panel",
        "studio_status",
    }
    names = {node.name for node in _tree().body if isinstance(node, ast.FunctionDef)}
    assert expected <= names
    assert "그림 넷" in _function_source("method_guide")


def test_studio_status_is_replay_or_live_only_and_model_free() -> None:
    fn = _function("studio_status")
    refs = {
        node.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert {"mo", "USE_PRECOMPUTED"} <= refs
    assert refs.isdisjoint({"MODEL_PATH", "attention_model", "attention_processor", "DEVICE"})
    text = _function_source("studio_status")
    assert "라이브" in text and "재생" in text


def test_teacher_forcing_caption_cache_identity_includes_token_budget() -> None:
    text = _function_source("tf_result_panel")

    assert "_tf_cap_key = caption_cache_key(" in text
    assert "_tf_max_tokens" in text
    assert "run_provenance" in text
    assert "max_new_tokens=_tf_max_tokens" in text
    assert ".stat().st_size" not in text


def test_native_forms_keep_field_sets_validators_and_submit_gates() -> None:
    contracts = {
        "band_form": {
            "returns": "band_controls",
            "fields": {"target", "layers", "null_band"},
            "validator": "_band_validate",
            "submit": "▶ 이 대역으로 다시 생성",
        },
        "diversity_form": {
            "returns": "ko_controls",
            "fields": {
                "clip",
                "video",
                "nframes",
                "prompt",
                "ko_enable",
                "ko_source",
                "ko_target",
                "ko_layers",
                "ko_rules_text",
                "compare",
            },
            "validator": "_ko_validate",
            "submit": "▶ Logit-lens 다양성 실행",
        },
        "tf_form": {
            "returns": "tf_controls",
            "fields": {"clip", "video", "nframes", "prompt", "max_new_tokens", "target", "layers"},
            "validator": "_tf_validate",
            "submit": "▶ 티처 포싱 Δ log-우도 실행",
        },
        "verdict_panel": {
            "returns": "verdict_form",
            "fields": {"run_id", "claim", "verdict", "rival"},
            "validator": None,
            "submit": "판정 기록",
        },
    }

    for function_name, contract in contracts.items():
        text = _function_source(function_name)
        assert f"return ({contract['returns']}," in text, function_name
        assert ".form(" in text and ".batch(" in text, function_name
        assert f"submit_button_label=\"{contract['submit']}\"" in text, function_name
        if contract["validator"] is not None:
            assert f"validate={contract['validator']}" in text, function_name
        for field in contract["fields"]:
            assert f"{field}=mo.ui." in text, (function_name, field)
        assert "prediction=mo.ui." not in text


def test_long_help_upload_and_advanced_copy_is_progressively_disclosed() -> None:
    for function_name in ("band_form", "diversity_form", "tf_form"):
        text = _function_source(function_name)
        assert "<details" in text and "<summary" in text, function_name
        assert "prediction" not in text.lower(), function_name
