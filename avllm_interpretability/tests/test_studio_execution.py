"""Execution-boundary tests for Studio-projected teacher forcing."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

marimo = pytest.importorskip("marimo", reason="Studio execution checks need marimo")
pytest.importorskip("torch", reason="teacher-forcing fakes use CPU torch tensors")

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "CTP49906_avllm_molab_kr.py"
sys.path.insert(0, str(PROJECT))


def _notebook_source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def _extract_function(name: str):
    tree = ast.parse(_notebook_source(), filename=str(NOTEBOOK))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.decorator_list = []
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace: dict[str, object] = {}
            exec(compile(module, str(NOTEBOOK), "exec"), namespace)  # noqa: S102 -- trusted cell, mocked inference
            return namespace[name]
    raise AssertionError(f"missing notebook function {name}")


class Controls:
    def __init__(self, value):
        self.value = value


class FakeProcessor:
    def __init__(self):
        self.templates: list[object] = []
        self.calls = 0

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
        self.templates.append(conversation)
        assert add_generation_prompt is True
        assert tokenize is False
        return "chat-template"

    def __call__(self, **kwargs):
        import torch

        self.calls += 1
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["use_audio_in_video"] is True
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }


class FakeModel:
    def __init__(self):
        import torch

        self.device = torch.device("cpu")
        thinker_config = types.SimpleNamespace()
        self.config = types.SimpleNamespace(thinker_config=thinker_config)


def _cache_put(cache, key, value):
    cache[key] = value
    return value


def _submitted_value(*, layers=(0, 4), target="audio", prompt="  classroom prompt  ", max_new_tokens=16):
    return {
        "clip": "Default clip",
        "video": None,
        "nframes": 8,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "target": target,
        "layers": layers,
    }


def _base_kwargs(tmp_path: Path, *, controls_value, use_precomputed=False):
    import numpy as np

    from src.classroom_display import caption_cache_key
    from src.run_ledger import append_run, run_record

    processor = FakeProcessor()
    runs = []

    def set_runs(updater):
        runs[:] = updater(runs)

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")

    def experiment_config(clip_path, **settings):
        return {
            "clip": clip_path.name,
            "clip_sha256": "fake-sha256",
            "model_id": "Qwen/Fake",
            "model_revision": "fake-revision",
            "repo_revision": "fake-repo",
            "notebook_sha256": "fake-notebook",
            **settings,
        }

    return {
        "kwargs": {
            "LEDGER_LOG": tmp_path / "runs.jsonl",
            "USE_PRECOMPUTED": use_precomputed,
            "append_run": append_run,
            "attention_model": FakeModel(),
            "attention_processor": processor,
            "cache_put": _cache_put,
            "caption_cache_key": caption_cache_key,
            "create_attention_token_mapping": lambda input_ids, config: ["query_text", "audio", "video"],
            "experiment_config": experiment_config,
            "mo": marimo,
            "np": np,
            "playground_caches": {"encode": {}, "caption": {}},
            "resolve_clip": lambda clip_name, upload: (clip, False, None),
            "run_provenance": {"model_id": "Qwen/Fake", "model_revision": "fake-revision"},
            "run_record": run_record,
            "set_runs": set_runs,
            "tf_controls": Controls(controls_value),
            "validate_encoded_inputs": lambda *args, **kwargs: None,
            "validate_experiment": lambda *args, **kwargs: None,
        },
        "processor": processor,
        "runs": runs,
        "clip": clip,
    }


def _install_heavy_boundary_fakes(monkeypatch, *, fail_on_score=False):
    import torch

    from src import teacher_forcing

    calls = {"mm": [], "tfd": []}

    def process_mm_info(conversation, *, use_audio_in_video):
        calls["mm"].append({"conversation": conversation, "use_audio_in_video": use_audio_in_video})
        return ["audio"], ["image"], ["video"]

    def teacher_forced_delta(model, processor, inputs, token_types, rules, **kwargs):
        if fail_on_score:
            raise AssertionError("teacher_forced_delta should not be called")
        call = {
            "rules": list(rules),
            "max_new_tokens": kwargs["max_new_tokens"],
            "cached_caption_ids": kwargs["cached_caption_ids"],
        }
        calls["tfd"].append(call)
        if kwargs["cached_caption_ids"] is None:
            start = int(kwargs["max_new_tokens"])
            caption_ids = torch.tensor([[start, start + 1]], dtype=torch.long)
        else:
            caption_ids = kwargs["cached_caption_ids"]
        return {
            "caption_ids": caption_ids,
            "caption_tokens": [" cow", " bell"],
            "caption_token_kinds": ["text", "text"],
            "caption_text": "cow bell",
            "delta": torch.tensor([-0.25, -0.75], dtype=torch.float32),
            "delta_total": -1.0,
            "delta_mean": -0.5,
            "baseline_logprobs": torch.tensor([-0.1, -0.2], dtype=torch.float32),
            "knockout_logprobs": torch.tensor([-0.35, -0.95], dtype=torch.float32),
            "baseline_distribution": {"audio": 0.6},
            "knockout_distribution": {"audio": 0.2},
            "generation_truncated": False,
            "generation_end_reason": "eos",
        }

    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        types.SimpleNamespace(process_mm_info=process_mm_info),
    )
    monkeypatch.setattr(teacher_forcing, "teacher_forced_delta", teacher_forced_delta)
    return calls


@pytest.mark.parametrize(
    ("controls_value", "use_precomputed", "resolve_error"),
    [
        (None, False, None),
        (_submitted_value(), True, None),
        (_submitted_value(), False, "upload is invalid"),
    ],
)
def test_tf_result_panel_does_not_score_unsubmitted_replay_or_invalid_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controls_value,
    use_precomputed,
    resolve_error,
) -> None:
    calls = _install_heavy_boundary_fakes(monkeypatch, fail_on_score=True)
    panel = _extract_function("tf_result_panel")
    env = _base_kwargs(
        tmp_path,
        controls_value=controls_value,
        use_precomputed=use_precomputed,
    )
    if resolve_error is not None:
        env["kwargs"]["resolve_clip"] = lambda clip_name, upload: (None, False, resolve_error)

    result = panel(**env["kwargs"])

    assert result == (None,)
    assert calls["tfd"] == []
    assert env["processor"].calls == 0
    assert env["runs"] == []


def test_tf_result_panel_reuses_caption_cache_by_budget_and_logs_run_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_heavy_boundary_fakes(monkeypatch)
    panel = _extract_function("tf_result_panel")
    env = _base_kwargs(tmp_path, controls_value=_submitted_value(max_new_tokens=16))
    kwargs = env["kwargs"]

    first = panel(**kwargs)[0]
    kwargs["tf_controls"] = Controls(_submitted_value(layers=(1, 3), target="video"))
    second = panel(**kwargs)[0]
    kwargs["tf_controls"] = Controls(_submitted_value(layers=(2, 4), target="audio", max_new_tokens=24))
    third = panel(**kwargs)[0]

    assert first["caption_ids"].tolist() == [[16, 17]]
    assert second["caption_ids"].tolist() == [[16, 17]]
    assert third["caption_ids"].tolist() == [[24, 25]]
    assert calls["mm"] and len(calls["mm"]) == 1
    assert env["processor"].calls == 1
    assert [call["max_new_tokens"] for call in calls["tfd"]] == [16, 16, 24]
    assert calls["tfd"][0]["cached_caption_ids"] is None
    assert calls["tfd"][1]["cached_caption_ids"].tolist() == [[16, 17]]
    assert calls["tfd"][2]["cached_caption_ids"] is None

    caption_cache = kwargs["playground_caches"]["caption"]
    assert len(caption_cache) == 2
    assert all(isinstance(key, str) for key in caption_cache)
    assert len(set(caption_cache)) == 2

    assert len(env["runs"]) == 3
    assert env["runs"][-1]["kind"] == "teacher_forcing"
    assert env["runs"][-1]["condition"] == "answer→audio [2,4)"
    assert env["runs"][-1]["metric_name"] == "delta_per_token"
    assert env["runs"][-1]["metric_value"] == -0.5
    assert env["runs"][-1]["metric_unit"] == "nats/token"
    assert env["runs"][-1]["config"] | {
        "clip": env["clip"].name,
        "nframes": 8,
        "prompt": "classroom prompt",
        "target": "audio",
        "start": 2,
        "end": 4,
        "max_new_tokens": 24,
    } == env["runs"][-1]["config"]
    assert env["runs"][-1]["config"]["model_id"] == "Qwen/Fake"
    assert env["runs"][-1]["config"]["model_revision"] == "fake-revision"
    assert env["runs"][-1]["extra"]["delta_total"] == -1.0
    assert env["runs"][-1]["extra"]["delta_mean"] == -0.5
    assert env["runs"][-1]["extra"]["n_tokens"] == 2
    assert env["runs"][-1]["extra"]["caption_text"] == "cow bell"


@pytest.mark.parametrize("cell,result_key", [
    ("tf_threshold_panel", "tf_result"),
    ("guided_tf_threshold", "w9_tf_result"),
])
def test_threshold_retyping_displayed_value_does_not_snap_or_launch_inference(cell, result_key):
    import torch
    from src.classroom_display import selected_drop_share
    result = {
        "caption_tokens": [" low", " high"],
        "delta": torch.tensor([-1.59, -6.8]),
        "caption_token_kinds": ["text", "text"],
    }
    control = _extract_function(cell)(marimo, result)[0]
    # The native number value is scalar and round-trips at its displayed precision.
    assert isinstance(control.value, (float, int))
    assert control.value == 6.73
    control._update(6.73)
    assert control.value == 6.73
    tokens_cell = "tf_tokens_panel" if cell == "tf_threshold_panel" else "guided_tf_tokens"
    result["caption_text"] = "low high"
    for value in (0.0, 6.73, 7.14):
        control._update(value)
        _extract_function(tokens_cell)(**{
            "mo": marimo, "selected_drop_share": selected_drop_share,
            result_key: result,
            "tf_threshold" if cell == "tf_threshold_panel" else "w9_threshold": control,
        })
        assert control.value == value


def test_last_run_heading_matches_recorded_inputs_after_draft_changes(tmp_path, monkeypatch):
    import CTP49906_avllm_molab_kr as notebook
    calls = _install_heavy_boundary_fakes(monkeypatch)
    env = _base_kwargs(tmp_path, controls_value=_submitted_value(prompt="submitted", max_new_tokens=16))
    output, _ = notebook.tf_result_panel.run(**env["kwargs"])
    run = env["runs"][0]
    env["kwargs"]["tf_controls"].value = _submitted_value(prompt="unsubmitted draft", max_new_tokens=64)
    assert run["run_id"] in output.text
    assert "submitted" in output.text
    assert "unsubmitted draft" not in output.text
    assert "16토큰" in output.text
    assert len(calls["tfd"]) == 1 and len(env["runs"]) == 1


@pytest.mark.parametrize("cell,result_key,control_key", [
    ("tf_tokens_panel", "tf_result", "tf_threshold"),
    ("guided_tf_tokens", "w9_tf_result", "w9_threshold"),
])
def test_cleared_threshold_shows_prompt_and_retyping_restores_counts(cell, result_key, control_key):
    import torch
    import CTP49906_avllm_molab_kr as notebook
    from src.classroom_display import selected_drop_share
    result = {"caption_tokens": [" low", " high"], "delta": torch.tensor([-1.59, -6.8]),
              "caption_token_kinds": ["text", "text"]}
    threshold_cell = "tf_threshold_panel" if cell == "tf_tokens_panel" else "guided_tf_threshold"
    control = _extract_function(threshold_cell)(marimo, result)[0]
    control._update(None)
    output, _ = getattr(notebook, cell).run(**{
        "mo": marimo, "selected_drop_share": selected_drop_share,
        result_key: result, control_key: control,
    })
    assert "강조 임계값을 입력" in output.text
    control._update(1.52)
    output, _ = getattr(notebook, cell).run(**{
        "mo": marimo, "selected_drop_share": selected_drop_share,
        result_key: result, control_key: control,
    })
    assert "전체 2개 표시 묶음 중 <strong>2개</strong>" in output.text
    assert control.value == 1.52


@pytest.mark.parametrize("choice,filename,is_control", [
    ("scene02", "scene02.mp4", False),
    ("scene02_silent", "scene02_silent.mp4", True),
    ("scene03", "scene03.mp4", False),
    ("scene03_silent", "scene03_silent.mp4", True),
])
def test_new_video_selection_reaches_model_input_and_record(tmp_path, monkeypatch, choice, filename, is_control):
    calls = _install_heavy_boundary_fakes(monkeypatch)
    env = _base_kwargs(tmp_path, controls_value={**_submitted_value(), "clip": choice})
    library = _extract_function("clip_library")
    _, _, _, resolve = library(PROJECT, Path, tmp_path, lambda path: None)
    env["kwargs"]["resolve_clip"] = resolve
    _extract_function("tf_result_panel")(**env["kwargs"])
    conversation = calls["mm"][0]["conversation"]
    assert conversation[0]["content"][1]["video"] == str(PROJECT / "assets" / filename)
    assert env["runs"][-1]["config"]["clip"] == filename
    assert env["runs"][-1]["is_control"] is is_control
    assert len(calls["tfd"]) == 1


def test_changing_video_cannot_reuse_another_videos_caption(tmp_path, monkeypatch):
    calls = _install_heavy_boundary_fakes(monkeypatch)
    env = _base_kwargs(tmp_path, controls_value=None)
    _, _, _, resolve = _extract_function("clip_library")(PROJECT, Path, tmp_path, lambda path: None)
    env["kwargs"]["resolve_clip"] = resolve
    panel = _extract_function("tf_result_panel")
    for choice in ("scene02", "scene03", "scene03_silent", "scene02"):
        env["kwargs"]["tf_controls"] = Controls({**_submitted_value(), "clip": choice})
        panel(**env["kwargs"])
    assert len(calls["mm"]) == 3
    assert all(call["cached_caption_ids"] is None for call in calls["tfd"][:3])
    assert calls["tfd"][3]["cached_caption_ids"] is not None


def test_preview_selection_does_not_become_an_inference_dependency():
    source = ast.parse(_notebook_source())
    readers = {node.name for node in source.body if isinstance(node, ast.FunctionDef)
               and any(arg.arg == "preview_clip" for arg in node.args.args)}
    assert readers == {"clip_preview"}
    for name in ("tf_result_panel", "diversity_result_panel", "band_result_panel"):
        cell = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert all(arg.arg not in {"preview_clip", "clip_preview"} for arg in cell.args.args)
