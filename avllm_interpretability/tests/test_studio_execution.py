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


def _submitted_value(*, layers=(0, 4), target="audio", prompt="  classroom prompt  "):
    return {
        "clip": "Default clip",
        "video": None,
        "nframes": 8,
        "prompt": prompt,
        "target": target,
        "layers": layers,
    }


def _base_kwargs(tmp_path: Path, *, controls_value, use_precomputed=False, max_new_tokens=16):
    import numpy as np

    from src.run_ledger import append_run, run_record

    processor = FakeProcessor()
    runs = []

    def set_runs(updater):
        runs[:] = updater(runs)

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")

    return {
        "kwargs": {
            "LEDGER_LOG": tmp_path / "runs.jsonl",
            "LOGIT_PROMPT": "fallback prompt",
            "MAX_NEW_TOKENS": max_new_tokens,
            "USE_PRECOMPUTED": use_precomputed,
            "append_run": append_run,
            "attention_model": FakeModel(),
            "attention_processor": processor,
            "cache_put": _cache_put,
            "create_attention_token_mapping": lambda input_ids, config: ["query_text", "audio", "video"],
            "mo": marimo,
            "np": np,
            "playground_caches": {"encode": {}, "caption": {}},
            "resolve_clip": lambda clip_name, upload: (clip, False, None),
            "run_record": run_record,
            "set_runs": set_runs,
            "tf_controls": Controls(controls_value),
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
            "delta": torch.tensor([-0.25, -0.75], dtype=torch.float32),
            "delta_total": -1.0,
            "delta_mean": -0.5,
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
    env = _base_kwargs(tmp_path, controls_value=_submitted_value(), max_new_tokens=16)
    kwargs = env["kwargs"]

    first = panel(**kwargs)[0]
    kwargs["tf_controls"] = Controls(_submitted_value(layers=(1, 3), target="video"))
    second = panel(**kwargs)[0]
    kwargs["MAX_NEW_TOKENS"] = 24
    kwargs["tf_controls"] = Controls(_submitted_value(layers=(2, 4), target="audio"))
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
    assert {
        key[-1] for key in caption_cache
    } == {16, 24}

    assert len(env["runs"]) == 3
    assert env["runs"][-1]["kind"] == "teacher_forcing"
    assert env["runs"][-1]["condition"] == "answer→audio [2,4)"
    assert env["runs"][-1]["metric_name"] == "delta_per_token"
    assert env["runs"][-1]["metric_value"] == -0.5
    assert env["runs"][-1]["metric_unit"] == "nats/token"
    assert env["runs"][-1]["config"] == {
        "clip": env["clip"].name,
        "nframes": 8,
        "prompt": "classroom prompt",
        "target": "audio",
        "start": 2,
        "end": 4,
        "max_new_tokens": 24,
    }
    assert env["runs"][-1]["extra"] == {"delta_total": -1.0, "n_tokens": 2}
