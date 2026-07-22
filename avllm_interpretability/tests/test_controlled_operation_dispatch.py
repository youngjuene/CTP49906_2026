"""Dispatcher tests for the eight controlled-operation families."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controlled_operations import (  # noqa: E402
    AttentionEdgeExecution,
    ControlledOperationRequest,
    build_result_link_command,
    dispatch_controlled_operation,
)


SOURCE = "sha256:" + "1" * 64
DONOR = "sha256:" + "2" * 64


def _request(operation: str, **changes) -> ControlledOperationRequest:
    values = {
        "technical_operation": operation,
        "source_content_sha256": SOURCE,
        "audio_samples": (1.0, 2.0, 3.0, 4.0),
        "video_frames": ("v0", "v1", "v2"),
        "sample_rate_hz": 1000,
        "source_reference": "private-fixture.mp4",
        "token_layout_fingerprint": "layout:fixture-v1",
    }
    values.update(changes)
    return ControlledOperationRequest(**values)


def test_media_operations_change_only_the_declared_signal() -> None:
    original = dispatch_controlled_operation(_request("original_reference"))
    assert original.audio_samples == (1.0, 2.0, 3.0, 4.0)
    assert original.video_frames == ("v0", "v1", "v2")
    assert original.intervention_kind == "media_transform"
    assert original.token_layout_fingerprint == "layout:fixture-v1"

    swap = dispatch_controlled_operation(
        _request(
            "audio_swap_duration_matched",
            donor_audio_samples=(9.0, 8.0),
            donor_sample_rate_hz=1000,
            donor_content_sha256=DONOR,
        )
    )
    assert swap.audio_samples == (9.0, 8.0, 0, 0)
    assert swap.video_frames == original.video_frames
    assert swap.provenance["donor_content_sha256"] == DONOR

    offset = dispatch_controlled_operation(
        _request("temporal_offset", offset_ms=2)
    )
    assert offset.audio_samples == (0, 0, 1.0, 2.0)
    assert offset.video_frames == original.video_frames

    silence = dispatch_controlled_operation(_request("audio_silence_control"))
    assert silence.audio_samples == (0, 0, 0, 0)
    assert silence.video_frames == original.video_frames
    assert silence.intervention_kind == "signal_control"

    neutral = dispatch_controlled_operation(
        _request("video_neutral_control", neutral_video_value="black")
    )
    assert neutral.audio_samples == original.audio_samples
    assert neutral.video_frames == ("black", "black", "black")
    assert neutral.intervention_kind == "signal_control"


def test_every_declared_ui_operation_executes_through_the_dispatcher() -> None:
    requests = (
        _request("original_reference"),
        _request(
            "audio_swap_duration_matched",
            donor_audio_samples=(9.0,),
            donor_sample_rate_hz=1000,
            donor_content_sha256=DONOR,
        ),
        _request("temporal_offset", offset_ms=-1),
        _request("audio_silence_control"),
        _request("video_neutral_control", neutral_video_value="black"),
        _request("audio_omitted_model_input", omission_nframes=3),
        _request("video_omitted_model_input"),
        _request(
            "direct_attention_edge_knockout",
            knockout_rules=(("answer", "audio", 0, 2),),
        ),
    )
    results = tuple(dispatch_controlled_operation(request) for request in requests)
    assert tuple(result.technical_operation for result in results) == tuple(
        request.technical_operation for request in requests
    )
    assert len({result.result_digest for result in results}) == len(results)


def test_true_omission_never_substitutes_silence_or_neutral_signal() -> None:
    audio_omitted = dispatch_controlled_operation(
        _request("audio_omitted_model_input", omission_nframes=3)
    )
    assert audio_omitted.audio_samples is None
    assert audio_omitted.video_frames == ("v0", "v1", "v2")
    assert audio_omitted.modality_presence == {"audio": False, "video": True}
    assert audio_omitted.omission_request is not None
    assert audio_omitted.omission_request.absent_modality == "audio"
    assert audio_omitted.intervention_kind == "modality_omission"
    assert audio_omitted.token_layout_fingerprint is None
    assert audio_omitted.provenance["layout_status"] == "requires_processor_attestation"

    video_omitted = dispatch_controlled_operation(
        _request("video_omitted_model_input")
    )
    assert video_omitted.audio_samples == (1.0, 2.0, 3.0, 4.0)
    assert video_omitted.video_frames is None
    assert video_omitted.modality_presence == {"audio": True, "video": False}
    assert video_omitted.omission_request is not None
    assert video_omitted.omission_request.absent_modality == "video"


def test_edge_knockout_is_model_intervention_not_media_or_omission() -> None:
    rules = (("answer", "audio", 2, 4),)
    result = dispatch_controlled_operation(
        _request("direct_attention_edge_knockout", knockout_rules=rules)
    )
    assert result.audio_samples == (1.0, 2.0, 3.0, 4.0)
    assert result.video_frames == ("v0", "v1", "v2")
    assert result.knockout_rules == rules
    assert result.omission_request is None
    assert result.intervention_kind == "model_intervention"
    assert result.token_layout_fingerprint == "layout:fixture-v1"
    assert isinstance(result.edge_execution, AttentionEdgeExecution)
    assert callable(result.edge_execution.context)


def test_omission_and_edge_execution_boundaries_are_each_called_exactly_once() -> None:
    omission_calls = []
    edge_calls = []

    def execute_omission(operation, request):
        omission_calls.append((operation, request.source_reference))
        absent = "audio" if operation.startswith("audio_") else "video"
        return SimpleNamespace(
            technical_operation=operation,
            absent_modality=absent,
            present_modality="video" if absent == "audio" else "audio",
        )

    def execute_edge(rules, request):
        edge_calls.append((rules, request.source_content_sha256))
        return {"handler": "fake-edge", "rules": rules}

    audio = dispatch_controlled_operation(
        _request("audio_omitted_model_input", omission_nframes=3),
        omission_executor=execute_omission,
    )
    video = dispatch_controlled_operation(
        _request("video_omitted_model_input"),
        omission_executor=execute_omission,
    )
    edge = dispatch_controlled_operation(
        _request(
            "direct_attention_edge_knockout",
            knockout_rules=(("answer", "audio", 1, 3),),
        ),
        edge_executor=execute_edge,
    )

    assert omission_calls == [
        ("audio_omitted_model_input", "private-fixture.mp4"),
        ("video_omitted_model_input", "private-fixture.mp4"),
    ]
    assert edge_calls == [
        ((("answer", "audio", 1, 3),), SOURCE),
    ]
    assert audio.omission_request.absent_modality == "audio"
    assert video.omission_request.absent_modality == "video"
    assert edge.edge_execution["handler"] == "fake-edge"


def test_dispatch_provenance_is_stable_and_semantic_inputs_change_identity() -> None:
    request = _request("temporal_offset", offset_ms=-1)
    first = dispatch_controlled_operation(request)
    second = dispatch_controlled_operation(request)
    changed = dispatch_controlled_operation(
        _request("temporal_offset", offset_ms=1)
    )

    assert first.provenance == second.provenance
    assert first.provenance["dispatch_digest"].startswith("sha256:")
    assert first.provenance["dispatch_digest"] != changed.provenance["dispatch_digest"]
    assert first.provenance["evidence_scope"] == "local_only"
    assert first.provenance["target_runtime_verified"] is False


def test_invalid_dispatch_runs_failure_cleanup_without_returning_partial_state() -> None:
    cleanup_calls = []
    with pytest.raises(ValueError, match="donor"):
        dispatch_controlled_operation(
            _request("audio_swap_duration_matched"),
            on_failure_cleanup=lambda: cleanup_calls.append("cleaned"),
        )
    assert cleanup_calls == ["cleaned"]

    with pytest.raises(ValueError, match="knockout_rules"):
        dispatch_controlled_operation(
            _request("direct_attention_edge_knockout"),
            on_failure_cleanup=lambda: cleanup_calls.append("edge-cleaned"),
        )
    assert cleanup_calls[-1] == "edge-cleaned"


def test_result_link_command_is_exactly_once_in_the_neutral_session_log() -> None:
    from datetime import datetime, timezone
    from uuid import UUID

    from curriculum_common.session_records import new_session, reduce_command

    run_id = "00000000-0000-4000-8000-000000000001"
    result_id = "00000000-0000-4000-8000-000000000002"
    UUID(run_id)
    UUID(result_id)
    now = datetime(2026, 7, 22, tzinfo=timezone.utc)
    log = new_session("fixture-session", now=now)
    committed = reduce_command(
        log,
        {
            "kind": "commit_run",
            "command_nonce": "run-once",
            "run_id": run_id,
            "artifact_id": "artifact:fixture",
            "stimulus_id": "stimulus:fixture",
            "condition_code": "original-reference",
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
            "prompt": "fixture prompt",
            "parameters": {},
            "prediction": "fixture prediction",
            "initial_explanation": "fixture explanation",
        },
        now=now,
    )
    dispatch = dispatch_controlled_operation(_request("original_reference"))
    command = build_result_link_command(
        dispatch,
        run_id=run_id,
        command_nonce="result-once",
        metrics={"fixture_metric": 1},
    )
    command["result_id"] = result_id
    first = reduce_command(committed.log, command, now=now)
    second = reduce_command(first.log, command, now=now)

    assert first.replayed is False
    assert second.replayed is True
    assert first.record_ids == second.record_ids == (result_id,)
    assert len(second.log.records_of_type("result")) == 1
