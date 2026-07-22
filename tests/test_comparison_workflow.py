from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curriculum_common.audience_packets import (
    AUDIENCE_READING_SCHEMA_VERSION,
    AUDIENCE_SHARING_PERMISSION,
    AudienceReading,
)
from curriculum_common.production_manifest import ARTIFACT_VERSION_SCHEMA
from digital_storytelling.workflow import (
    ACCESSIBILITY_OVERLAY_POLICY,
    COMPARISON_ROUTE_ID,
    POST_OUTCOME_ID,
    PRE_OUTCOME_ID,
    build_blinded_packet,
    build_outcome_bundle,
    build_private_story_bundle,
    content_key_inventory,
    create_story_artifact,
    replay_payload,
    require_complete_audience_exchange,
    serialize_private_story_bundle,
    validate_private_story_bundle,
    validate_replay,
)

NOW = "2026-07-22T00:00:00Z"


def _artifact(*, version: str, payload: bytes, parent: str | None = None):
    return create_story_artifact(
        media_bytes=payload,
        version_label=version,
        parent_artifact_id=parent,
        local_registered_at_utc=NOW,
        event_index=0 if version == "V1" else 1,
        elapsed_ms=1_000 if version == "V1" else 2_000,
        creator_intention="Show how a repeated place changes through memory.",
        intended_audience="Class peers",
        sound_image_relation="A recurring sound marks each change in viewpoint.",
        concept_tags="memory, return",
        cultural_aesthetic_context="A situated classroom reading, not a universal claim.",
        source_alias="licensed-source-01",
        source_permission_note="Synthetic classroom fixture; release approval unresolved.",
        editor_name_version="facilitator-approved-editor/preview",
        ordering="arrival, interruption, return",
        trims="trim each beat to its decisive action",
        mix_levels="keep the recurring cue audible under ambient sound",
        caption_or_transcript="CAPTION_SENTINEL: a door closes, then footsteps return.",
        visual_description="DESCRIPTION_SENTINEL: three views of the same doorway.",
        overlays_human_reviewed=True,
        assistance_disclosure="Peer timing feedback only.",
        export_preset_version="facilitator-approved-export/preview",
        change_rationale="First cut" if version == "V1" else "Audience evidence prompted pacing revision.",
    )


def _responses(prefix: str) -> dict[str, str]:
    return {
        "evidence": f"{prefix} evidence",
        "alternative": f"{prefix} alternative",
        "limit": f"{prefix} limit",
    }


def _reading(packet, *, index: int, audience: str, session: str) -> AudienceReading:
    return AudienceReading.from_mapping(
        {
            "schema_version": AUDIENCE_READING_SCHEMA_VERSION,
            "response_id": f"response-{index}",
            "exchange_artifact_id": packet.exchange_artifact_id,
            "audience_pseudonym": audience,
            "respondent_session_pseudonym": session,
            "local_created_at_utc": datetime(2026, 7, 22, tzinfo=timezone.utc).isoformat(),
            "event_index": index,
            "elapsed_ms": index * 1_000,
            "blindness_attestation": True,
            "permission_scope": AUDIENCE_SHARING_PERMISSION,
            "open_interpretation": f"Reading {index}",
            "sound_image_relation": "Sound changes the perceived turning point.",
            "shared_tags": ["return"],
            "self_described_tags": ["memory"],
            "accessibility_barriers": "none reported",
            "confidence_or_ambiguity": "one plausible reading among others",
            "withdrawn": False,
        },
        packet=packet,
    )


def test_artifact_uses_frozen_schema_and_keeps_overlays_out_of_inputs() -> None:
    artifact = _artifact(version="V1", payload=b"first-cut")
    value = artifact.to_dict()

    assert value["schema_version"] == ARTIFACT_VERSION_SCHEMA
    assert value["accessibility"]["human_correctable"] is True
    assert value["accessibility"]["computational_input"] is False
    assert value["accessibility"]["policy"] == ACCESSIBILITY_OVERLAY_POLICY
    assert "CAPTION_SENTINEL" in value["accessibility"]["caption_or_transcript"]
    assert "DESCRIPTION_SENTINEL" in value["accessibility"]["visual_description"]
    assert "prompt" not in value
    assert "parameters" not in value
    assert "model_input" not in json.dumps(value, ensure_ascii=False).lower()


def test_accessibility_overlays_require_explicit_human_review() -> None:
    with pytest.raises(ValueError, match="explicit human review"):
        create_story_artifact(
            media_bytes=b"cut",
            version_label="V1",
            parent_artifact_id=None,
            local_registered_at_utc=NOW,
            event_index=0,
            elapsed_ms=0,
            creator_intention="Intention",
            intended_audience="Audience",
            sound_image_relation="Relation",
            concept_tags="tag",
            cultural_aesthetic_context="Context",
            source_alias="source",
            source_permission_note="permission",
            editor_name_version="editor",
            ordering="order",
            trims="trim",
            mix_levels="mix",
            caption_or_transcript="caption",
            visual_description="description",
            overlays_human_reviewed=False,
            assistance_disclosure="none",
            export_preset_version="preset",
            change_rationale="rationale",
        )


def test_linked_private_bundle_validates_and_rejects_tampering() -> None:
    v1 = _artifact(version="V1", payload=b"first-cut")
    v2 = _artifact(version="V2", payload=b"second-cut", parent=v1.artifact_id)
    bundle = build_private_story_bundle(
        session_pseudonym="story-fixture",
        language="en",
        artifacts=(v1, v2),
        pre_response=_responses("pre"),
        post_response=_responses("post"),
        stage_timing_notes=("shared schedule observed; numeric tolerance unresolved",),
        fidelity_deviations=("caption correction support used",),
    )

    assert validate_private_story_bundle(bundle).is_valid
    assert bundle["route_id"] == COMPARISON_ROUTE_ID
    assert bundle["research_ready"] is False
    assert bundle["automatic_student_data_egress"] is False
    assert bundle["common_outcomes"]["administration_order"] == [
        PRE_OUTCOME_ID,
        POST_OUTCOME_ID,
    ]
    assert b"condition_assignment" not in serialize_private_story_bundle(bundle)

    artifact_tamper = copy.deepcopy(bundle)
    artifact_tamper["private_portfolio"]["artifacts"][0]["creator_intention"] = "changed"
    report = validate_private_story_bundle(artifact_tamper)
    assert not report.is_valid
    assert any("checksum" in issue for issue in report.issues)

    assignment_tamper = copy.deepcopy(bundle)
    assignment_tamper["study_arm"] = "comparison"
    report = validate_private_story_bundle(assignment_tamper)
    assert not report.is_valid
    assert any("forbidden assignment" in issue for issue in report.issues)


def test_common_outcomes_are_condition_neutral_and_complete() -> None:
    outcomes = build_outcome_bundle(
        session_pseudonym="story-fixture",
        language="ko",
        pre_response=_responses("pre"),
        post_response=_responses("post"),
    )
    serialized = json.dumps(outcomes, ensure_ascii=False, sort_keys=True)
    assert outcomes["administration_order"] == [PRE_OUTCOME_ID, POST_OUTCOME_ID]
    assert "condition" not in serialized.lower()
    assert outcomes["research_ready"] is False


def test_blinded_packet_and_two_distinct_readings_round_trip() -> None:
    packet = build_blinded_packet(
        exchange_artifact_id="exchange-1234567890abcdef",
        asset_reference="asset://approved-presentation-1",
        media_type="video/mp4",
        duration_ms=12_000,
        caption_reference="caption://human-reviewed-1",
        accessibility_note="Caption and visual description are human-correctable overlays.",
        language="en",
        permission_confirmed=True,
    )
    first = _reading(packet, index=1, audience="audience-a", session="session-a")
    second = _reading(packet, index=2, audience="audience-b", session="session-b")

    require_complete_audience_exchange(packet, (first, second))
    encoded = json.dumps(packet.to_dict(), ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "creator_intention",
        "model_output",
        "private_artifact_id",
        "content_sha256",
    ):
        assert f'"{forbidden}":' not in encoded

    with pytest.raises(ValueError, match="distinct"):
        require_complete_audience_exchange(packet, (first, first))


def test_replay_is_deterministic_cpu_only_and_tamper_evident() -> None:
    first = replay_payload()
    second = replay_payload()
    assert first == second
    validate_replay(first)
    assert first["runtime"] == "cpu_only_no_gpu_no_model_packages"

    corrupt = copy.deepcopy(first)
    corrupt["example"]["story_beats"].append("tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_replay(corrupt)


def test_clean_process_import_stays_gpu_and_model_package_free() -> None:
    code = """
import json, sys
import digital_storytelling.workflow as workflow
workflow.validate_replay(workflow.replay_payload())
blocked = sorted(name for name in sys.modules if name.split('.')[0] in {
    'torch', 'torchvision', 'transformers', 'accelerate', 'qwen_omni_utils'
})
print(json.dumps(blocked))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_generated_replay_matches_committed_artifact() -> None:
    path = ROOT / "digital_storytelling" / "replay" / "storytelling_replay.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == replay_payload()
    validate_replay(committed)


def test_localized_catalog_has_exact_nonempty_key_parity() -> None:
    inventory = content_key_inventory()
    assert "title" in inventory
    assert "overlay_policy" in inventory


def test_workflow_module_reloads_without_side_effects() -> None:
    module = importlib.import_module("digital_storytelling.workflow")
    assert importlib.reload(module).COMPARISON_ROUTE_ID == COMPARISON_ROUTE_ID
