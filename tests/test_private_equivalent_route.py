from __future__ import annotations

import json
from pathlib import Path

import pytest

from curriculum_common.pilot_profile import (
    PEER_EXCHANGE_ROUTE,
    PRIVATE_EQUIVALENT_ROUTE,
    resolve_exchange_route,
)
from digital_storytelling.workflow import (
    build_blinded_packet,
    build_private_story_bundle,
    create_story_artifact,
    validate_private_story_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
TREATMENT = ROOT / "avllm_interpretability" / "CTP49906_avllm_molab.py"
COMPARISON = ROOT / "digital_storytelling" / "CTP49906_storytelling_molab.py"


def _artifact(version: str, payload: bytes, parent: str | None = None):
    return create_story_artifact(
        media_bytes=payload,
        version_label=version,
        parent_artifact_id=parent,
        local_registered_at_utc="2026-07-22T00:00:00Z",
        event_index=0 if version == "V1" else 1,
        elapsed_ms=0,
        creator_intention="A situated artistic intention.",
        intended_audience="Classroom viewers",
        sound_image_relation="Sound reframes the image.",
        concept_tags="counterpoint, context",
        cultural_aesthetic_context="One situated reading, not a universal claim.",
        source_alias="student-owned-private-source",
        source_permission_note="Student retains the media; no peer sharing granted.",
        editor_name_version="no-cost-editor/pilot",
        ordering="opening, change, return",
        trims="bounded trims",
        mix_levels="audible dialogue",
        caption_or_transcript="Human-correctable caption",
        visual_description="Human-correctable visual description",
        overlays_human_reviewed=True,
        assistance_disclosure="Instructor example only",
        export_preset_version="common-export/pilot",
        change_rationale="Private comparison evidence prompted revision.",
    )


def _response(prefix: str) -> dict[str, str]:
    return {
        "evidence": f"{prefix} evidence",
        "alternative": f"{prefix} alternative",
        "limit": f"{prefix} limit",
    }


def test_private_synthetic_and_instructor_routes_match_peer_checkpoint_without_penalty() -> None:
    peer = resolve_exchange_route(
        PEER_EXCHANGE_ROUTE,
        permission_confirmed=True,
    )
    for source in ("synthetic_example", "instructor_example"):
        private = resolve_exchange_route(
            PRIVATE_EQUIVALENT_ROUTE,
            permission_confirmed=False,
            evidence_source=source,
        )
        assert private.checkpoint_id == peer.checkpoint_id
        assert private.audience_exchange_required is False
        assert private.no_penalty is peer.no_penalty is True
        assert private.research_ready is peer.research_ready is False


def test_peer_route_fails_closed_without_permission() -> None:
    with pytest.raises(PermissionError, match="permission"):
        resolve_exchange_route(PEER_EXCHANGE_ROUTE, permission_confirmed=False)


def test_private_bundle_completes_without_peer_readings_and_records_equivalence() -> None:
    v1 = _artifact("V1", b"first")
    v2 = _artifact("V2", b"second", parent=v1.artifact_id)
    decision = resolve_exchange_route(
        PRIVATE_EQUIVALENT_ROUTE,
        permission_confirmed=False,
        evidence_source="instructor_example",
    )
    bundle = build_private_story_bundle(
        session_pseudonym="private-route-fixture",
        language="en",
        artifacts=(v1, v2),
        pre_response=_response("pre"),
        post_response=_response("post"),
        stage_timing_notes=(),
        fidelity_deviations=(),
        exchange_route=decision,
        audience_reading_count=0,
    )

    assert validate_private_story_bundle(bundle).is_valid
    assert bundle["learning_exchange"] == {
        "checkpoint_id": "compare_readings",
        "evidence_source": "instructor_example",
        "no_penalty": True,
        "research_ready": False,
    }
    assert bundle["private_portfolio"]["learning_exchange"]["no_penalty"] is True
    assert bundle["research_ready"] is False
    serialized = json.dumps(bundle, ensure_ascii=False, sort_keys=True)
    assert PRIVATE_EQUIVALENT_ROUTE not in serialized
    assert "student_reason" not in serialized


def test_peer_bundle_requires_two_permission_valid_readings() -> None:
    v1 = _artifact("V1", b"first")
    v2 = _artifact("V2", b"second", parent=v1.artifact_id)
    peer = resolve_exchange_route(PEER_EXCHANGE_ROUTE, permission_confirmed=True)

    with pytest.raises(ValueError, match="two distinct"):
        build_private_story_bundle(
            session_pseudonym="peer-route-fixture",
            language="ko",
            artifacts=(v1, v2),
            pre_response=_response("pre"),
            post_response=_response("post"),
            stage_timing_notes=(),
            fidelity_deviations=(),
            exchange_route=peer,
            audience_reading_count=1,
        )


def test_both_notebooks_make_peer_optional_and_private_route_no_penalty() -> None:
    for notebook in (TREATMENT, COMPARISON):
        source = notebook.read_text(encoding="utf-8")
        assert "resolve_exchange_route" in source
        assert "PEER_EXCHANGE_ROUTE" in source
        assert "PRIVATE_EQUIVALENT_ROUTE" in source
        lowered = source.lower()
        assert "optional" in lowered or "선택" in source
        assert "no penalty" in lowered or "불이익" in source


def test_private_choice_never_leaks_into_shareable_packet() -> None:
    packet = build_blinded_packet(
        exchange_artifact_id="exchange-private-boundary-fixture",
        asset_reference="asset://permission-approved-reference",
        media_type="video/mp4",
        duration_ms=1_000,
        caption_reference="caption://human-reviewed",
        accessibility_note="Human-correctable overlays are available.",
        language="en",
        permission_confirmed=True,
    )
    serialized = json.dumps(packet.to_dict(), ensure_ascii=False, sort_keys=True)

    assert PRIVATE_EQUIVALENT_ROUTE not in serialized
    assert "student_reason" not in serialized
    assert "instructor_example" not in serialized
    assert "synthetic_example" not in serialized
