"""Pure helpers for the production-matched digital-storytelling classroom route.

The module depends inward on ``curriculum_common`` and the Python standard library
only.  It deliberately contains no model client, processor, or GPU dependency.
Accessibility captions and descriptions are stored as human-correctable display
overlays; they are never transformed into computational inputs by this route.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

from curriculum_common.audience_packets import (
    AUDIENCE_PACKET_SCHEMA_VERSION,
    AUDIENCE_SHARING_PERMISSION,
    PACKET_FORBIDDEN_FIELDS,
    AudiencePacket,
    AudienceReading,
    RevealState,
    presentation_asset_checksum,
    validate_audience_exchange,
)
from curriculum_common.export_validation import validate_private_portfolio
from curriculum_common.portfolio_export import (
    build_private_portfolio,
    serialize_private_portfolio,
)
from curriculum_common.production_manifest import (
    ARTIFACT_VERSION_SCHEMA,
    ArtifactVersion,
    EditDecisionManifest,
    create_artifact_version,
)
from curriculum_common.session_records import new_session, teaching_mode


COMPARISON_ROUTE_ID = "matched-comparison/1.0.0"
PRIVATE_BUNDLE_SCHEMA = "storytelling-private-bundle/1.0.0"
OUTCOME_RESPONSE_SCHEMA = "common-outcome-response/1.0.0"
PRE_OUTCOME_ID = "critical-ai-reasoning-pre/0.1.0-pilot"
POST_OUTCOME_ID = "critical-ai-reasoning-post/0.1.0-pilot"
ARTIFACT_RUBRIC_ID = "audiovisual-intentionality-rubric/0.1.0-pilot"
FLEXIBILITY_RUBRIC_ID = "creative-flexibility-rubric/0.1.0-pilot"
REPLAY_SCHEMA = "storytelling-replay/1.0.0"

ACCESSIBILITY_OVERLAY_POLICY = (
    "Human-correctable captions and visual descriptions are display/share overlays "
    "only. This model-free route never converts them into computational inputs."
)

MATCHED_DIMENSIONS = (
    "brief",
    "activity_map",
    "time",
    "hardware",
    "source_budget",
    "editor_export",
    "process",
    "help_contact",
    "audience",
    "outcomes",
    "accessibility",
    "cognitive_demand",
)

ROUTE_STAGES = (
    "orient",
    "common_pre_task",
    "plan_first_cut",
    "register_v1",
    "story_structure_activity",
    "audience_exchange",
    "revise_v2",
    "common_post_task",
    "private_export",
)

ROUTE_MANIFEST: Mapping[str, Any] = {
    "route_id": COMPARISON_ROUTE_ID,
    "structural_status": "pass",
    "concrete_fidelity_status": "blocked_pending_human_freeze",
    "activity_sequence": list(ROUTE_STAGES),
    "matched_dimensions": {
        name: {
            "value": None,
            "tolerance": None,
            "approval_status": "unresolved",
        }
        for name in MATCHED_DIMENSIONS
    },
    "numeric_time_budget_minutes": None,
    "time_budget_status": "use_facilitator_matched_schedule_pending_pilot_freeze",
    "research_ready": False,
}


CONTENT: Mapping[str, Mapping[str, str]] = {
    "en": {
        "title": "Digital storytelling studio",
        "question": "How do sequence, sound, image, pacing, and context shape a viewer's reading of an audiovisual story?",
        "route_note": "Follow the facilitator's shared schedule and production brief. Numeric stage limits remain pending a matched rehearsal and human approval.",
        "teaching_mode": "Teaching",
        "teaching_caption": "default and fail-safe; no research destination",
        "egress": "Manual only",
        "egress_caption": "private download or permission-authorized audience packet",
        "release_status": "Structural preview",
        "release_caption": "concrete fidelity and release approvals remain unresolved",
        "section_prepare": "1. Prepare your story",
        "stage_orient": "1.1 Orient — Make, read, revise",
        "orient_body": "Create one accessible first cut, study how story form changes a reading, gather two independent blinded audience readings, revise a second cut, and save the common post-task. Required controls use native labels and remain keyboard operable.",
        "boundary": "Uploads and session state are processed inside the hosted Molab session/container, not solely on your device. Nothing is sent automatically to an instructor or outside service.",
        "pre_heading": "1.2 Common pre-task — Reason from limited evidence",
        "pre_prompt": "Before making, explain what you would check when an automated caption sounds plausible but may miss how sound and image work together.",
        "pre_evidence_label": "Evidence you would seek",
        "pre_alternative_label": "A plausible alternative explanation",
        "pre_limit_label": "What the caption alone cannot establish",
        "pre_submit": "Commit pre-task response",
        "section_make": "2. Make and register",
        "stage_plan": "2.1 Plan — First cut",
        "plan_body": "Use the facilitator-approved brief, source pool, editor route, export preset, accessibility support, and help policy. If any shared value is missing, record a deviation rather than silently substituting it.",
        "v1_upload": "First-cut audiovisual file",
        "intention": "Creator intention",
        "audience": "Intended audience",
        "relation": "Intended sound–image relationship",
        "tags": "Concept tags separated by commas",
        "context": "Cultural or aesthetic context",
        "source_alias": "Approved source alias",
        "source_rights": "Source permission or provenance note",
        "editor": "Facilitator-approved editor and version",
        "ordering": "Clip order or scene sequence",
        "trims": "Trim decisions",
        "mix": "Sound mix or level decisions",
        "caption": "Human-correctable caption or transcript",
        "description": "Human-correctable visual description",
        "overlay_review": "I reviewed these accessibility overlays and they describe the shared presentation.",
        "assistance": "Human or tool assistance disclosure",
        "export_preset": "Facilitator-approved export preset",
        "change_rationale_v1": "Why this first cut answers the production brief",
        "v1_submit": "Register first cut (V1)",
        "v1_waiting": "Register V1 to continue.",
        "v1_complete": "V1 is registered with the shared artifact schema.",
        "overlay_policy": ACCESSIBILITY_OVERLAY_POLICY,
        "section_activity": "3. Story form, audience, and revision",
        "stage_structure": "3.1 Explore — Story structure",
        "structure_body": "Work with story form rather than system internals. Identify one turning point, test an alternative sequence or sound role, and explain what a viewer could reasonably read differently.",
        "turning_point": "Turning point or narrative beat",
        "alternative_sequence": "Alternative sequence or pacing choice",
        "sound_role": "Role of sound in the story",
        "expected_reading": "Expected change in a viewer's reading",
        "structure_submit": "Commit story-structure note",
        "stage_audience": "3.2 Exchange — Independent audience readings",
        "audience_body": "Create a permission-authorized blinded packet, then import at least two distinct independent readings. Two readings are the classroom minimum, never a research sampling justification.",
        "asset_reference": "Permission-approved presentation reference",
        "media_type": "Media type",
        "duration_ms": "Duration in milliseconds",
        "caption_reference": "Caption or transcript reference",
        "sharing_permission": "I have permission to share this presentation package for the classroom audience activity.",
        "packet_submit": "Create blinded audience packet",
        "packet_waiting": "A packet is created only after explicit sharing permission.",
        "packet_ready": "Blinded packet ready for manual download.",
        "packet_download": "Download blinded audience packet",
        "audience_packet_upload": "Blinded presentation packet",
        "audience_readings_upload": "At least two audience-reading JSON files",
        "audience_import_submit": "Validate audience readings",
        "audience_waiting": "Creator context remains private until two blinded readings validate.",
        "audience_complete": "Two or more distinct blinded readings validated.",
        "disagreement": "Where the audience readings agree, diverge, or leave a reading unrepresented",
        "disagreement_context": "How culture, accessibility, context, or label choice may shape the difference",
        "disagreement_submit": "Commit disagreement reflection",
        "stage_revise": "3.3 Revise — Second cut (V2)",
        "v2_upload": "Revised audiovisual file",
        "v2_change": "Evidence-triggered revision and rationale",
        "v2_submit": "Register linked second cut (V2)",
        "v2_waiting": "Complete the structure note and audience reflection before V2.",
        "v2_complete": "V2 is linked to V1; V1 remains unchanged.",
        "section_finish": "4. Common outcome and private export",
        "post_heading": "4.1 Complete the common post-task",
        "post_prompt": "After making and audience exchange, explain what evidence you would now seek when an automated caption sounds plausible but may miss how sound and image work together.",
        "post_evidence_label": "Evidence you would seek now",
        "post_alternative_label": "A rival account you would keep open",
        "post_limit_label": "A claim you would not make from the caption alone",
        "post_submit": "Commit post-task response",
        "timing_label": "Observed stage timing or schedule deviation",
        "help_label": "Help, tool, source, accessibility, or disruption deviation",
        "export_heading": "Private learning bundle",
        "export_body": "The download contains the common private portfolio, linked V1/V2 artifact manifests, common outcome responses, and append-only fidelity notes. It contains no student-facing allocation field and is not a research submission.",
        "export_download": "Download private storytelling bundle",
        "export_waiting": "Complete V1, V2, and both common tasks before export.",
        "replay_heading": "Saved classroom example",
        "replay_body": "The checked-in example is deterministic, CPU-only, teaching-only, and invokes no model or GPU package. It is practice evidence, not student research data.",
    },
    "ko": {
        "title": "디지털 스토리텔링 스튜디오",
        "question": "순서, 소리, 이미지, 속도, 맥락은 시청자의 시청각 이야기 해석을 어떻게 형성할까요?",
        "route_note": "교수가 제시한 공통 일정과 제작 브리프를 따르세요. 단계별 수치 제한은 비교 리허설과 사람의 승인 전까지 확정되지 않았습니다.",
        "teaching_mode": "교육",
        "teaching_caption": "기본 안전 모드이며 연구 전송 목적지가 없습니다",
        "egress": "수동만 허용",
        "egress_caption": "개인 다운로드 또는 권한이 확인된 관객 패킷",
        "release_status": "구조 미리보기",
        "release_caption": "구체적 충실도와 배포 승인은 아직 해결되지 않았습니다",
        "section_prepare": "1. 이야기 준비",
        "stage_orient": "1.1 방향 잡기 — 만들고, 읽고, 수정하기",
        "orient_body": "접근 가능한 1차 편집본을 만들고, 이야기 형식이 해석을 바꾸는 방식을 살펴본 뒤, 서로 독립적인 익명 관객 해석 두 개를 모아 2차 편집본을 수정하고 공통 사후 과제를 저장합니다. 필수 조작 요소에는 기본 레이블이 있으며 키보드로 사용할 수 있습니다.",
        "boundary": "업로드와 세션 상태는 학생 기기에서만이 아니라 호스팅된 Molab 세션/컨테이너 안에서 처리됩니다. 어떤 자료도 교수나 외부 서비스로 자동 전송되지 않습니다.",
        "pre_heading": "1.2 공통 사전 과제 — 제한된 근거로 추론하기",
        "pre_prompt": "제작 전에, 자동 생성 캡션이 그럴듯하지만 소리와 이미지의 관계를 놓칠 수 있을 때 어떤 근거를 확인할지 설명하세요.",
        "pre_evidence_label": "찾아볼 근거",
        "pre_alternative_label": "가능한 대안 설명",
        "pre_limit_label": "캡션만으로 확정할 수 없는 것",
        "pre_submit": "사전 과제 응답 저장",
        "section_make": "2. 만들고 등록하기",
        "stage_plan": "2.1 계획 — 1차 편집본",
        "plan_body": "교수가 승인한 브리프, 소스 묶음, 편집 경로, 내보내기 설정, 접근성 지원, 도움 정책을 사용하세요. 공통 값이 없으면 임의로 대체하지 말고 편차를 기록하세요.",
        "v1_upload": "1차 시청각 파일",
        "intention": "창작 의도",
        "audience": "예상 관객",
        "relation": "의도한 소리–이미지 관계",
        "tags": "쉼표로 구분한 개념 태그",
        "context": "문화적 또는 미학적 맥락",
        "source_alias": "승인된 소스 별칭",
        "source_rights": "소스 권한 또는 출처 메모",
        "editor": "교수가 승인한 편집기와 버전",
        "ordering": "클립 순서 또는 장면 배열",
        "trims": "자르기 결정",
        "mix": "사운드 믹스 또는 레벨 결정",
        "caption": "사람이 수정할 수 있는 캡션 또는 대본",
        "description": "사람이 수정할 수 있는 시각 설명",
        "overlay_review": "이 접근성 오버레이를 검토했으며 공유 프레젠테이션을 설명합니다.",
        "assistance": "사람 또는 도구의 도움 공개",
        "export_preset": "교수가 승인한 내보내기 설정",
        "change_rationale_v1": "이 1차 편집본이 제작 브리프에 답하는 방식",
        "v1_submit": "1차 편집본(V1) 등록",
        "v1_waiting": "계속하려면 V1을 등록하세요.",
        "v1_complete": "V1이 공통 아티팩트 스키마로 등록되었습니다.",
        "overlay_policy": "사람이 수정할 수 있는 캡션과 시각 설명은 표시/공유용 오버레이일 뿐입니다. 이 무모델 경로는 이를 계산 입력으로 변환하지 않습니다.",
        "section_activity": "3. 이야기 형식, 관객, 수정",
        "stage_structure": "3.1 탐색 — 이야기 구조",
        "structure_body": "시스템 내부가 아니라 이야기 형식을 다룹니다. 전환점 하나를 찾고, 대안적 순서나 사운드 역할을 시험하며, 시청자가 무엇을 다르게 해석할 수 있는지 설명하세요.",
        "turning_point": "전환점 또는 서사 비트",
        "alternative_sequence": "대안적 순서 또는 속도 선택",
        "sound_role": "이야기에서 소리의 역할",
        "expected_reading": "예상되는 시청자 해석의 변화",
        "structure_submit": "이야기 구조 메모 저장",
        "stage_audience": "3.2 교환 — 독립적인 관객 해석",
        "audience_body": "권한이 확인된 익명 패킷을 만든 뒤 서로 다른 독립 해석을 두 개 이상 가져오세요. 두 해석은 수업 활동의 최소치일 뿐 연구 표본의 근거가 아닙니다.",
        "asset_reference": "권한이 확인된 프레젠테이션 참조",
        "media_type": "미디어 유형",
        "duration_ms": "재생 시간(밀리초)",
        "caption_reference": "캡션 또는 대본 참조",
        "sharing_permission": "수업 관객 활동을 위해 이 프레젠테이션 패키지를 공유할 권한이 있습니다.",
        "packet_submit": "익명 관객 패킷 만들기",
        "packet_waiting": "명시적 공유 권한을 확인한 뒤에만 패킷이 만들어집니다.",
        "packet_ready": "익명 패킷을 수동으로 다운로드할 수 있습니다.",
        "packet_download": "익명 관객 패킷 다운로드",
        "audience_packet_upload": "익명 프레젠테이션 패킷",
        "audience_readings_upload": "관객 해석 JSON 파일 두 개 이상",
        "audience_import_submit": "관객 해석 검증",
        "audience_waiting": "익명 해석 두 개가 검증될 때까지 창작자 맥락은 비공개로 유지됩니다.",
        "audience_complete": "서로 다른 익명 해석 두 개 이상이 검증되었습니다.",
        "disagreement": "관객 해석이 일치하거나 달라지거나 빠뜨린 관점",
        "disagreement_context": "문화, 접근성, 맥락, 레이블 선택이 차이에 미칠 수 있는 영향",
        "disagreement_submit": "불일치 성찰 저장",
        "stage_revise": "3.3 수정 — 2차 편집본(V2)",
        "v2_upload": "수정한 시청각 파일",
        "v2_change": "근거에 따라 수정한 내용과 이유",
        "v2_submit": "연결된 2차 편집본(V2) 등록",
        "v2_waiting": "V2 전에 이야기 구조 메모와 관객 성찰을 완료하세요.",
        "v2_complete": "V2가 V1에 연결되었고 V1은 변경되지 않았습니다.",
        "section_finish": "4. 공통 성과와 개인 내보내기",
        "post_heading": "4.1 공통 사후 과제 완료",
        "post_prompt": "제작과 관객 교환 후, 자동 생성 캡션이 그럴듯하지만 소리와 이미지의 관계를 놓칠 수 있을 때 이제 어떤 근거를 확인할지 설명하세요.",
        "post_evidence_label": "이제 찾아볼 근거",
        "post_alternative_label": "열어 둘 경쟁 설명",
        "post_limit_label": "캡션만으로 주장하지 않을 것",
        "post_submit": "사후 과제 응답 저장",
        "timing_label": "관찰한 단계 시간 또는 일정 편차",
        "help_label": "도움, 도구, 소스, 접근성, 중단 관련 편차",
        "export_heading": "개인 학습 번들",
        "export_body": "다운로드에는 공통 개인 포트폴리오, 연결된 V1/V2 아티팩트 명세, 공통 성과 응답, 추가 전용 충실도 메모가 들어갑니다. 학생에게 배정 정보를 노출하지 않으며 연구 제출물이 아닙니다.",
        "export_download": "개인 스토리텔링 번들 다운로드",
        "export_waiting": "내보내기 전에 V1, V2, 공통 사전·사후 과제를 완료하세요.",
        "replay_heading": "저장된 수업 예시",
        "replay_body": "체크인된 예시는 결정론적이며 CPU 전용이고 교육 목적으로만 사용되며 어떤 모델이나 GPU 패키지도 호출하지 않습니다. 연습 근거일 뿐 학생 연구 데이터가 아닙니다.",
    },
}


@dataclass(frozen=True)
class StoryBundleValidation:
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if self.issues:
            raise ValueError("; ".join(self.issues))


def content_key_inventory() -> tuple[str, ...]:
    """Return the frozen student-text key inventory after parity validation."""

    languages = tuple(CONTENT)
    if languages != ("en", "ko"):
        raise ValueError("the classroom catalog must contain exactly en and ko")
    english = set(CONTENT["en"])
    korean = set(CONTENT["ko"])
    if english != korean:
        missing_en = sorted(korean - english)
        missing_ko = sorted(english - korean)
        raise ValueError(f"content key mismatch: missing_en={missing_en}; missing_ko={missing_ko}")
    empty = sorted(
        f"{language}.{key}"
        for language, values in CONTENT.items()
        for key, value in values.items()
        if not value.strip()
    )
    if empty:
        raise ValueError("empty localized text: " + ", ".join(empty))
    return tuple(sorted(english))


def content_for(language: str) -> Mapping[str, str]:
    content_key_inventory()
    if language not in CONTENT:
        raise ValueError("language must be 'en' or 'ko'")
    return CONTENT[language]


def content_sha256(payload: bytes) -> str:
    if not payload:
        raise ValueError("the audiovisual file must not be empty")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _split_text(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError("at least one non-empty item is required")
    return parts


def create_story_artifact(
    *,
    media_bytes: bytes,
    version_label: str,
    parent_artifact_id: str | None,
    local_registered_at_utc: str,
    event_index: int,
    elapsed_ms: int,
    creator_intention: str,
    intended_audience: str,
    sound_image_relation: str,
    concept_tags: str | Sequence[str],
    cultural_aesthetic_context: str,
    source_alias: str,
    source_permission_note: str,
    editor_name_version: str,
    ordering: str,
    trims: str,
    mix_levels: str,
    caption_or_transcript: str,
    visual_description: str,
    overlays_human_reviewed: bool,
    assistance_disclosure: str,
    export_preset_version: str,
    change_rationale: str,
) -> ArtifactVersion:
    """Create one immutable common-schema story artifact.

    Caption and description values are stored only inside the accessibility map.
    There is intentionally no model-input or prompt argument.
    """

    if not overlays_human_reviewed:
        raise ValueError("accessibility overlays require explicit human review")
    tags = _split_text(concept_tags) if isinstance(concept_tags, str) else tuple(concept_tags)
    if not caption_or_transcript.strip() or not visual_description.strip():
        raise ValueError("caption/transcript and visual description are required")
    manifest = EditDecisionManifest(
        editor_name_version=editor_name_version.strip(),
        source_assets=(
            {
                "source_alias": source_alias.strip(),
                "permission_or_provenance": source_permission_note.strip(),
            },
        ),
        ordering=(_required_text("ordering", ordering),),
        trims=({"decision": _required_text("trims", trims)},),
        mix_levels=({"decision": _required_text("mix_levels", mix_levels)},),
        accessibility_work={
            "caption_or_transcript": caption_or_transcript.strip(),
            "visual_description": visual_description.strip(),
            "human_correctable": True,
            "human_reviewed": True,
            "usage": "display_and_permission_authorized_sharing_only",
            "computational_input": False,
            "policy": ACCESSIBILITY_OVERLAY_POLICY,
        },
        assistance_disclosure={"disclosure": assistance_disclosure.strip()},
        export_preset_version=export_preset_version.strip(),
    )
    return create_artifact_version(
        content_sha256=content_sha256(media_bytes),
        version_label=version_label,
        parent_artifact_id=parent_artifact_id,
        local_registered_at_utc=local_registered_at_utc,
        event_index=event_index,
        elapsed_ms=elapsed_ms,
        media_facts={
            "byte_length": len(media_bytes),
            "media_type": "video",
            "inspection_status": "human_confirmed_classroom_upload",
        },
        creator_intention=_required_text("creator_intention", creator_intention),
        intended_audience=_required_text("intended_audience", intended_audience),
        sound_image_relation=_required_text("sound_image_relation", sound_image_relation),
        concept_tags=tags,
        cultural_aesthetic_context=_required_text(
            "cultural_aesthetic_context", cultural_aesthetic_context
        ),
        source_license_provenance={
            "source_alias": _required_text("source_alias", source_alias),
            "permission_or_provenance": _required_text(
                "source_permission_note", source_permission_note
            ),
            "release_approved": False,
        },
        edit_manifest=manifest,
        change_rationale=_required_text("change_rationale", change_rationale),
        processing_boundary=(
            "hosted Molab session/container; user-initiated private download only"
        ),
    )


def build_blinded_packet(
    *,
    exchange_artifact_id: str,
    asset_reference: str,
    media_type: str,
    duration_ms: int,
    caption_reference: str,
    accessibility_note: str,
    language: str,
    permission_confirmed: bool,
) -> AudiencePacket:
    """Build the frozen audience allowlist without creator/private fields."""

    if not permission_confirmed:
        raise ValueError("classroom audience sharing permission must be confirmed")
    if not exchange_artifact_id.startswith("exchange-"):
        raise ValueError("exchange_artifact_id must be an opaque exchange identifier")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError("duration_ms must be a positive integer")
    presentation_asset = {
        "asset_reference": _required_text("asset_reference", asset_reference),
        "media_type": _required_text("media_type", media_type),
        "duration_ms": duration_ms,
        "caption_reference": _required_text("caption_reference", caption_reference),
        "accessibility_note": _required_text("accessibility_note", accessibility_note),
        "language": _required_text("language", language),
    }
    packet = AudiencePacket(
        packet_schema_version=AUDIENCE_PACKET_SCHEMA_VERSION,
        exchange_artifact_id=exchange_artifact_id,
        presentation_asset=presentation_asset,
        presentation_checksum=presentation_asset_checksum(presentation_asset),
        permitted_accessibility_overlays=(
            "caption_or_transcript",
            "visual_description",
        ),
        excluded_field_attestation=tuple(sorted(PACKET_FORBIDDEN_FIELDS)),
        reveal_state=RevealState.BLINDED,
        protocol_deviation=None,
        sharing_permission_scope=AUDIENCE_SHARING_PERMISSION,
    )
    return AudiencePacket.from_mapping(packet.to_dict())


def require_complete_audience_exchange(
    packet: AudiencePacket,
    readings: Sequence[AudienceReading],
) -> None:
    report = validate_audience_exchange(packet, readings, minimum_required=2)
    if not report.is_complete:
        detail = "; ".join(issue.message for issue in report.issues)
        raise ValueError(detail or "two distinct blinded readings are required")


def build_outcome_bundle(
    *,
    session_pseudonym: str,
    language: str,
    pre_response: Mapping[str, str],
    post_response: Mapping[str, str],
) -> dict[str, Any]:
    """Build condition-neutral common pre/post response data."""

    expected = frozenset({"evidence", "alternative", "limit"})
    for stage, response in (("pre", pre_response), ("post", post_response)):
        if set(response) != expected:
            raise ValueError(f"{stage} response fields must be {sorted(expected)}")
        if any(not isinstance(value, str) or not value.strip() for value in response.values()):
            raise ValueError(f"{stage} response fields must contain non-empty text")
    if language not in CONTENT:
        raise ValueError("language must be 'en' or 'ko'")
    return {
        "schema_version": OUTCOME_RESPONSE_SCHEMA,
        "classification": "private formative learning response; not research data",
        "session_pseudonym": _required_text("session_pseudonym", session_pseudonym),
        "language": language,
        "administration_order": [PRE_OUTCOME_ID, POST_OUTCOME_ID],
        "responses": {
            "pre": {"instrument_id": PRE_OUTCOME_ID, **dict(pre_response)},
            "post": {"instrument_id": POST_OUTCOME_ID, **dict(post_response)},
        },
        "missingness": {"pre": "observed", "post": "observed"},
        "automatic_student_data_egress": False,
        "research_ready": False,
    }


def build_private_story_bundle(
    *,
    session_pseudonym: str,
    language: str,
    artifacts: Sequence[ArtifactVersion],
    pre_response: Mapping[str, str],
    post_response: Mapping[str, str],
    stage_timing_notes: Sequence[str],
    fidelity_deviations: Sequence[str],
) -> dict[str, Any]:
    """Wrap the common private portfolio without mutating its canonical bytes."""

    labels = tuple(artifact.version_label for artifact in artifacts)
    if labels != ("V1", "V2"):
        raise ValueError("the private bundle requires linked V1 and V2 artifacts")
    if artifacts[1].parent_artifact_id != artifacts[0].artifact_id:
        raise ValueError("V2 must preserve the immutable V1 parent link")
    log = new_session(session_pseudonym, decision=teaching_mode())
    private_portfolio = build_private_portfolio(
        log,
        artifact_versions=tuple(artifact.to_dict() for artifact in artifacts),
        boundary_disclosure=(
            "Session state is processed inside the hosted Molab session/container "
            "boundary, not solely on the student's device. This private download is "
            "user-initiated and is not research data."
        ),
    )
    validate_private_portfolio(private_portfolio).require_valid()
    outcome_bundle = build_outcome_bundle(
        session_pseudonym=session_pseudonym,
        language=language,
        pre_response=pre_response,
        post_response=post_response,
    )
    bundle = {
        "schema_version": PRIVATE_BUNDLE_SCHEMA,
        "classification": "private learning bundle; pseudonymous; not research data",
        "route_id": COMPARISON_ROUTE_ID,
        "route_release_status": "structural_preview_human_gates_unresolved",
        "private_portfolio": private_portfolio,
        "common_outcomes": outcome_bundle,
        "stage_timing_notes": [str(item) for item in stage_timing_notes if str(item).strip()],
        "fidelity_deviations": [str(item) for item in fidelity_deviations if str(item).strip()],
        "accessibility_overlay_policy": ACCESSIBILITY_OVERLAY_POLICY,
        "automatic_student_data_egress": False,
        "research_ready": False,
    }
    validate_private_story_bundle(bundle).require_valid()
    return bundle


def serialize_private_story_bundle(bundle: Mapping[str, Any]) -> bytes:
    validation = validate_private_story_bundle(bundle)
    validation.require_valid()
    return _canonical_bytes(bundle)


def validate_private_story_bundle(value: Mapping[str, Any]) -> StoryBundleValidation:
    issues: list[str] = []
    if value.get("schema_version") != PRIVATE_BUNDLE_SCHEMA:
        issues.append("unsupported private storytelling bundle schema")
    if value.get("route_id") != COMPARISON_ROUTE_ID:
        issues.append("unexpected route identifier")
    if value.get("automatic_student_data_egress") is not False:
        issues.append("automatic student-data egress must remain false")
    if value.get("research_ready") is not False:
        issues.append("the unresolved route cannot claim research readiness")
    portfolio = value.get("private_portfolio")
    if not isinstance(portfolio, Mapping):
        issues.append("private_portfolio must be a mapping")
    else:
        report = validate_private_portfolio(portfolio)
        issues.extend(f"private_portfolio.{issue.path}: {issue.message}" for issue in report.issues)
        artifacts = portfolio.get("artifacts")
        if not isinstance(artifacts, list):
            issues.append("private_portfolio.artifacts must be a list")
        else:
            try:
                loaded = tuple(ArtifactVersion.from_dict(item) for item in artifacts)
            except (TypeError, ValueError) as exc:
                issues.append(f"private_portfolio artifact schema: {exc}")
            else:
                if tuple(item.version_label for item in loaded) != ("V1", "V2"):
                    issues.append("private_portfolio must contain V1 then V2")
                elif loaded[1].parent_artifact_id != loaded[0].artifact_id:
                    issues.append("private_portfolio V2 parent link is invalid")
        expected_checksum = _portfolio_checksum(portfolio)
        if portfolio.get("checksum_sha256") != expected_checksum:
            issues.append("private_portfolio checksum does not match its canonical content")
    outcomes = value.get("common_outcomes")
    if not isinstance(outcomes, Mapping):
        issues.append("common_outcomes must be a mapping")
    else:
        issues.extend(_validate_outcomes(outcomes))
    forbidden = _find_forbidden_student_fields(
        value,
        frozenset({"condition_assignment", "allocation", "allocated_arm", "study_arm"}),
    )
    if forbidden:
        issues.append("student bundle contains forbidden assignment fields: " + ", ".join(forbidden))
    return StoryBundleValidation(tuple(issues))


def replay_payload() -> dict[str, Any]:
    """Return the deterministic CPU-only classroom example."""

    payload = {
        "schema_version": REPLAY_SCHEMA,
        "route_id": COMPARISON_ROUTE_ID,
        "provenance": "saved classroom replay; teaching-only example",
        "runtime": "cpu_only_no_gpu_no_model_packages",
        "generator": "python -m digital_storytelling.generate_replay",
        "example": {
            "story_beats": ["arrival", "interruption", "return"],
            "sound_role": "the recurring sound marks a change in viewpoint",
            "alternative_sequence": ["interruption", "arrival", "return"],
            "accessibility_alternative": (
                "Text summary: a repeated sound joins three still-image story beats."
            ),
        },
    }
    payload["payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def validate_replay(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != REPLAY_SCHEMA:
        raise ValueError("unsupported replay schema")
    supplied = value.get("payload_sha256")
    candidate = dict(value)
    candidate.pop("payload_sha256", None)
    expected = hashlib.sha256(_canonical_bytes(candidate)).hexdigest()
    if supplied != expected:
        raise ValueError("replay checksum mismatch")
    if value.get("runtime") != "cpu_only_no_gpu_no_model_packages":
        raise ValueError("comparison replay must remain CPU-only and model-free")


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _portfolio_checksum(portfolio: Mapping[str, Any]) -> str:
    candidate = dict(portfolio)
    candidate.pop("checksum_sha256", None)
    return hashlib.sha256(_canonical_bytes(candidate)).hexdigest()


def _validate_outcomes(value: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    if value.get("schema_version") != OUTCOME_RESPONSE_SCHEMA:
        issues.append("unsupported common outcome schema")
    if value.get("administration_order") != [PRE_OUTCOME_ID, POST_OUTCOME_ID]:
        issues.append("common outcome administration order changed")
    responses = value.get("responses")
    if not isinstance(responses, Mapping):
        issues.append("common outcome responses must be a mapping")
    else:
        for stage, instrument_id in (("pre", PRE_OUTCOME_ID), ("post", POST_OUTCOME_ID)):
            response = responses.get(stage)
            if not isinstance(response, Mapping) or response.get("instrument_id") != instrument_id:
                issues.append(f"{stage} outcome instrument identifier changed")
            elif any(not str(response.get(key, "")).strip() for key in ("evidence", "alternative", "limit")):
                issues.append(f"{stage} outcome response is incomplete")
    return tuple(issues)


def _find_forbidden_student_fields(
    value: Any,
    forbidden: frozenset[str],
    path: str = "$",
) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in forbidden:
                found.append(child_path)
            found.extend(_find_forbidden_student_fields(child, forbidden, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_student_fields(child, forbidden, f"{path}[{index}]"))
    return tuple(found)


__all__ = [
    "ACCESSIBILITY_OVERLAY_POLICY",
    "ARTIFACT_RUBRIC_ID",
    "ARTIFACT_VERSION_SCHEMA",
    "COMPARISON_ROUTE_ID",
    "CONTENT",
    "FLEXIBILITY_RUBRIC_ID",
    "MATCHED_DIMENSIONS",
    "POST_OUTCOME_ID",
    "PRE_OUTCOME_ID",
    "PRIVATE_BUNDLE_SCHEMA",
    "REPLAY_SCHEMA",
    "ROUTE_MANIFEST",
    "ROUTE_STAGES",
    "StoryBundleValidation",
    "build_blinded_packet",
    "build_outcome_bundle",
    "build_private_story_bundle",
    "content_for",
    "content_key_inventory",
    "content_sha256",
    "create_story_artifact",
    "replay_payload",
    "require_complete_audience_exchange",
    "serialize_private_portfolio",
    "serialize_private_story_bundle",
    "utc_now_text",
    "validate_private_story_bundle",
    "validate_replay",
]

