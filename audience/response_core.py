"""Pure, GPU-free helpers for blinded classroom audience exchange.

The Marimo surface imports this module, while the command-line entry point keeps
packet validation and reading export testable without Marimo, CUDA, a model, or
network access.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curriculum_common.audience_packets import (  # noqa: E402
    AUDIENCE_READING_SCHEMA_VERSION,
    AUDIENCE_SHARING_PERMISSION,
    AudiencePacket,
    AudienceReading,
    RevealState,
    canonical_json_bytes,
    validate_audience_exchange,
)


DISPLAY_PACKET_FIELDS = frozenset(
    {
        "exchange_artifact_id",
        "presentation_asset",
        "presentation_checksum",
        "permitted_accessibility_overlays",
        "reveal_state",
        "protocol_deviation",
        "sharing_permission_scope",
    }
)

CLASSROOM_MINIMUM_NOTICE = (
    "Two distinct valid blinded readings are the classroom minimum for comparing "
    "interpretations. This minimum is a learning checkpoint, not evidence that the "
    "readings represent a population or justify a research claim."
)

AUDIENCE_CONTENT: dict[str, dict[str, str]] = {
    "en": {
        "language_name": "English",
        "title": "Blinded audience reading",
        "purpose": (
            "Describe what you notice before any creator explanation or machine "
            "label is shown. Your reading may differ from another viewer's."
        ),
        "boundary": (
            "Teaching mode only. The app reads a permission-valid packet in this "
            "session and creates a local download only when you press Download. "
            "It does not automatically send your response anywhere."
        ),
        "packet_heading": "1. Load the blinded presentation packet",
        "packet_upload": "Blinded packet JSON",
        "packet_submit": "Validate packet",
        "packet_ready": "Packet validated. Creator and machine fields are absent.",
        "response_heading": "2. Record an independent reading",
        "audience_pseudonym": "Audience pseudonym",
        "session_pseudonym": "Session pseudonym",
        "open_interpretation": "What do you think the work is expressing?",
        "sound_image_relation": "How do sound and image work together or apart?",
        "shared_tags": "Shared descriptive tags (comma-separated)",
        "self_tags": "Your own descriptive tags (comma-separated)",
        "accessibility": (
            "Any access barrier you noticed, or 'prefer not to answer'"
        ),
        "confidence": "What is uncertain or could support another reading?",
        "blindness": (
            "I have not been shown the creator explanation, condition, or machine labels"
        ),
        "response_submit": "Commit blinded reading",
        "download": "Download blinded reading JSON",
        "complete": (
            "Reading validated. Download it and return it through the classroom "
            "method named by your instructor."
        ),
    },
    "ko": {
        "language_name": "한국어",
        "title": "블라인드 청중 읽기",
        "purpose": (
            "창작자의 설명이나 기계 라벨을 보기 전에 무엇을 느끼고 해석했는지 "
            "기록하세요. 다른 관객의 읽기와 달라도 됩니다."
        ),
        "boundary": (
            "수업 모드 전용입니다. 이 앱은 권한이 확인된 패킷을 현재 세션에서 "
            "읽고, 다운로드 버튼을 눌렀을 때만 로컬 파일을 만듭니다. 응답을 "
            "어디에도 자동 전송하지 않습니다."
        ),
        "packet_heading": "1. 블라인드 제시 패킷 불러오기",
        "packet_upload": "블라인드 패킷 JSON",
        "packet_submit": "패킷 검증",
        "packet_ready": "패킷 검증 완료. 창작자 및 기계 필드는 포함되어 있지 않습니다.",
        "response_heading": "2. 독립적인 읽기 기록하기",
        "audience_pseudonym": "청중 가명",
        "session_pseudonym": "세션 가명",
        "open_interpretation": "이 작품이 무엇을 표현한다고 생각하나요?",
        "sound_image_relation": "소리와 이미지는 어떻게 함께 또는 따로 작동하나요?",
        "shared_tags": "공통 묘사 태그(쉼표로 구분)",
        "self_tags": "나만의 묘사 태그(쉼표로 구분)",
        "accessibility": "발견한 접근 장벽 또는 '응답하지 않음'",
        "confidence": "무엇이 불확실하며 다른 읽기를 가능하게 하나요?",
        "blindness": "창작자 설명, 조건, 기계 라벨을 보지 않았습니다",
        "response_submit": "블라인드 읽기 확정",
        "download": "블라인드 읽기 JSON 다운로드",
        "complete": (
            "읽기 검증 완료. 파일을 다운로드한 뒤 교사가 안내한 수업 방식으로 "
            "전달하세요."
        ),
    },
}


@dataclass(frozen=True)
class AudienceRevealBundle:
    """Post-import comparison data that exists only after the blind gate passes."""

    exchange_artifact_id: str
    reveal_state: str
    audience_readings: tuple[Mapping[str, Any], ...]
    creator_reading: str
    creator_tags: tuple[str, ...]
    model_labels: tuple[str, ...]
    disagreement_matrix: tuple[Mapping[str, Any], ...]
    reflection_prompt: str
    classroom_minimum_notice: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


def packet_for_display(packet: AudiencePacket) -> dict[str, Any]:
    """Return only fields approved for the separate blinded surface."""

    value = packet.to_dict()
    return {field: value[field] for field in sorted(DISPLAY_PACKET_FIELDS)}


def parse_json_object(data: bytes | str, *, max_bytes: int = 1_000_000) -> dict[str, Any]:
    """Parse one bounded JSON object while rejecting duplicate keys and NaN values."""

    raw = data if isinstance(data, bytes) else data.encode("utf-8")
    if len(raw) > max_bytes:
        raise ValueError(f"JSON input exceeds the {max_bytes}-byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON input must be UTF-8") from exc

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def _constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("input must contain a JSON object")
    return value


def build_reading(
    packet: AudiencePacket,
    response: Mapping[str, Any],
    *,
    audience_pseudonym: str,
    respondent_session_pseudonym: str,
    event_index: int,
    elapsed_ms: int,
    response_id: str | None = None,
    created_at_utc: str | None = None,
) -> AudienceReading:
    """Build and validate one appendable audience reading."""

    reading = {
        "schema_version": AUDIENCE_READING_SCHEMA_VERSION,
        "response_id": response_id or f"response-{uuid4()}",
        "exchange_artifact_id": packet.exchange_artifact_id,
        "audience_pseudonym": audience_pseudonym,
        "respondent_session_pseudonym": respondent_session_pseudonym,
        "local_created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_index": event_index,
        "elapsed_ms": elapsed_ms,
        "blindness_attestation": response.get("blindness_attestation"),
        "permission_scope": response.get("permission_scope"),
        "open_interpretation": response.get("open_interpretation"),
        "sound_image_relation": response.get("sound_image_relation"),
        "shared_tags": response.get("shared_tags"),
        "self_described_tags": response.get("self_described_tags"),
        "accessibility_barriers": response.get("accessibility_barriers"),
        "confidence_or_ambiguity": response.get("confidence_or_ambiguity"),
        "withdrawn": response.get("withdrawn", False),
    }
    return AudienceReading.from_mapping(reading, packet=packet)


def prepare_reveal_bundle(
    packet: AudiencePacket,
    readings: Iterable[AudienceReading | Mapping[str, Any]],
    *,
    creator_reading: str,
    creator_tags: Sequence[str],
    model_labels: Sequence[str],
) -> AudienceRevealBundle:
    """Validate imports before creator or machine readings can enter a reveal bundle."""

    normalized = tuple(
        item if isinstance(item, AudienceReading) else AudienceReading.from_mapping(item, packet=packet)
        for item in readings
    )
    report = validate_audience_exchange(packet, normalized)
    if not report.is_complete:
        detail = "; ".join(
            issue.message for issue in report.issues if issue.severity == "error"
        )
        raise ValueError(
            "reveal requires two distinct valid blinded readings"
            + (f": {detail}" if detail else "")
        )

    creator_text = creator_reading.strip()
    creator_values = _normalized_tags(creator_tags)
    model_values = _normalized_tags(model_labels)
    if not creator_text or not creator_values or not model_values:
        raise ValueError("creator reading, creator tags, and model labels are required at reveal")

    sources: dict[str, tuple[str, ...]] = {
        "creator": creator_values,
        "model": model_values,
    }
    for index, reading in enumerate(normalized, start=1):
        sources[f"audience_{index}"] = _normalized_tags(
            (*reading.shared_tags, *reading.self_described_tags)
        )
    all_labels = sorted(
        {label for values in sources.values() for label in values}, key=str.casefold
    )
    rows = tuple(
        {
            "label": label,
            **{source: label in values for source, values in sources.items()},
        }
        for label in all_labels
    )
    revealed = packet.transition_to(RevealState.REVEALED)
    return AudienceRevealBundle(
        exchange_artifact_id=packet.exchange_artifact_id,
        reveal_state=revealed.reveal_state.value,
        audience_readings=tuple(reading.to_dict() for reading in normalized),
        creator_reading=creator_text,
        creator_tags=creator_values,
        model_labels=model_values,
        disagreement_matrix=rows,
        reflection_prompt=(
            "Where do the creator, audience, and machine readings overlap or diverge? "
            "Name another reading that is not represented, then consider culture, "
            "accessibility, prompt wording, and training-data unknowns. Agreement is "
            "not correctness, and disagreement is not error."
        ),
        classroom_minimum_notice=CLASSROOM_MINIMUM_NOTICE,
    )


def _normalized_tags(values: Sequence[str]) -> tuple[str, ...]:
    unique: dict[str, str] = {}
    for value in values:
        cleaned = str(value).strip()
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    return tuple(unique[key] for key in sorted(unique))


def _load_object(path: Path) -> dict[str, Any]:
    return parse_json_object(path.read_bytes())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audience-pseudonym", required=True)
    parser.add_argument("--session-pseudonym", required=True)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--elapsed-ms", type=int, default=0)
    parser.add_argument("--response-id")
    parser.add_argument("--created-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = AudiencePacket.from_mapping(_load_object(args.packet))
    reading = build_reading(
        packet,
        _load_object(args.response),
        audience_pseudonym=args.audience_pseudonym,
        respondent_session_pseudonym=args.session_pseudonym,
        event_index=args.event_index,
        elapsed_ms=args.elapsed_ms,
        response_id=args.response_id,
        created_at_utc=args.created_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(reading.to_dict()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
