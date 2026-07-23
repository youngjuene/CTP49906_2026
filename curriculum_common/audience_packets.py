"""Frozen WP-0 contracts for blinded audience exchange.

This module deliberately uses only the Python standard library.  The audience
surface must stay GPU-free and must not depend on treatment-only code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


AUDIENCE_PACKET_SCHEMA_VERSION = "audience-packet/1.0.0"
AUDIENCE_READING_SCHEMA_VERSION = "audience-reading/1.0.0"
AUDIENCE_SHARING_PERMISSION = "classroom_or_audience_sharing"
PREFER_NOT_TO_ANSWER = "prefer not to answer"

PACKET_REQUIRED_FIELDS = frozenset(
    {
        "packet_schema_version",
        "exchange_artifact_id",
        "presentation_asset",
        "presentation_checksum",
        "permitted_accessibility_overlays",
        "excluded_field_attestation",
        "reveal_state",
        "protocol_deviation",
        "sharing_permission_scope",
    }
)
PACKET_FORBIDDEN_FIELDS = frozenset(
    {
        "creator_intention",
        "model_output",
        "condition_assignment",
        "private_artifact_id",
        "content_sha256",
        "unapproved_raw_media",
    }
)
PRESENTATION_ASSET_ALLOWED_FIELDS = frozenset(
    {
        "asset_reference",
        "media_type",
        "duration_ms",
        "caption_reference",
        "transcript_reference",
        "accessibility_note",
        "language",
    }
)
PRESENTATION_ASSET_REQUIRED_FIELDS = frozenset(
    {"asset_reference", "media_type", "duration_ms"}
)
READING_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "response_id",
        "exchange_artifact_id",
        "audience_pseudonym",
        "local_created_at_utc",
        "event_index",
        "elapsed_ms",
        "blindness_attestation",
        "permission_scope",
        "open_interpretation",
        "sound_image_relation",
        "shared_tags",
        "self_described_tags",
        "accessibility_barriers",
        "confidence_or_ambiguity",
        "withdrawn",
    }
)
READING_OPTIONAL_FIELDS = frozenset({"respondent_session_pseudonym"})
DIRECT_IDENTIFIER_FIELDS = frozenset(
    {
        "legal_name",
        "name",
        "email",
        "phone",
        "filename",
        "file_path",
        "device_id",
        "device_metadata",
        "platform_metadata",
        "ip_address",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RevealState(str, Enum):
    """Monotonic audience reveal state."""

    BLINDED = "blinded"
    REVEALED = "revealed"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: str = "error"


class AudienceValidationError(ValueError):
    """Raised when a packet or reading violates the frozen contract."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(detail or "audience exchange validation failed")


@dataclass(frozen=True)
class AudiencePacket:
    packet_schema_version: str
    exchange_artifact_id: str
    presentation_asset: Mapping[str, Any]
    presentation_checksum: str
    permitted_accessibility_overlays: tuple[str, ...]
    excluded_field_attestation: tuple[str, ...]
    reveal_state: RevealState
    protocol_deviation: str | None
    sharing_permission_scope: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AudiencePacket":
        issues = validate_packet_mapping(value)
        if _errors(issues):
            raise AudienceValidationError(_errors(issues))
        return cls(
            packet_schema_version=str(value["packet_schema_version"]),
            exchange_artifact_id=str(value["exchange_artifact_id"]),
            presentation_asset=_json_copy(value["presentation_asset"]),
            presentation_checksum=str(value["presentation_checksum"]),
            permitted_accessibility_overlays=tuple(
                str(item) for item in value["permitted_accessibility_overlays"]
            ),
            excluded_field_attestation=tuple(
                str(item) for item in value["excluded_field_attestation"]
            ),
            reveal_state=RevealState(str(value["reveal_state"])),
            protocol_deviation=_optional_text(value["protocol_deviation"]),
            sharing_permission_scope=str(value["sharing_permission_scope"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["presentation_asset"] = _json_copy(self.presentation_asset)
        value["permitted_accessibility_overlays"] = list(
            self.permitted_accessibility_overlays
        )
        value["excluded_field_attestation"] = list(self.excluded_field_attestation)
        value["reveal_state"] = self.reveal_state.value
        return value

    def reveal(self, *, protocol_deviation: str | None = None) -> "AudiencePacket":
        """Advance to revealed state; callers cannot restore blindness afterward."""

        deviation = _optional_text(protocol_deviation) or self.protocol_deviation
        return replace(
            self,
            reveal_state=RevealState.REVEALED,
            protocol_deviation=deviation,
        )

    def transition_to(self, state: RevealState) -> "AudiencePacket":
        if self.reveal_state is RevealState.REVEALED and state is RevealState.BLINDED:
            raise ValueError("audience reveal is monotonic and blindness cannot be restored")
        if state is RevealState.REVEALED:
            return self.reveal()
        return self


@dataclass(frozen=True)
class AudienceReading:
    schema_version: str
    response_id: str
    exchange_artifact_id: str
    audience_pseudonym: str
    local_created_at_utc: str
    event_index: int
    elapsed_ms: int
    blindness_attestation: bool
    permission_scope: str
    open_interpretation: str
    sound_image_relation: str
    shared_tags: tuple[str, ...]
    self_described_tags: tuple[str, ...]
    accessibility_barriers: str
    confidence_or_ambiguity: str
    withdrawn: bool
    respondent_session_pseudonym: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        packet: AudiencePacket | None = None,
    ) -> "AudienceReading":
        issues = validate_reading_mapping(value, packet=packet)
        if _errors(issues):
            raise AudienceValidationError(_errors(issues))
        return cls(
            schema_version=str(value["schema_version"]),
            response_id=str(value["response_id"]),
            exchange_artifact_id=str(value["exchange_artifact_id"]),
            audience_pseudonym=str(value["audience_pseudonym"]),
            local_created_at_utc=str(value["local_created_at_utc"]),
            event_index=int(value["event_index"]),
            elapsed_ms=int(value["elapsed_ms"]),
            blindness_attestation=bool(value["blindness_attestation"]),
            permission_scope=str(value["permission_scope"]),
            open_interpretation=str(value["open_interpretation"]),
            sound_image_relation=str(value["sound_image_relation"]),
            shared_tags=tuple(str(item) for item in value["shared_tags"]),
            self_described_tags=tuple(
                str(item) for item in value["self_described_tags"]
            ),
            accessibility_barriers=str(value["accessibility_barriers"]),
            confidence_or_ambiguity=str(value["confidence_or_ambiguity"]),
            withdrawn=bool(value["withdrawn"]),
            respondent_session_pseudonym=_optional_text(
                value.get("respondent_session_pseudonym")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shared_tags"] = list(self.shared_tags)
        value["self_described_tags"] = list(self.self_described_tags)
        if self.respondent_session_pseudonym is None:
            value.pop("respondent_session_pseudonym")
        return value


@dataclass(frozen=True)
class AudienceExchangeReport:
    issues: tuple[ValidationIssue, ...]
    accepted_response_ids: tuple[str, ...]
    valid_blinded_response_ids: tuple[str, ...]
    minimum_required: int

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def is_complete(self) -> bool:
        return self.is_valid and len(self.valid_blinded_response_ids) >= self.minimum_required


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes and reject NaN/Infinity."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def presentation_asset_checksum(presentation_asset: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(presentation_asset)).hexdigest()


def validate_packet_mapping(value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if not isinstance(value, Mapping):
        return (ValidationIssue("packet_type", "$", "packet must be a mapping"),)

    keys = {str(key) for key in value}
    for field in sorted(PACKET_REQUIRED_FIELDS - keys):
        issues.append(ValidationIssue("missing_field", field, "required field is missing"))
    for field in sorted(keys - PACKET_REQUIRED_FIELDS):
        code = "forbidden_field" if field in PACKET_FORBIDDEN_FIELDS else "unknown_field"
        issues.append(ValidationIssue(code, field, "field is not in the packet allowlist"))
    if _errors(issues):
        return tuple(issues)

    if value["packet_schema_version"] != AUDIENCE_PACKET_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                "schema_version",
                "packet_schema_version",
                f"expected {AUDIENCE_PACKET_SCHEMA_VERSION}",
            )
        )

    exchange_id = value["exchange_artifact_id"]
    if not _is_opaque_identifier(exchange_id):
        issues.append(
            ValidationIssue(
                "exchange_id",
                "exchange_artifact_id",
                "must be a non-hash opaque identifier without path data",
            )
        )

    asset = value["presentation_asset"]
    if not isinstance(asset, Mapping):
        issues.append(
            ValidationIssue(
                "presentation_asset_type",
                "presentation_asset",
                "must be a neutral presentation-reference mapping, not raw media",
            )
        )
    else:
        asset_keys = {str(key) for key in asset}
        for field in sorted(PRESENTATION_ASSET_REQUIRED_FIELDS - asset_keys):
            issues.append(
                ValidationIssue(
                    "missing_asset_field",
                    f"presentation_asset.{field}",
                    "required presentation reference field is missing",
                )
            )
        for field in sorted(asset_keys - PRESENTATION_ASSET_ALLOWED_FIELDS):
            issues.append(
                ValidationIssue(
                    "unsafe_asset_field",
                    f"presentation_asset.{field}",
                    "field is not in the neutral presentation allowlist",
                )
            )
        duration_ms = asset.get("duration_ms")
        if not _is_nonnegative_int(duration_ms):
            issues.append(
                ValidationIssue(
                    "asset_duration",
                    "presentation_asset.duration_ms",
                    "must be a non-negative integer",
                )
            )

    checksum = value["presentation_checksum"]
    if not isinstance(checksum, str) or not _SHA256_RE.fullmatch(checksum):
        issues.append(
            ValidationIssue(
                "checksum_format",
                "presentation_checksum",
                "must be a lowercase SHA-256 hex digest",
            )
        )
    elif isinstance(asset, Mapping) and checksum != presentation_asset_checksum(asset):
        issues.append(
            ValidationIssue(
                "checksum_mismatch",
                "presentation_checksum",
                "does not match the canonical presentation reference",
            )
        )
    if isinstance(exchange_id, str) and exchange_id == checksum:
        issues.append(
            ValidationIssue(
                "exchange_id_leak",
                "exchange_artifact_id",
                "opaque exchange identity cannot reuse the content checksum",
            )
        )

    overlays = value["permitted_accessibility_overlays"]
    if not _is_text_sequence(overlays):
        issues.append(
            ValidationIssue(
                "overlay_type",
                "permitted_accessibility_overlays",
                "must be a list of non-empty overlay references",
            )
        )

    attestation = value["excluded_field_attestation"]
    if not _is_text_sequence(attestation):
        issues.append(
            ValidationIssue(
                "attestation_type",
                "excluded_field_attestation",
                "must enumerate every frozen forbidden field",
            )
        )
    else:
        missing_attestations = PACKET_FORBIDDEN_FIELDS - set(attestation)
        if missing_attestations:
            issues.append(
                ValidationIssue(
                    "attestation_incomplete",
                    "excluded_field_attestation",
                    "missing: " + ", ".join(sorted(missing_attestations)),
                )
            )

    try:
        reveal_state = RevealState(str(value["reveal_state"]))
    except ValueError:
        reveal_state = None
        issues.append(
            ValidationIssue(
                "reveal_state",
                "reveal_state",
                "must be 'blinded' or 'revealed'",
            )
        )
    deviation = value["protocol_deviation"]
    if deviation is not None and _optional_text(deviation) is None:
        issues.append(
            ValidationIssue(
                "protocol_deviation",
                "protocol_deviation",
                "must be null or non-empty explanatory text",
            )
        )
    if reveal_state is RevealState.REVEALED and _optional_text(deviation) is None:
        issues.append(
            ValidationIssue(
                "revealed_without_record",
                "protocol_deviation",
                "revealed packets must retain an explicit reveal/deviation record",
                severity="warning",
            )
        )
    if value["sharing_permission_scope"] != AUDIENCE_SHARING_PERMISSION:
        issues.append(
            ValidationIssue(
                "permission_scope",
                "sharing_permission_scope",
                f"must be the independent {AUDIENCE_SHARING_PERMISSION!r} scope",
            )
        )

    forbidden_paths = _find_keys(value, PACKET_FORBIDDEN_FIELDS)
    for path in sorted(forbidden_paths):
        if path not in PACKET_FORBIDDEN_FIELDS:
            issues.append(
                ValidationIssue(
                    "nested_forbidden_field",
                    path,
                    "forbidden creator/model/private data is nested in the packet",
                )
            )
    return tuple(issues)


def validate_reading_mapping(
    value: Mapping[str, Any],
    *,
    packet: AudiencePacket | None = None,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if not isinstance(value, Mapping):
        return (ValidationIssue("reading_type", "$", "reading must be a mapping"),)

    keys = {str(key) for key in value}
    for field in sorted(READING_REQUIRED_FIELDS - keys):
        issues.append(ValidationIssue("missing_field", field, "required field is missing"))
    for field in sorted(keys - READING_REQUIRED_FIELDS - READING_OPTIONAL_FIELDS):
        issues.append(
            ValidationIssue(
                "unknown_field",
                field,
                "field is not in the audience-reading allowlist",
            )
        )
    for path in sorted(_find_keys(value, PACKET_FORBIDDEN_FIELDS | DIRECT_IDENTIFIER_FIELDS)):
        issues.append(
            ValidationIssue(
                "unsafe_import_field",
                path,
                "creator/model/private/direct-identifier field is not importable",
            )
        )
    if _errors(issues):
        return tuple(issues)

    if value["schema_version"] != AUDIENCE_READING_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                "schema_version",
                "schema_version",
                f"expected {AUDIENCE_READING_SCHEMA_VERSION}",
            )
        )
    for identifier_field in (
        "response_id",
        "exchange_artifact_id",
        "audience_pseudonym",
    ):
        if not _is_identifier(value[identifier_field]):
            issues.append(
                ValidationIssue(
                    "identifier",
                    identifier_field,
                    "must be a non-empty pseudonymous identifier without path data",
                )
            )
    session_pseudonym = value.get("respondent_session_pseudonym")
    if session_pseudonym is not None and not _is_identifier(session_pseudonym):
        issues.append(
            ValidationIssue(
                "identifier",
                "respondent_session_pseudonym",
                "must be a non-empty pseudonymous identifier without path data",
            )
        )
    if not _is_timezone_aware_iso8601(value["local_created_at_utc"]):
        issues.append(
            ValidationIssue(
                "timestamp",
                "local_created_at_utc",
                "must be an ISO-8601 timestamp with a UTC offset",
            )
        )
    for field in ("event_index", "elapsed_ms"):
        if not _is_nonnegative_int(value[field]):
            issues.append(
                ValidationIssue(
                    "nonnegative_integer",
                    field,
                    "must be a non-negative integer",
                )
            )
    if type(value["blindness_attestation"]) is not bool:
        issues.append(
            ValidationIssue(
                "blindness_type",
                "blindness_attestation",
                "must be boolean",
            )
        )
    if type(value["withdrawn"]) is not bool:
        issues.append(
            ValidationIssue("withdrawn_type", "withdrawn", "must be boolean")
        )
    if value["permission_scope"] != AUDIENCE_SHARING_PERMISSION:
        issues.append(
            ValidationIssue(
                "permission_scope",
                "permission_scope",
                f"must be the independent {AUDIENCE_SHARING_PERMISSION!r} scope",
            )
        )
    for field in (
        "open_interpretation",
        "sound_image_relation",
        "accessibility_barriers",
        "confidence_or_ambiguity",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            issues.append(
                ValidationIssue(
                    "response_text",
                    field,
                    f"must contain text or {PREFER_NOT_TO_ANSWER!r}",
                )
            )
    for field in ("shared_tags", "self_described_tags"):
        if not _is_text_sequence(value[field], allow_empty=True):
            issues.append(
                ValidationIssue(
                    "tag_list",
                    field,
                    "must be a list of non-empty strings",
                )
            )

    if packet is not None:
        if value["exchange_artifact_id"] != packet.exchange_artifact_id:
            issues.append(
                ValidationIssue(
                    "packet_link",
                    "exchange_artifact_id",
                    "does not match the presentation packet",
                )
            )
        has_deviation = packet.protocol_deviation is not None
        if packet.reveal_state is RevealState.REVEALED and not has_deviation:
            issues.append(
                ValidationIssue(
                    "response_after_reveal",
                    "blindness_attestation",
                    "cannot commit after reveal without an explicit protocol deviation",
                )
            )
        elif packet.reveal_state is RevealState.REVEALED:
            issues.append(
                ValidationIssue(
                    "response_after_reveal",
                    "blindness_attestation",
                    "response is retained as a protocol deviation, not a valid blinded reading",
                    severity="warning",
                )
            )
        if value["blindness_attestation"] is False:
            issues.append(
                ValidationIssue(
                    "blindness_not_attested",
                    "blindness_attestation",
                    "response is not a valid blinded reading",
                    severity="warning" if has_deviation else "error",
                )
            )
    return tuple(issues)


def validate_audience_exchange(
    packet: AudiencePacket,
    readings: Iterable[AudienceReading | Mapping[str, Any]],
    *,
    minimum_required: int = 2,
) -> AudienceExchangeReport:
    if minimum_required < 1:
        raise ValueError("minimum_required must be positive")

    issues = list(validate_packet_mapping(packet.to_dict()))
    accepted: list[str] = []
    valid_blinded: list[str] = []
    response_ids: set[str] = set()
    audience_ids: set[str] = set()
    session_ids: set[str] = set()

    for index, candidate in enumerate(readings):
        raw = candidate.to_dict() if isinstance(candidate, AudienceReading) else candidate
        candidate_issues = validate_reading_mapping(raw, packet=packet)
        issues.extend(
            replace(issue, path=f"readings[{index}].{issue.path}")
            for issue in candidate_issues
        )
        if _errors(candidate_issues):
            continue
        reading = (
            candidate
            if isinstance(candidate, AudienceReading)
            else AudienceReading.from_mapping(raw, packet=packet)
        )
        duplicate = False
        for identifier, seen, code, field in (
            (reading.response_id, response_ids, "duplicate_response", "response_id"),
            (
                reading.audience_pseudonym,
                audience_ids,
                "duplicate_audience",
                "audience_pseudonym",
            ),
            (
                reading.respondent_session_pseudonym or reading.audience_pseudonym,
                session_ids,
                "duplicate_session",
                "respondent_session_pseudonym",
            ),
        ):
            if identifier in seen:
                duplicate = True
                issues.append(
                    ValidationIssue(
                        code,
                        f"readings[{index}].{field}",
                        "must be distinct within one audience exchange",
                    )
                )
            else:
                seen.add(identifier)
        if duplicate:
            continue
        accepted.append(reading.response_id)
        if (
            reading.blindness_attestation
            and packet.reveal_state is RevealState.BLINDED
            and not reading.withdrawn
        ):
            valid_blinded.append(reading.response_id)

    if len(valid_blinded) < minimum_required:
        issues.append(
            ValidationIssue(
                "minimum_audience_count",
                "readings",
                f"requires {minimum_required} distinct valid blinded responses; got {len(valid_blinded)}",
            )
        )
    return AudienceExchangeReport(
        issues=tuple(issues),
        accepted_response_ids=tuple(accepted),
        valid_blinded_response_ids=tuple(valid_blinded),
        minimum_required=minimum_required,
    )


def _errors(issues: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    return tuple(issue for issue in issues if issue.severity == "error")


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _is_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value.strip()) <= 128
        and "/" not in value
        and "\\" not in value
        and "@" not in value
    )


def _is_opaque_identifier(value: Any) -> bool:
    return _is_identifier(value) and _SHA256_RE.fullmatch(str(value)) is None


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_text_sequence(value: Any, *, allow_empty: bool = False) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    if not allow_empty and not value:
        return False
    return all(isinstance(item, str) and bool(item.strip()) for item in value)


def _is_timezone_aware_iso8601(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _find_keys(value: Any, names: frozenset[str], path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in names:
                found.add(child_path)
            found.update(_find_keys(child, names, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found.update(_find_keys(child, names, child_path))
    return found
