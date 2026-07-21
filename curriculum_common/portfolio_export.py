"""Pure private-portfolio and minimized Research-mode projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from curriculum_common.session_records import (
    ModeDecision,
    OperatingMode,
    PermissionScope,
    RESEARCH_ALWAYS_FORBIDDEN_FIELDS,
    SessionLog,
    validate_process_log,
)


PRIVATE_PORTFOLIO_SCHEMA_VERSION = "private-portfolio/1.0.0"
RESEARCH_PROJECTION_SCHEMA_VERSION = "research-projection/1.0.0"

DEFAULT_MOLAB_BOUNDARY_DISCLOSURE = (
    "Session state is processed inside the hosted Molab session/container boundary, "
    "not solely on the student's device. This private download is user-initiated."
)

ALWAYS_FORBIDDEN_RESEARCH_FIELDS = RESEARCH_ALWAYS_FORBIDDEN_FIELDS
CONTENT_HASH_FIELDS = frozenset({"content_sha256", "sha256"})

IDENTITY_FIELDS = frozenset(
    {"artifact_id", "private_artifact_id", "exchange_artifact_id"}
)
RECORD_LINK_FIELDS = frozenset(
    {
        "record_id",
        "run_id",
        "result_id",
        "reflection_id",
        "revision_id",
        "parent_reflection_id",
        "target_record_id",
        "withdrawal_id",
    }
)
FREE_TEXT_FIELDS = frozenset(
    {
        "prompt",
        "prediction",
        "initial_explanation",
        "reflection",
        "creator_intention",
        "change_rationale",
        "open_text",
        "notes",
        "reason",
    }
)
STRUCTURAL_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "event_index",
        "elapsed_ms",
        "status",
        "mode",
        "protocol_version",
        "course_release_id",
    }
)


class ProjectionError(ValueError):
    """A private or research projection could not satisfy its contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"projection contains non-canonical JSON: {exc}") from exc


def _clone(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def build_private_portfolio(
    log: SessionLog,
    *,
    artifact_versions: Sequence[Mapping[str, Any]] = (),
    boundary_disclosure: str = DEFAULT_MOLAB_BOUNDARY_DISCLOSURE,
) -> dict[str, Any]:
    """Build a rich user-controlled private download without a research claim."""

    if not boundary_disclosure.strip():
        raise ProjectionError("the Molab processing-boundary disclosure is required")
    records = [record.to_dict() for record in log.records]
    issues = validate_process_log(log)
    payload: dict[str, Any] = {
        "schema_version": PRIVATE_PORTFOLIO_SCHEMA_VERSION,
        "classification": "private learning portfolio; pseudonymous; not research data",
        "owner": "student_or_creator",
        "mode": log.mode.value,
        "protocol_version": log.protocol_version,
        "course_release_id": log.course_release_id,
        "session_pseudonym": log.session_pseudonym,
        "boundary_disclosure": boundary_disclosure.strip(),
        "automatic_student_data_egress": False,
        "research_destination": None,
        "artifacts": [_clone(dict(item)) for item in artifact_versions],
        "records": records,
        "validation": {
            "valid": not issues,
            "issues": [
                {
                    "code": issue.code,
                    "record_id": issue.record_id,
                    "field": issue.field,
                    "message": issue.message,
                }
                for issue in issues
            ],
        },
    }
    content_checksum = _checksum(payload)
    payload["portfolio_id"] = f"private-{content_checksum[:24]}"
    payload["checksum_sha256"] = _checksum(payload)
    return payload


def serialize_private_portfolio(portfolio: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(dict(portfolio))


@dataclass(frozen=True)
class ResearchProjection:
    data: dict[str, Any]
    redaction_report: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _clone(self.data)

    def to_bytes(self) -> bytes:
        return _canonical_json_bytes(self.data)


def _study_record_id(value: str) -> str:
    digest = hashlib.sha256(
        f"{RESEARCH_PROJECTION_SCHEMA_VERSION}:{value}".encode("utf-8")
    ).hexdigest()
    return f"sr_{digest[:32]}"


def _report(
    report: list[dict[str, Any]],
    *,
    path: str,
    field: str,
    action: str,
    reason: str,
) -> None:
    report.append({"location": path, "field": field, "action": action, "reason": reason})


def _project_nested(
    value: Any,
    *,
    path: str,
    permitted: frozenset[str],
    identity_map: Mapping[str, str],
    report: list[dict[str, Any]],
) -> Any:
    if isinstance(value, list):
        return [
            _project_nested(
                item,
                path=f"{path}[{index}]",
                permitted=permitted,
                identity_map=identity_map,
                report=report,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return _clone(value)

    projected: dict[str, Any] = {}
    for key, child in value.items():
        child_path = f"{path}.{key}" if path else key
        if key in IDENTITY_FIELDS:
            source_id = str(child)
            study_id = identity_map.get(source_id)
            if not study_id:
                raise ProjectionError(f"{child_path} has no investigator-issued study mapping")
            existing = projected.get("study_artifact_id")
            if existing is not None and existing != study_id:
                raise ProjectionError(f"{path} maps to conflicting study artifact identifiers")
            projected["study_artifact_id"] = study_id
            _report(
                report,
                path=child_path,
                field=key,
                action="replace",
                reason="private/exchange identity replaced by investigator-issued study_artifact_id",
            )
            continue
        if key in CONTENT_HASH_FIELDS:
            study_id = identity_map.get(str(child))
            if study_id:
                existing = projected.get("study_artifact_id")
                if existing is not None and existing != study_id:
                    raise ProjectionError(f"{path} maps to conflicting study artifact identifiers")
                projected["study_artifact_id"] = study_id
                action = "replace"
                reason = "private content hash replaced by investigator-issued study_artifact_id"
            else:
                action = "remove"
                reason = "raw content hashes are forbidden in the Research projection"
            _report(report, path=child_path, field=key, action=action, reason=reason)
            continue
        if key in RECORD_LINK_FIELDS:
            if child is None:
                continue
            output_key = "study_" + key
            projected[output_key] = _study_record_id(str(child))
            _report(
                report,
                path=child_path,
                field=key,
                action="replace",
                reason="private process identifier replaced by projection-scoped identifier",
            )
            continue
        if key in ALWAYS_FORBIDDEN_RESEARCH_FIELDS:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="field is always forbidden in the Research projection",
            )
            continue
        if key in FREE_TEXT_FIELDS and key not in permitted:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="non-allowlisted free text is forbidden",
            )
            continue
        projected[key] = _project_nested(
            child,
            path=child_path,
            permitted=permitted,
            identity_map=identity_map,
            report=report,
        )
    return projected


def _project_record(
    record: Mapping[str, Any],
    *,
    index: int,
    permitted: frozenset[str],
    identity_map: Mapping[str, str],
    report: list[dict[str, Any]],
) -> dict[str, Any]:
    path = f"records[{index}]"
    projected: dict[str, Any] = {}
    for key, value in record.items():
        child_path = f"{path}.{key}"
        if key in IDENTITY_FIELDS:
            source_id = str(value)
            study_id = identity_map.get(source_id)
            if not study_id:
                raise ProjectionError(f"{child_path} has no investigator-issued study mapping")
            existing = projected.get("study_artifact_id")
            if existing is not None and existing != study_id:
                raise ProjectionError(f"{path} maps to conflicting study artifact identifiers")
            projected["study_artifact_id"] = study_id
            _report(
                report,
                path=child_path,
                field=key,
                action="replace",
                reason="private/exchange identity replaced by investigator-issued study_artifact_id",
            )
        elif key in CONTENT_HASH_FIELDS:
            study_id = identity_map.get(str(value))
            if study_id:
                existing = projected.get("study_artifact_id")
                if existing is not None and existing != study_id:
                    raise ProjectionError(f"{path} maps to conflicting study artifact identifiers")
                projected["study_artifact_id"] = study_id
                action = "replace"
                reason = "private content hash replaced by investigator-issued study_artifact_id"
            else:
                action = "remove"
                reason = "raw content hashes are forbidden in the Research projection"
            _report(report, path=child_path, field=key, action=action, reason=reason)
        elif key in RECORD_LINK_FIELDS:
            projected["study_" + key] = _study_record_id(str(value))
            _report(
                report,
                path=child_path,
                field=key,
                action="replace",
                reason="private process identifier replaced by projection-scoped identifier",
            )
        elif key in ALWAYS_FORBIDDEN_RESEARCH_FIELDS:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="field is always forbidden in the Research projection",
            )
        elif key in FREE_TEXT_FIELDS and key not in permitted:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="non-allowlisted free text is forbidden",
            )
        elif key in STRUCTURAL_RECORD_FIELDS or key in permitted:
            projected[key] = _project_nested(
                value,
                path=child_path,
                permitted=permitted,
                identity_map=identity_map,
                report=report,
            )
        else:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="field is outside the approved Research allowlist",
            )
    return projected


def _project_artifact(
    artifact: Mapping[str, Any],
    *,
    index: int,
    permitted: frozenset[str],
    identity_map: Mapping[str, str],
    report: list[dict[str, Any]],
) -> dict[str, Any]:
    path = f"artifacts[{index}]"
    projected: dict[str, Any] = {}
    for key, value in artifact.items():
        child_path = f"{path}.{key}"
        if key in IDENTITY_FIELDS or key in CONTENT_HASH_FIELDS:
            study_id = identity_map.get(str(value))
            if not study_id:
                if key in CONTENT_HASH_FIELDS and "study_artifact_id" in projected:
                    _report(
                        report,
                        path=child_path,
                        field=key,
                        action="remove",
                        reason="raw content hash removed after artifact identity mapping",
                    )
                    continue
                raise ProjectionError(f"{child_path} has no investigator-issued study mapping")
            existing = projected.get("study_artifact_id")
            if existing is not None and existing != study_id:
                raise ProjectionError(f"{path} maps to conflicting study artifact identifiers")
            projected["study_artifact_id"] = study_id
            _report(
                report,
                path=child_path,
                field=key,
                action="replace",
                reason="private artifact identity replaced by investigator-issued study_artifact_id",
            )
        elif key in ALWAYS_FORBIDDEN_RESEARCH_FIELDS:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="field is always forbidden in the Research projection",
            )
        elif key in permitted:
            projected[key] = _project_nested(
                value,
                path=child_path,
                permitted=permitted,
                identity_map=identity_map,
                report=report,
            )
        else:
            _report(
                report,
                path=child_path,
                field=key,
                action="remove",
                reason="field is outside the approved Research allowlist",
            )
    return projected


def project_research(
    private_portfolio: Mapping[str, Any],
    *,
    decision: ModeDecision,
    identity_map: Mapping[str, str],
    withdrawal_policy: str = "tombstone",
) -> ResearchProjection:
    """Create a pure, minimized Research projection plus a redaction report."""

    if decision.mode is not OperatingMode.RESEARCH or decision.configuration is None:
        raise ProjectionError("Research projection is disabled outside approved Research mode")
    if not decision.permits(PermissionScope.PROCESS_DATA_ANALYSIS):
        raise ProjectionError("process-data analysis permission is not active")
    if withdrawal_policy not in {"omit", "tombstone"}:
        raise ProjectionError("withdrawal_policy must be 'omit' or 'tombstone'")

    before = _canonical_json_bytes(dict(private_portfolio))
    source = _clone(dict(private_portfolio))
    if source.get("schema_version") != PRIVATE_PORTFOLIO_SCHEMA_VERSION:
        raise ProjectionError("unsupported private portfolio schema")
    if not source.get("validation", {}).get("valid", False):
        raise ProjectionError("invalid private history cannot be projected as research-ready")

    configuration = decision.configuration
    assert configuration is not None
    permitted = decision.permitted_fields
    report: list[dict[str, Any]] = []
    records = source.get("records", [])
    if not isinstance(records, list):
        raise ProjectionError("private portfolio records must be a list")
    artifacts = source.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ProjectionError("private portfolio artifacts must be a list")
    projected_artifacts = [
        _project_artifact(
            artifact,
            index=index,
            permitted=permitted,
            identity_map=identity_map,
            report=report,
        )
        for index, artifact in enumerate(artifacts)
        if isinstance(artifact, dict)
    ]
    if len(projected_artifacts) != len(artifacts):
        raise ProjectionError("private portfolio artifacts must be objects")

    withdrawn_ids = {
        str(record.get("target_record_id"))
        for record in records
        if isinstance(record, dict)
        and record.get("record_type") == "withdrawal"
        and record.get("target_record_id")
    }
    projected_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ProjectionError(f"records[{index}] is not an object")
        record_id = str(record.get("record_id", ""))
        if record_id in withdrawn_ids:
            if withdrawal_policy == "omit":
                _report(
                    report,
                    path=f"records[{index}]",
                    field="record",
                    action="remove",
                    reason="withdrawn record omitted by approved projection policy",
                )
                continue
            projected_records.append(
                {
                    "schema_version": RESEARCH_PROJECTION_SCHEMA_VERSION,
                    "record_type": "withdrawn_record_tombstone",
                    "study_record_id": _study_record_id(record_id),
                    "event_index": record.get("event_index"),
                    "elapsed_ms": record.get("elapsed_ms"),
                    "status": "withdrawn",
                }
            )
            _report(
                report,
                path=f"records[{index}]",
                field="record",
                action="replace",
                reason="withdrawn private record replaced by a minimized tombstone",
            )
            continue
        projected_records.append(
            _project_record(
                record,
                index=index,
                permitted=permitted,
                identity_map=identity_map,
                report=report,
            )
        )

    data: dict[str, Any] = {
        "schema_version": RESEARCH_PROJECTION_SCHEMA_VERSION,
        "classification": "pseudonymous minimized research projection",
        "protocol_version": configuration["approved_protocol_version"],
        "course_release_id": configuration["compatible_course_release_id"],
        "destination": configuration["destination"],
        "retention_period": configuration["retention_period"],
        "withdrawal_path": configuration["withdrawal_path"],
        "permitted_fields": sorted(permitted),
        "withdrawal_policy": withdrawal_policy,
        "artifacts": projected_artifacts,
        "records": projected_records,
        "redaction_report": report,
    }
    data["projection_id"] = f"research-{_checksum(data)[:24]}"
    data["checksum_sha256"] = _checksum(data)

    after = _canonical_json_bytes(dict(private_portfolio))
    if before != after:
        raise ProjectionError("private portfolio changed during projection")
    return ResearchProjection(_clone(data), tuple(_clone(report)))


def forbidden_research_paths(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return forbidden-field paths for a lightweight projection safety audit."""

    paths: list[str] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else key
                if key in ALWAYS_FORBIDDEN_RESEARCH_FIELDS or key in IDENTITY_FIELDS:
                    paths.append(child_path)
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(dict(value), "")
    return tuple(paths)


__all__ = [
    "ALWAYS_FORBIDDEN_RESEARCH_FIELDS",
    "DEFAULT_MOLAB_BOUNDARY_DISCLOSURE",
    "PRIVATE_PORTFOLIO_SCHEMA_VERSION",
    "ProjectionError",
    "RESEARCH_PROJECTION_SCHEMA_VERSION",
    "ResearchProjection",
    "build_private_portfolio",
    "forbidden_research_paths",
    "project_research",
    "serialize_private_portfolio",
]
