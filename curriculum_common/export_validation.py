"""Validation helpers for private, audience, and research export boundaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import html
import io
import json
from typing import Any, Iterable, Mapping, Sequence

from curriculum_common.audience_packets import (
    AudiencePacket,
    ValidationIssue,
    canonical_json_bytes,
    validate_packet_mapping,
)


PRIVATE_PORTFOLIO_SCHEMA_VERSION = "private-portfolio/1.0.0"
RESEARCH_PROJECTION_SCHEMA_VERSION = "research-projection/1.0.0"

RESEARCH_ALWAYS_FORBIDDEN_FIELDS = frozenset(
    {
        "content_sha256",
        "sha256",
        "raw_content_hashes",
        "filename",
        "file_name",
        "file_path",
        "path",
        "raw_media",
        "media_bytes",
        "exact_utc",
        "local_created_at_utc",
        "session_started_at_utc",
        "device_metadata",
        "device_id",
        "platform_metadata",
        "ip_address",
        "legal_name",
        "email",
        "uncontrolled_demographics",
        "non_allowlisted_free_text",
    }
)
PRIVATE_RESEARCH_CLAIM_FIELDS = frozenset(
    {
        "participant_id",
        "research_participant_id",
        "research_destination",
        "study_artifact_id",
        "consent_checkbox",
    }
)
FORMULA_PREFIXES = frozenset({"=", "+", "-", "@"})


@dataclass(frozen=True)
class ExportValidationReport:
    projection_kind: str
    issues: tuple[ValidationIssue, ...]
    canonical_sha256: str | None

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def require_valid(self) -> None:
        if not self.is_valid:
            detail = "; ".join(
                f"{issue.path}: {issue.message}"
                for issue in self.issues
                if issue.severity == "error"
            )
            raise ValueError(detail or "export validation failed")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_private_portfolio(value: Any) -> ExportValidationReport:
    issues: list[ValidationIssue] = []
    mapping = _as_mapping(value)
    if mapping is None:
        return _invalid_type("private", "private portfolio must be a mapping")

    _require_equal(
        mapping,
        "schema_version",
        PRIVATE_PORTFOLIO_SCHEMA_VERSION,
        issues,
    )
    _require_equal(mapping, "mode", "Teaching", issues)
    disclosure = mapping.get("boundary_disclosure")
    if not isinstance(disclosure, str) or not disclosure.strip():
        issues.append(
            ValidationIssue(
                "boundary_disclosure",
                "boundary_disclosure",
                "Teaching private export must disclose the browser-to-Molab boundary",
            )
        )
    classification = str(mapping.get("classification", "")).strip().lower()
    if "not research data" not in classification:
        issues.append(
            ValidationIssue(
                "research_claim",
                "classification",
                "Teaching private portfolio must explicitly say it is not research data",
            )
        )
    for path in sorted(_find_nonempty_keys(mapping, PRIVATE_RESEARCH_CLAIM_FIELDS)):
        issues.append(
            ValidationIssue(
                "research_claim",
                path,
                "private portfolio cannot claim research identity, consent, or destination",
            )
        )
    validation = mapping.get("validation")
    if isinstance(validation, Mapping) and validation.get("valid") is not True:
        issues.append(
            ValidationIssue(
                "source_validation",
                "validation.valid",
                "private portfolio reports invalid process history",
            )
        )
    issues.extend(_validate_record_links(mapping))
    return _report("private", mapping, issues)


def validate_research_projection(value: Any) -> ExportValidationReport:
    issues: list[ValidationIssue] = []
    mapping = _as_mapping(value)
    if mapping is None:
        return _invalid_type("research", "research projection must be a mapping")

    _require_equal(
        mapping,
        "schema_version",
        RESEARCH_PROJECTION_SCHEMA_VERSION,
        issues,
    )
    classification = str(mapping.get("classification", "")).strip().lower()
    if "pseudonymous" not in classification or "research projection" not in classification:
        issues.append(
            ValidationIssue(
                "identity_status",
                "classification",
                "research projection must be explicitly described as pseudonymous",
            )
        )
    if not isinstance(mapping.get("redaction_report"), list):
        issues.append(
            ValidationIssue(
                "redaction_report",
                "redaction_report",
                "pure research projection must include a structured redaction report",
            )
        )
    for path in sorted(_find_keys(mapping, RESEARCH_ALWAYS_FORBIDDEN_FIELDS)):
        issues.append(
            ValidationIssue(
                "forbidden_research_field",
                path,
                "field is always forbidden from the minimized research projection",
            )
        )
    has_artifacts = bool(mapping.get("artifacts"))
    if has_artifacts and not _find_keys(mapping, frozenset({"study_artifact_id"})):
        issues.append(
            ValidationIssue(
                "study_identity",
                "study_artifact_id",
                "research projection must use an investigator-issued study artifact ID",
            )
        )
    issues.extend(_validate_research_record_links(mapping))
    return _report("research", mapping, issues)


def validate_audience_packet_export(
    value: AudiencePacket | Mapping[str, Any],
) -> ExportValidationReport:
    mapping = value.to_dict() if isinstance(value, AudiencePacket) else value
    if not isinstance(mapping, Mapping):
        return _invalid_type("audience", "audience packet must be a mapping")
    return _report("audience", mapping, list(validate_packet_mapping(mapping)))


def validate_projection(
    value: Any,
    *,
    projection_kind: str,
) -> ExportValidationReport:
    if projection_kind == "private":
        return validate_private_portfolio(value)
    if projection_kind == "research":
        return validate_research_projection(value)
    if projection_kind == "audience":
        return validate_audience_packet_export(value)
    raise ValueError("projection_kind must be 'private', 'audience', or 'research'")


def neutralize_spreadsheet_formula(value: Any) -> str:
    """Preserve text while preventing CSV spreadsheet formula execution."""

    text = "" if value is None else str(value)
    first_nonspace = text.lstrip()[:1]
    if first_nonspace in FORMULA_PREFIXES or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text


def safe_csv_text(rows: Iterable[Sequence[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for row in rows:
        writer.writerow([neutralize_spreadsheet_formula(value) for value in row])
    return stream.getvalue()


def safe_report_text(value: Any) -> str:
    """Escape untrusted audience text for inclusion in an HTML report."""

    return html.escape("" if value is None else str(value), quote=True)


def export_json_text(value: Mapping[str, Any]) -> str:
    """Canonical round-trip JSON for deterministic bundle members."""

    return canonical_json_bytes(value).decode("utf-8")


def import_json_text(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("export member must contain a JSON object")
    return value


def _invalid_type(kind: str, message: str) -> ExportValidationReport:
    return ExportValidationReport(
        projection_kind=kind,
        issues=(ValidationIssue("projection_type", "$", message),),
        canonical_sha256=None,
    )


def _report(
    kind: str,
    value: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> ExportValidationReport:
    try:
        digest = canonical_sha256(value)
    except (TypeError, ValueError):
        digest = None
        issues.append(
            ValidationIssue(
                "noncanonical_value",
                "$",
                "export must contain finite JSON-compatible values",
            )
        )
    return ExportValidationReport(kind, tuple(issues), digest)


def _require_equal(
    value: Mapping[str, Any],
    field: str,
    expected: str,
    issues: list[ValidationIssue],
) -> None:
    if value.get(field) != expected:
        issues.append(
            ValidationIssue(
                "field_value",
                field,
                f"expected {expected!r}",
            )
        )


def _validate_record_links(value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    raw_records = value.get("records", [])
    if not isinstance(raw_records, list):
        return (ValidationIssue("record_type", "records", "must be a list"),)

    records: dict[str, Mapping[str, Any]] = {}
    command_nonces: set[str] = set()
    for index, record in enumerate(raw_records):
        path = f"records[{index}]"
        if not isinstance(record, Mapping):
            issues.append(ValidationIssue("record_type", path, "must be a mapping"))
            continue
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            issues.append(
                ValidationIssue("missing_record_id", f"{path}.record_id", "is required")
            )
        elif record_id in records:
            issues.append(
                ValidationIssue(
                    "duplicate_record_id",
                    f"{path}.record_id",
                    f"duplicate record ID {record_id!r}",
                )
            )
        else:
            records[record_id] = record
        nonce = record.get("command_nonce")
        if isinstance(nonce, str):
            if nonce in command_nonces:
                issues.append(
                    ValidationIssue(
                        "duplicate_command_nonce",
                        f"{path}.command_nonce",
                        f"duplicate command nonce {nonce!r}",
                    )
                )
            command_nonces.add(nonce)
        if record.get("event_index") != index:
            issues.append(
                ValidationIssue(
                    "event_index",
                    f"{path}.event_index",
                    "must be contiguous and match serialized order",
                )
            )
        if record.get("record_type") == "run":
            for field in ("prediction", "initial_explanation"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    issues.append(
                        ValidationIssue(
                            "missing_prerun_field",
                            f"{path}.{field}",
                            "required before export",
                        )
                    )

    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            continue
        path = f"records[{index}]"
        record_type = record.get("record_type")
        if record_type in {"result", "reflection", "reflection_revision"}:
            run_id = record.get("run_id")
            parent = records.get(str(run_id))
            if parent is None or parent.get("record_type") != "run":
                issues.append(
                    ValidationIssue(
                        "orphan_run_link",
                        f"{path}.run_id",
                        f"unknown run ID {run_id!r}",
                    )
                )
        if record_type == "reflection_revision":
            parent_id = record.get("parent_reflection_id")
            parent = records.get(str(parent_id))
            if parent is None or parent.get("record_type") not in {
                "reflection",
                "reflection_revision",
            }:
                issues.append(
                    ValidationIssue(
                        "orphan_reflection_revision",
                        f"{path}.parent_reflection_id",
                        f"unknown reflection ID {parent_id!r}",
                    )
                )
        if record_type == "withdrawal":
            target_id = record.get("target_record_id")
            if str(target_id) not in records:
                issues.append(
                    ValidationIssue(
                        "orphan_withdrawal",
                        f"{path}.target_record_id",
                        f"unknown record ID {target_id!r}",
                    )
                )
    return tuple(issues)


def _validate_research_record_links(
    value: Mapping[str, Any],
) -> tuple[ValidationIssue, ...]:
    records = value.get("records", [])
    if not isinstance(records, list):
        return (ValidationIssue("record_type", "records", "must be a list"),)
    issues: list[ValidationIssue] = []
    study_record_ids: set[str] = set()
    for index, record in enumerate(records):
        path = f"records[{index}]"
        if not isinstance(record, Mapping):
            issues.append(ValidationIssue("record_type", path, "must be a mapping"))
            continue
        study_record_id = record.get("study_record_id")
        if study_record_id is not None:
            if not isinstance(study_record_id, str) or not study_record_id:
                issues.append(
                    ValidationIssue(
                        "study_record_id",
                        f"{path}.study_record_id",
                        "must be a non-empty projection-scoped identifier",
                    )
                )
            elif study_record_id in study_record_ids:
                issues.append(
                    ValidationIssue(
                        "duplicate_study_record_id",
                        f"{path}.study_record_id",
                        f"duplicate study record ID {study_record_id!r}",
                    )
                )
            else:
                study_record_ids.add(study_record_id)
    return tuple(issues)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    return None


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


def _find_nonempty_keys(
    value: Any,
    names: frozenset[str],
    path: str = "",
) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in names and child not in (None, "", False, [], {}):
                found.add(child_path)
            found.update(_find_nonempty_keys(child, names, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found.update(_find_nonempty_keys(child, names, child_path))
    return found
