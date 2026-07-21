"""Treatment-neutral append-only session records and operating-mode gates.

The reducer in this module is deliberately pure: callers submit a command and
receive a new immutable ``SessionLog``.  Marimo cells can therefore rerun after
reactive invalidation without duplicating side effects.  A command nonce is
bound to the canonical command payload; an exact replay returns the prior
result, while reuse with a different payload is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import UUID, uuid4


PROCESS_SCHEMA_VERSION = "append-only-process/1.0.0"
MODE_PROTOCOL_VERSION = "teaching-research-mode/1.0.0"

RESEARCH_CONFIG_REQUIRED_FIELDS = frozenset(
    {
        "approved_protocol_version",
        "approved_wording_reference",
        "independent_permission_scopes",
        "permitted_fields",
        "destination",
        "retention_period",
        "withdrawal_path",
        "contact",
        "expiry",
        "compatible_schema_version",
        "compatible_course_release_id",
    }
)

RESEARCH_ALWAYS_FORBIDDEN_FIELDS = frozenset(
    {
        "raw_content_hashes",
        "content_sha256",
        "sha256",
        "filename",
        "file_name",
        "path",
        "raw_media",
        "media_bytes",
        "local_created_at_utc",
        "session_started_at_utc",
        "exact_utc",
        "device_metadata",
        "device_id",
        "platform_metadata",
        "ip_address",
        "legal_name",
        "email",
        "uncontrolled_demographics",
    }
)


class OperatingMode(str, Enum):
    TEACHING = "Teaching"
    RESEARCH = "Research"


class PermissionScope(str, Enum):
    PROCESS_DATA_ANALYSIS = "process_data_analysis"
    CLASSROOM_OR_AUDIENCE_SHARING = "classroom_or_audience_sharing"
    QUOTATION_OR_REPRODUCTION = "quotation_or_reproduction"
    FUTURE_REUSE = "future_reuse"


ALL_PERMISSION_SCOPES = frozenset(scope.value for scope in PermissionScope)


class ProcessRecordError(ValueError):
    """Base error for rejected process commands and invalid logs."""


class NonceConflictError(ProcessRecordError):
    """A command nonce was reused with a different canonical payload."""


class LinkageError(ProcessRecordError):
    """A result, reflection, revision, or withdrawal has no valid parent."""


class ResultConflictError(ProcessRecordError):
    """A committed run already has a different result."""


class RecordValidationError(ProcessRecordError):
    """Raised when serialized process history violates the frozen contract."""

    def __init__(self, issues: Sequence["ValidationIssue"]):
        self.issues = tuple(issues)
        summary = "; ".join(issue.message for issue in self.issues)
        super().__init__(summary or "invalid process log")


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
        raise ProcessRecordError(f"value is not canonical JSON: {exc}") from exc


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProcessRecordError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProcessRecordError("expiry must be a non-empty UTC timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ProcessRecordError("expiry is not an ISO-8601 timestamp") from exc
    return _as_utc(parsed)


@dataclass(frozen=True)
class ModeDecision:
    """Fail-closed result of resolving an instructor-supplied mode config."""

    mode: OperatingMode
    reason: str
    _configuration: bytes | None = None

    @property
    def configuration(self) -> dict[str, Any] | None:
        if self._configuration is None:
            return None
        return json.loads(self._configuration.decode("utf-8"))

    @property
    def permitted_fields(self) -> frozenset[str]:
        configuration = self.configuration
        if self.mode is not OperatingMode.RESEARCH or configuration is None:
            return frozenset()
        return frozenset(configuration["permitted_fields"])

    def permits(self, scope: PermissionScope | str) -> bool:
        configuration = self.configuration
        if self.mode is not OperatingMode.RESEARCH or configuration is None:
            return False
        scope_name = scope.value if isinstance(scope, PermissionScope) else str(scope)
        scopes = configuration["independent_permission_scopes"]
        return scopes.get(scope_name) is True


def teaching_mode(reason: str = "Teaching is the initial and fail-safe mode") -> ModeDecision:
    return ModeDecision(OperatingMode.TEACHING, reason)


def resolve_operating_mode(
    configuration: Mapping[str, Any] | None,
    *,
    expected_schema_version: str,
    expected_course_release_id: str,
    approved_field_allowlist: Iterable[str],
    now: datetime | None = None,
) -> ModeDecision:
    """Resolve Research only for a complete, approved, compatible configuration.

    Every failure returns Teaching rather than partially enabling research fields.
    Permission scopes are explicit booleans and remain independent.
    """

    if configuration is None:
        return teaching_mode("research configuration is missing")
    try:
        candidate = _json_clone(dict(configuration))
    except ProcessRecordError:
        return teaching_mode("research configuration is not canonical JSON")

    missing = sorted(RESEARCH_CONFIG_REQUIRED_FIELDS.difference(candidate))
    if missing:
        return teaching_mode(f"research configuration is missing: {', '.join(missing)}")
    if candidate.get("mode", OperatingMode.RESEARCH.value) != OperatingMode.RESEARCH.value:
        return teaching_mode("research configuration does not request Research mode")
    if candidate.get("approved") is not True:
        return teaching_mode("research configuration is not institutionally approved")
    if candidate["compatible_schema_version"] != expected_schema_version:
        return teaching_mode("research configuration has the wrong schema version")
    if candidate["compatible_course_release_id"] != expected_course_release_id:
        return teaching_mode("research configuration has the wrong course release")

    required_text = (
        "approved_protocol_version",
        "approved_wording_reference",
        "destination",
        "retention_period",
        "withdrawal_path",
        "contact",
    )
    if any(not isinstance(candidate[name], str) or not candidate[name].strip() for name in required_text):
        return teaching_mode("research configuration contains an empty approval field")

    scopes = candidate["independent_permission_scopes"]
    if not isinstance(scopes, dict) or set(scopes) != ALL_PERMISSION_SCOPES:
        return teaching_mode("research permission scopes must be listed independently")
    if any(type(scopes[name]) is not bool for name in ALL_PERMISSION_SCOPES):
        return teaching_mode("research permission scopes must be explicit booleans")

    permitted = candidate["permitted_fields"]
    if not isinstance(permitted, list) or any(not isinstance(field, str) for field in permitted):
        return teaching_mode("permitted research fields must be a string list")
    permitted_set = set(permitted)
    allowlist = set(approved_field_allowlist)
    excess = sorted(
        permitted_set.difference(allowlist).union(
            permitted_set.intersection(RESEARCH_ALWAYS_FORBIDDEN_FIELDS)
        )
    )
    if excess:
        return teaching_mode(f"research configuration is over-permissive: {', '.join(excess)}")

    try:
        expiry = _parse_utc(candidate["expiry"])
    except ProcessRecordError:
        return teaching_mode("research configuration has an invalid expiry")
    reference_now = _as_utc(now or _utc_now())
    if expiry <= reference_now:
        return teaching_mode("research configuration is expired")

    candidate["permitted_fields"] = sorted(permitted_set)
    candidate["independent_permission_scopes"] = {
        name: scopes[name] for name in sorted(ALL_PERMISSION_SCOPES)
    }
    return ModeDecision(
        OperatingMode.RESEARCH,
        "approved compatible Research configuration",
        _canonical_json_bytes(candidate),
    )


@dataclass(frozen=True)
class ProcessRecord:
    """A byte-stable serialized process record."""

    serialized: bytes

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProcessRecord":
        return cls(_canonical_json_bytes(dict(value)))

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.serialized.decode("utf-8"))
        if not isinstance(value, dict):
            raise ProcessRecordError("process record must be a JSON object")
        return value

    @property
    def record_id(self) -> str:
        return str(self.to_dict().get("record_id", ""))

    @property
    def record_type(self) -> str:
        return str(self.to_dict().get("record_type", ""))


@dataclass(frozen=True)
class CommandReceipt:
    command_nonce: str
    command_digest: str
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class SessionLog:
    schema_version: str
    session_pseudonym: str
    mode: OperatingMode
    session_started_at_utc: str
    protocol_version: str | None = None
    course_release_id: str | None = None
    records: tuple[ProcessRecord, ...] = ()
    receipts: tuple[CommandReceipt, ...] = ()

    def receipt_for(self, command_nonce: str) -> CommandReceipt | None:
        return next(
            (receipt for receipt in self.receipts if receipt.command_nonce == command_nonce),
            None,
        )

    def record_for(self, record_id: str) -> ProcessRecord | None:
        return next((record for record in self.records if record.record_id == record_id), None)

    def records_of_type(self, record_type: str) -> tuple[ProcessRecord, ...]:
        return tuple(record for record in self.records if record.record_type == record_type)

    def to_jsonl(self) -> bytes:
        if not self.records:
            return b""
        return b"\n".join(record.serialized for record in self.records) + b"\n"


@dataclass(frozen=True)
class ReductionResult:
    log: SessionLog
    record_ids: tuple[str, ...]
    replayed: bool


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    record_id: str | None
    field: str | None
    message: str


def new_session(
    session_pseudonym: str,
    *,
    decision: ModeDecision | None = None,
    schema_version: str = PROCESS_SCHEMA_VERSION,
    course_release_id: str | None = None,
    now: datetime | None = None,
) -> SessionLog:
    if not isinstance(session_pseudonym, str) or not session_pseudonym.strip():
        raise ProcessRecordError("session_pseudonym is required")
    selected = decision or teaching_mode()
    configuration = selected.configuration
    if selected.mode is OperatingMode.RESEARCH:
        if configuration is None:
            raise ProcessRecordError("Research mode requires a resolved approved configuration")
        protocol_version = configuration["approved_protocol_version"]
        configured_release = configuration["compatible_course_release_id"]
        if course_release_id is not None and course_release_id != configured_release:
            raise ProcessRecordError("course release differs from the approved Research configuration")
        course_release_id = configured_release
    else:
        protocol_version = None
    started = _utc_text(now or _utc_now())
    return SessionLog(
        schema_version=schema_version,
        session_pseudonym=session_pseudonym.strip(),
        mode=selected.mode,
        session_started_at_utc=started,
        protocol_version=protocol_version,
        course_release_id=course_release_id,
    )


RUN_REQUIRED_FIELDS = (
    "artifact_id",
    "stimulus_id",
    "condition_code",
    "model_id",
    "model_revision",
    "prompt",
    "parameters",
    "prediction",
    "initial_explanation",
)


def _required(command: Mapping[str, Any], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in command or command[name] is None]
    if missing:
        raise ProcessRecordError(f"command is missing required fields: {', '.join(missing)}")


def _require_uuid(value: str, field: str) -> None:
    try:
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProcessRecordError(f"{field} must be a UUID") from exc


def _record_payload(log: SessionLog, record_id: str) -> dict[str, Any]:
    record = log.record_for(record_id)
    if record is None:
        raise LinkageError(f"unknown record: {record_id}")
    return record.to_dict()


def _matching_record(
    log: SessionLog,
    record_type: str,
    field: str,
    value: str,
) -> ProcessRecord | None:
    for record in log.records_of_type(record_type):
        if record.to_dict().get(field) == value:
            return record
    return None


def _append_receipt(
    log: SessionLog,
    nonce: str,
    digest: str,
    record_ids: tuple[str, ...],
) -> SessionLog:
    receipt = CommandReceipt(nonce, digest, record_ids)
    return replace(log, receipts=log.receipts + (receipt,))


def _append_record(
    log: SessionLog,
    *,
    record_type: str,
    record_id: str,
    command_nonce: str,
    command_digest: str,
    payload: Mapping[str, Any],
    now: datetime,
) -> tuple[SessionLog, ProcessRecord]:
    started = _parse_utc(log.session_started_at_utc)
    current = _as_utc(now)
    elapsed_ms = max(0, int((current - started).total_seconds() * 1000))
    if log.records:
        elapsed_ms = max(elapsed_ms, int(log.records[-1].to_dict()["elapsed_ms"]))
    value = {
        "schema_version": log.schema_version,
        "record_type": record_type,
        "record_id": record_id,
        "command_nonce": command_nonce,
        "command_digest": command_digest,
        "event_index": len(log.records),
        "elapsed_ms": elapsed_ms,
        "local_created_at_utc": _utc_text(current),
        "session_started_at_utc": log.session_started_at_utc,
        "session_pseudonym": log.session_pseudonym,
        "mode": log.mode.value,
        "protocol_version": log.protocol_version,
        "course_release_id": log.course_release_id,
        **_json_clone(dict(payload)),
    }
    record = ProcessRecord.from_mapping(value)
    updated = replace(log, records=log.records + (record,))
    return updated, record


def reduce_command(
    log: SessionLog,
    command: Mapping[str, Any],
    *,
    now: datetime | None = None,
    id_factory: Callable[[], str] | None = None,
) -> ReductionResult:
    """Apply one append-only command with exactly-once nonce semantics."""

    _required(command, ("kind", "command_nonce"))
    canonical_command = _json_clone(dict(command))
    kind = canonical_command["kind"]
    nonce = canonical_command["command_nonce"]
    if not isinstance(kind, str) or not kind:
        raise ProcessRecordError("command kind must be a non-empty string")
    if not isinstance(nonce, str) or not nonce.strip():
        raise ProcessRecordError("command_nonce must be a non-empty string")
    digest = hashlib.sha256(_canonical_json_bytes(canonical_command)).hexdigest()
    prior = log.receipt_for(nonce)
    if prior is not None:
        if prior.command_digest != digest:
            raise NonceConflictError(f"command_nonce {nonce!r} was reused with a different payload")
        return ReductionResult(log, prior.record_ids, True)

    clock = _as_utc(now or _utc_now())
    make_id = id_factory or (lambda: str(uuid4()))

    if kind == "commit_run":
        _required(canonical_command, RUN_REQUIRED_FIELDS)
        run_id = str(canonical_command.get("run_id") or make_id())
        _require_uuid(run_id, "run_id")
        for field in ("prompt", "prediction", "initial_explanation"):
            if not isinstance(canonical_command[field], str) or not canonical_command[field].strip():
                raise ProcessRecordError(f"{field} must be a non-empty string")
        if log.record_for(run_id) is not None:
            raise LinkageError(f"record id already exists: {run_id}")
        payload = {name: canonical_command[name] for name in RUN_REQUIRED_FIELDS}
        payload.update(
            {
                "run_id": run_id,
                "status": "committed",
                "metric_versions": canonical_command.get("metric_versions", {}),
                "token_layout_fingerprint": canonical_command.get("token_layout_fingerprint"),
            }
        )
        updated, record = _append_record(
            log,
            record_type="run",
            record_id=run_id,
            command_nonce=nonce,
            command_digest=digest,
            payload=payload,
            now=clock,
        )

    elif kind == "attach_result":
        _required(canonical_command, ("run_id", "result_digest", "metrics"))
        run_id = str(canonical_command["run_id"])
        run = _record_payload(log, run_id)
        if run.get("record_type") != "run" or run.get("status") != "committed":
            raise LinkageError("results attach only to a committed run")
        existing = _matching_record(log, "result", "run_id", run_id)
        logical = {
            "run_id": run_id,
            "result_digest": canonical_command["result_digest"],
            "metrics": canonical_command["metrics"],
            "metric_versions": canonical_command.get("metric_versions", {}),
            "elapsed_seconds": canonical_command.get("elapsed_seconds"),
            "peak_vram_bytes": canonical_command.get("peak_vram_bytes"),
            "peak_host_rss_bytes": canonical_command.get("peak_host_rss_bytes"),
            "status": canonical_command.get("status", "completed"),
        }
        if existing is not None:
            existing_value = existing.to_dict()
            if any(existing_value.get(key) != value for key, value in logical.items()):
                raise ResultConflictError(f"run {run_id} already has a different result")
            replay_log = _append_receipt(log, nonce, digest, (existing.record_id,))
            return ReductionResult(replay_log, (existing.record_id,), True)
        result_id = str(canonical_command.get("result_id") or make_id())
        updated, record = _append_record(
            log,
            record_type="result",
            record_id=result_id,
            command_nonce=nonce,
            command_digest=digest,
            payload={"result_id": result_id, **logical},
            now=clock,
        )

    elif kind == "commit_reflection":
        _required(canonical_command, ("run_id", "reflection"))
        run_id = str(canonical_command["run_id"])
        run = _record_payload(log, run_id)
        if run.get("record_type") != "run":
            raise LinkageError("reflection parent is not a run")
        existing = _matching_record(log, "reflection", "run_id", run_id)
        logical = {
            "run_id": run_id,
            "reflection": canonical_command["reflection"],
            "initial_explanation": run.get("initial_explanation"),
            "tags": canonical_command.get("tags", []),
        }
        if existing is not None:
            existing_value = existing.to_dict()
            if all(existing_value.get(key) == value for key, value in logical.items()):
                replay_log = _append_receipt(log, nonce, digest, (existing.record_id,))
                return ReductionResult(replay_log, (existing.record_id,), True)
            raise LinkageError("edit a reflection by appending a revision")
        reflection_id = str(canonical_command.get("reflection_id") or make_id())
        updated, record = _append_record(
            log,
            record_type="reflection",
            record_id=reflection_id,
            command_nonce=nonce,
            command_digest=digest,
            payload={"reflection_id": reflection_id, **logical},
            now=clock,
        )

    elif kind == "revise_reflection":
        _required(canonical_command, ("parent_reflection_id", "reflection"))
        parent_id = str(canonical_command["parent_reflection_id"])
        parent = _record_payload(log, parent_id)
        if parent.get("record_type") not in {"reflection", "reflection_revision"}:
            raise LinkageError("reflection revision parent is invalid")
        revision_id = str(canonical_command.get("revision_id") or make_id())
        updated, record = _append_record(
            log,
            record_type="reflection_revision",
            record_id=revision_id,
            command_nonce=nonce,
            command_digest=digest,
            payload={
                "revision_id": revision_id,
                "parent_reflection_id": parent_id,
                "run_id": parent["run_id"],
                "reflection": canonical_command["reflection"],
                "initial_explanation": parent["initial_explanation"],
                "tags": canonical_command.get("tags", parent.get("tags", [])),
            },
            now=clock,
        )

    elif kind == "withdraw":
        _required(canonical_command, ("target_record_id", "reason"))
        target_id = str(canonical_command["target_record_id"])
        _record_payload(log, target_id)
        prior_withdrawal = _matching_record(log, "withdrawal", "target_record_id", target_id)
        if prior_withdrawal is not None:
            prior_value = prior_withdrawal.to_dict()
            if prior_value.get("reason") != canonical_command["reason"]:
                raise LinkageError("record already has a different withdrawal state")
            replay_log = _append_receipt(log, nonce, digest, (prior_withdrawal.record_id,))
            return ReductionResult(replay_log, (prior_withdrawal.record_id,), True)
        withdrawal_id = str(canonical_command.get("withdrawal_id") or make_id())
        updated, record = _append_record(
            log,
            record_type="withdrawal",
            record_id=withdrawal_id,
            command_nonce=nonce,
            command_digest=digest,
            payload={
                "withdrawal_id": withdrawal_id,
                "target_record_id": target_id,
                "reason": canonical_command["reason"],
                "status": "withdrawn",
            },
            now=clock,
        )

    else:
        raise ProcessRecordError(f"unsupported command kind: {kind}")

    completed = _append_receipt(updated, nonce, digest, (record.record_id,))
    return ReductionResult(completed, (record.record_id,), False)


def validate_process_log(log: SessionLog) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    seen_nonces: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    prior_elapsed_ms = -1

    for expected_index, record in enumerate(log.records):
        try:
            value = record.to_dict()
        except ProcessRecordError as exc:
            issues.append(ValidationIssue("invalid_json", None, None, str(exc)))
            continue
        record_id = value.get("record_id")
        display_id = record_id if isinstance(record_id, str) else None
        required = (
            "schema_version",
            "record_type",
            "record_id",
            "command_nonce",
            "command_digest",
            "event_index",
            "elapsed_ms",
            "local_created_at_utc",
            "session_pseudonym",
            "mode",
        )
        for field in required:
            if field not in value or value[field] is None:
                issues.append(
                    ValidationIssue(
                        "missing_field",
                        display_id,
                        field,
                        f"record {display_id or expected_index} is missing {field}",
                    )
                )
        if value.get("event_index") != expected_index:
            issues.append(
                ValidationIssue(
                    "event_index",
                    display_id,
                    "event_index",
                    f"record {display_id or expected_index} has a non-contiguous event index",
                )
            )
        elapsed_ms = value.get("elapsed_ms")
        if not isinstance(elapsed_ms, int) or elapsed_ms < prior_elapsed_ms:
            issues.append(
                ValidationIssue(
                    "elapsed_order",
                    display_id,
                    "elapsed_ms",
                    f"record {display_id or expected_index} has non-monotonic elapsed time",
                )
            )
        elif isinstance(elapsed_ms, int):
            prior_elapsed_ms = elapsed_ms
        if value.get("schema_version") != log.schema_version:
            issues.append(
                ValidationIssue(
                    "schema_mismatch",
                    display_id,
                    "schema_version",
                    f"record {display_id or expected_index} has a different schema version",
                )
            )
        if value.get("session_pseudonym") != log.session_pseudonym or value.get("mode") != log.mode.value:
            issues.append(
                ValidationIssue(
                    "session_mismatch",
                    display_id,
                    None,
                    f"record {display_id or expected_index} belongs to a different session or mode",
                )
            )
        nonce = value.get("command_nonce")
        if isinstance(nonce, str):
            if nonce in seen_nonces:
                issues.append(
                    ValidationIssue(
                        "duplicate_nonce",
                        display_id,
                        "command_nonce",
                        f"command nonce {nonce} appears in multiple events",
                    )
                )
            seen_nonces.add(nonce)
        if not isinstance(record_id, str) or not record_id:
            continue
        if record_id in seen_ids:
            issues.append(
                ValidationIssue("duplicate_record", record_id, "record_id", f"duplicate record {record_id}")
            )
        seen_ids.add(record_id)
        records[record_id] = value

    result_runs: set[str] = set()
    reflection_runs: set[str] = set()
    for record_id, value in records.items():
        record_type = value.get("record_type")
        if record_type == "run":
            try:
                _require_uuid(str(value.get("run_id", "")), "run_id")
            except ProcessRecordError:
                issues.append(
                    ValidationIssue(
                        "invalid_run_id",
                        record_id,
                        "run_id",
                        f"run {record_id} does not have a UUID run_id",
                    )
                )
            for field in RUN_REQUIRED_FIELDS:
                if field not in value or value[field] is None:
                    issues.append(
                        ValidationIssue(
                            "missing_run_field",
                            record_id,
                            field,
                            f"run {record_id} is missing {field}",
                        )
                    )
        elif record_type == "result":
            run_id = value.get("run_id")
            if run_id not in records or records.get(run_id, {}).get("record_type") != "run":
                issues.append(
                    ValidationIssue("orphan_result", record_id, "run_id", f"result {record_id} is orphaned")
                )
            elif records[run_id].get("event_index", -1) >= value.get("event_index", -1):
                issues.append(
                    ValidationIssue(
                        "forward_result_link",
                        record_id,
                        "run_id",
                        f"result {record_id} links to a later run",
                    )
                )
            elif run_id in result_runs:
                issues.append(
                    ValidationIssue(
                        "duplicate_result",
                        record_id,
                        "run_id",
                        f"run {run_id} has multiple results",
                    )
                )
            else:
                result_runs.add(str(run_id))
        elif record_type in {"reflection", "reflection_revision"}:
            run_id = value.get("run_id")
            if run_id not in records or records.get(run_id, {}).get("record_type") != "run":
                issues.append(
                    ValidationIssue(
                        "orphan_reflection",
                        record_id,
                        "run_id",
                        f"reflection {record_id} is orphaned",
                    )
                )
            if value.get("initial_explanation") is None:
                issues.append(
                    ValidationIssue(
                        "missing_snapshot",
                        record_id,
                        "initial_explanation",
                        f"reflection {record_id} lacks its initial-explanation snapshot",
                    )
                )
            elif run_id in records and value.get("initial_explanation") != records[run_id].get(
                "initial_explanation"
            ):
                issues.append(
                    ValidationIssue(
                        "snapshot_mismatch",
                        record_id,
                        "initial_explanation",
                        f"reflection {record_id} changed its initial-explanation snapshot",
                    )
                )
            if record_type == "reflection":
                if run_id in reflection_runs:
                    issues.append(
                        ValidationIssue(
                            "duplicate_reflection",
                            record_id,
                            "run_id",
                            f"run {run_id} has multiple initial reflections",
                        )
                    )
                reflection_runs.add(str(run_id))
            else:
                parent_id = value.get("parent_reflection_id")
                if parent_id not in records or records.get(parent_id, {}).get("record_type") not in {
                    "reflection",
                    "reflection_revision",
                }:
                    issues.append(
                        ValidationIssue(
                            "orphan_revision",
                            record_id,
                            "parent_reflection_id",
                            f"revision {record_id} has no valid parent",
                        )
                    )
                elif records[parent_id].get("event_index", -1) >= value.get("event_index", -1):
                    issues.append(
                        ValidationIssue(
                            "forward_revision_link",
                            record_id,
                            "parent_reflection_id",
                            f"revision {record_id} links to a later parent",
                        )
                    )
        elif record_type == "withdrawal":
            target = value.get("target_record_id")
            if target not in records:
                issues.append(
                    ValidationIssue(
                        "orphan_withdrawal",
                        record_id,
                        "target_record_id",
                        f"withdrawal {record_id} has no valid target",
                    )
                )
            elif records[target].get("event_index", -1) >= value.get("event_index", -1):
                issues.append(
                    ValidationIssue(
                        "forward_withdrawal_link",
                        record_id,
                        "target_record_id",
                        f"withdrawal {record_id} links to a later target",
                    )
                )

    receipt_nonces: set[str] = set()
    for receipt in log.receipts:
        if receipt.command_nonce in receipt_nonces:
            issues.append(
                ValidationIssue(
                    "duplicate_nonce",
                    None,
                    "command_nonce",
                    f"duplicate receipt nonce {receipt.command_nonce}",
                )
            )
        receipt_nonces.add(receipt.command_nonce)
        for record_id in receipt.record_ids:
            if record_id not in records:
                issues.append(
                    ValidationIssue(
                        "missing_receipt_record",
                        record_id,
                        "record_ids",
                        f"receipt references missing record {record_id}",
                    )
                )

    return tuple(issues)


def load_jsonl(data: bytes | str) -> SessionLog:
    """Rehydrate event history and nonce receipts without replaying side effects."""

    raw = data.encode("utf-8") if isinstance(data, str) else data
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise RecordValidationError(
            (ValidationIssue("empty_log", None, None, "process JSONL is empty"),)
        )
    records: list[ProcessRecord] = []
    decoded: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordValidationError(
                (ValidationIssue("invalid_json", None, None, f"invalid JSONL record: {exc}"),)
            ) from exc
        if not isinstance(value, dict):
            raise RecordValidationError(
                (ValidationIssue("invalid_record", None, None, "process record is not an object"),)
            )
        decoded.append(value)
        records.append(ProcessRecord.from_mapping(value))

    first = decoded[0]
    try:
        mode = OperatingMode(first["mode"])
        log = SessionLog(
            schema_version=first["schema_version"],
            session_pseudonym=first["session_pseudonym"],
            mode=mode,
            session_started_at_utc=first["session_started_at_utc"],
            protocol_version=first.get("protocol_version"),
            course_release_id=first.get("course_release_id"),
            records=tuple(records),
        )
    except (KeyError, ValueError) as exc:
        raise RecordValidationError(
            (ValidationIssue("missing_session", None, None, f"cannot reconstruct session: {exc}"),)
        ) from exc

    receipts: list[CommandReceipt] = []
    by_nonce: dict[str, tuple[str, list[str]]] = {}
    for value in decoded:
        nonce = value.get("command_nonce")
        digest = value.get("command_digest")
        record_id = value.get("record_id")
        if not all(isinstance(item, str) and item for item in (nonce, digest, record_id)):
            continue
        if nonce in by_nonce and by_nonce[nonce][0] != digest:
            raise RecordValidationError(
                (
                    ValidationIssue(
                        "nonce_conflict",
                        record_id,
                        "command_nonce",
                        f"nonce {nonce} has conflicting serialized payloads",
                    ),
                )
            )
        by_nonce.setdefault(nonce, (digest, []))[1].append(record_id)
    for nonce, (digest, record_ids) in by_nonce.items():
        receipts.append(CommandReceipt(nonce, digest, tuple(record_ids)))
    log = replace(log, receipts=tuple(receipts))
    issues = validate_process_log(log)
    if issues:
        raise RecordValidationError(issues)
    return log


__all__ = [
    "ALL_PERMISSION_SCOPES",
    "CommandReceipt",
    "LinkageError",
    "MODE_PROTOCOL_VERSION",
    "ModeDecision",
    "NonceConflictError",
    "OperatingMode",
    "PROCESS_SCHEMA_VERSION",
    "PermissionScope",
    "ProcessRecord",
    "ProcessRecordError",
    "RESEARCH_ALWAYS_FORBIDDEN_FIELDS",
    "RecordValidationError",
    "ReductionResult",
    "ResultConflictError",
    "SessionLog",
    "ValidationIssue",
    "load_jsonl",
    "new_session",
    "reduce_command",
    "resolve_operating_mode",
    "teaching_mode",
    "validate_process_log",
]
