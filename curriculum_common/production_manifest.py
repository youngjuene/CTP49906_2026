"""Pure production and artifact-version contracts for the common classroom kernel.

This module deliberately owns only immutable production metadata.  Append-only
session commands, audience exchange, and portfolio projections live in their own
modules so the production contract can be used by either classroom arm without
depending on treatment-only code.
"""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ARTIFACT_VERSION_SCHEMA = "artifact-version/1.0.0"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_ID_RE = re.compile(r"^artifact:sha256:[0-9a-f]{64}$")
_VERSION_LABEL_RE = re.compile(r"^V([1-9][0-9]*)$")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("manifest mapping keys must be strings")
        return {
            key: _normalize_json(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest numbers must be finite")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError("manifest values must be JSON-compatible")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON, rejecting lossy/non-finite values."""

    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_fields(data: Mapping[str, Any], expected: Sequence[str]) -> None:
    actual = set(data)
    required = set(expected)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError("invalid manifest fields: " + "; ".join(details))


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be non-empty text".format(name))
    return value


def _require_content_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("content_sha256 must be sha256:<64 lowercase hex>")
    return value


def _validate_utc(value: str) -> str:
    _require_text("local_registered_at_utc", value)
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise ValueError("local_registered_at_utc must include a UTC offset")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("local_registered_at_utc is not valid ISO-8601") from exc
    return value


@dataclass(frozen=True)
class EditDecisionManifest:
    """Common editor/export decisions serialized identically in both arms."""

    editor_name_version: str
    source_assets: Tuple[Mapping[str, Any], ...]
    ordering: Tuple[str, ...]
    trims: Tuple[Mapping[str, Any], ...]
    mix_levels: Tuple[Mapping[str, Any], ...]
    accessibility_work: Mapping[str, Any]
    assistance_disclosure: Mapping[str, Any]
    export_preset_version: str

    def __post_init__(self) -> None:
        _require_text("editor_name_version", self.editor_name_version)
        _require_text("export_preset_version", self.export_preset_version)
        if not self.source_assets:
            raise ValueError("source_assets must record at least one source")
        _normalize_json(self.to_artifact_fields())

    def to_artifact_fields(self) -> Dict[str, Any]:
        return {
            "editor_name_version": self.editor_name_version,
            "source_assets": list(self.source_assets),
            "edit_decisions": {
                "ordering": list(self.ordering),
                "trims": list(self.trims),
                "mix_levels": list(self.mix_levels),
            },
            "accessibility": dict(self.accessibility_work),
            "assistance_disclosure": dict(self.assistance_disclosure),
            "export_preset_version": self.export_preset_version,
        }

    @classmethod
    def from_artifact_fields(cls, data: Mapping[str, Any]) -> "EditDecisionManifest":
        expected = (
            "editor_name_version",
            "source_assets",
            "edit_decisions",
            "accessibility",
            "assistance_disclosure",
            "export_preset_version",
        )
        _require_exact_fields(data, expected)
        decisions = data["edit_decisions"]
        if not isinstance(decisions, Mapping):
            raise TypeError("edit_decisions must be a mapping")
        _require_exact_fields(decisions, ("ordering", "trims", "mix_levels"))
        return cls(
            editor_name_version=data["editor_name_version"],
            source_assets=tuple(data["source_assets"]),
            ordering=tuple(decisions["ordering"]),
            trims=tuple(decisions["trims"]),
            mix_levels=tuple(decisions["mix_levels"]),
            accessibility_work=dict(data["accessibility"]),
            assistance_disclosure=dict(data["assistance_disclosure"]),
            export_preset_version=data["export_preset_version"],
        )


def artifact_id_for(
    content_sha256: str,
    version_label: str,
    parent_artifact_id: Optional[str],
    schema_version: str = ARTIFACT_VERSION_SCHEMA,
) -> str:
    """Derive an immutable ID without filenames, timestamps, or creator text."""

    _require_content_sha256(content_sha256)
    _require_text("version_label", version_label)
    identity = {
        "schema_version": schema_version,
        "content_sha256": content_sha256,
        "version_label": version_label,
        "parent_artifact_id": parent_artifact_id,
    }
    return "artifact:sha256:" + sha256(canonical_json_bytes(identity)).hexdigest()


@dataclass(frozen=True)
class ArtifactVersion:
    schema_version: str
    artifact_id: str
    content_sha256: str
    version_label: str
    parent_artifact_id: Optional[str]
    local_registered_at_utc: str
    event_index: int
    elapsed_ms: int
    media_facts: Mapping[str, Any]
    creator_intention: str
    intended_audience: str
    sound_image_relation: str
    concept_tags: Tuple[str, ...]
    cultural_aesthetic_context: str
    source_license_provenance: Mapping[str, Any]
    accessibility: Mapping[str, Any]
    editor_name_version: str
    source_assets: Tuple[Mapping[str, Any], ...]
    edit_decisions: Mapping[str, Any]
    assistance_disclosure: Mapping[str, Any]
    export_preset_version: str
    change_rationale: str
    processing_boundary: str

    FIELDS = (
        "schema_version", "artifact_id", "content_sha256", "version_label",
        "parent_artifact_id", "local_registered_at_utc", "event_index",
        "elapsed_ms", "media_facts", "creator_intention", "intended_audience",
        "sound_image_relation", "concept_tags", "cultural_aesthetic_context",
        "source_license_provenance", "accessibility", "editor_name_version",
        "source_assets", "edit_decisions", "assistance_disclosure",
        "export_preset_version", "change_rationale", "processing_boundary",
    )

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_VERSION_SCHEMA:
            raise ValueError("unsupported artifact schema version")
        _require_content_sha256(self.content_sha256)
        match = _VERSION_LABEL_RE.fullmatch(self.version_label or "")
        if not match:
            raise ValueError("version_label must be V1, V2, ...")
        version_number = int(match.group(1))
        if version_number == 1 and self.parent_artifact_id is not None:
            raise ValueError("V1 cannot have a parent_artifact_id")
        if version_number > 1 and not self.parent_artifact_id:
            raise ValueError("V2 and later require an immutable parent_artifact_id")
        if self.parent_artifact_id is not None and not _ARTIFACT_ID_RE.fullmatch(
            self.parent_artifact_id
        ):
            raise ValueError("parent_artifact_id has an invalid format")
        expected_id = artifact_id_for(
            self.content_sha256,
            self.version_label,
            self.parent_artifact_id,
            self.schema_version,
        )
        if self.artifact_id != expected_id:
            raise ValueError("artifact_id does not match immutable identity fields")
        _validate_utc(self.local_registered_at_utc)
        if (
            isinstance(self.event_index, bool)
            or not isinstance(self.event_index, int)
            or self.event_index < 0
        ):
            raise ValueError("event_index must be a non-negative integer")
        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, int)
            or self.elapsed_ms < 0
        ):
            raise ValueError("elapsed_ms must be a non-negative integer")
        for name in (
            "creator_intention", "intended_audience", "sound_image_relation",
            "cultural_aesthetic_context", "editor_name_version",
            "export_preset_version", "change_rationale", "processing_boundary",
        ):
            _require_text(name, getattr(self, name))
        if not self.concept_tags or not all(
            isinstance(tag, str) and tag.strip() for tag in self.concept_tags
        ):
            raise ValueError("concept_tags must contain non-empty text")
        if not self.source_assets:
            raise ValueError("source_assets must not be empty")
        _normalize_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "version_label": self.version_label,
            "parent_artifact_id": self.parent_artifact_id,
            "local_registered_at_utc": self.local_registered_at_utc,
            "event_index": self.event_index,
            "elapsed_ms": self.elapsed_ms,
            "media_facts": dict(self.media_facts),
            "creator_intention": self.creator_intention,
            "intended_audience": self.intended_audience,
            "sound_image_relation": self.sound_image_relation,
            "concept_tags": list(self.concept_tags),
            "cultural_aesthetic_context": self.cultural_aesthetic_context,
            "source_license_provenance": dict(self.source_license_provenance),
            "accessibility": dict(self.accessibility),
            "editor_name_version": self.editor_name_version,
            "source_assets": list(self.source_assets),
            "edit_decisions": dict(self.edit_decisions),
            "assistance_disclosure": dict(self.assistance_disclosure),
            "export_preset_version": self.export_preset_version,
            "change_rationale": self.change_rationale,
            "processing_boundary": self.processing_boundary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactVersion":
        _require_exact_fields(data, cls.FIELDS)
        return cls(
            schema_version=data["schema_version"],
            artifact_id=data["artifact_id"],
            content_sha256=data["content_sha256"],
            version_label=data["version_label"],
            parent_artifact_id=data["parent_artifact_id"],
            local_registered_at_utc=data["local_registered_at_utc"],
            event_index=data["event_index"],
            elapsed_ms=data["elapsed_ms"],
            media_facts=dict(data["media_facts"]),
            creator_intention=data["creator_intention"],
            intended_audience=data["intended_audience"],
            sound_image_relation=data["sound_image_relation"],
            concept_tags=tuple(data["concept_tags"]),
            cultural_aesthetic_context=data["cultural_aesthetic_context"],
            source_license_provenance=dict(data["source_license_provenance"]),
            accessibility=dict(data["accessibility"]),
            editor_name_version=data["editor_name_version"],
            source_assets=tuple(data["source_assets"]),
            edit_decisions=dict(data["edit_decisions"]),
            assistance_disclosure=dict(data["assistance_disclosure"]),
            export_preset_version=data["export_preset_version"],
            change_rationale=data["change_rationale"],
            processing_boundary=data["processing_boundary"],
        )


def create_artifact_version(
    *,
    content_sha256: str,
    version_label: str,
    parent_artifact_id: Optional[str],
    local_registered_at_utc: str,
    event_index: int,
    elapsed_ms: int,
    media_facts: Mapping[str, Any],
    creator_intention: str,
    intended_audience: str,
    sound_image_relation: str,
    concept_tags: Sequence[str],
    cultural_aesthetic_context: str,
    source_license_provenance: Mapping[str, Any],
    edit_manifest: EditDecisionManifest,
    change_rationale: str,
    processing_boundary: str,
) -> ArtifactVersion:
    """Create a validated V1/V2 record without mutating an earlier version."""

    fields = edit_manifest.to_artifact_fields()
    artifact_id = artifact_id_for(content_sha256, version_label, parent_artifact_id)
    return ArtifactVersion(
        schema_version=ARTIFACT_VERSION_SCHEMA,
        artifact_id=artifact_id,
        content_sha256=content_sha256,
        version_label=version_label,
        parent_artifact_id=parent_artifact_id,
        local_registered_at_utc=local_registered_at_utc,
        event_index=event_index,
        elapsed_ms=elapsed_ms,
        media_facts=dict(media_facts),
        creator_intention=creator_intention,
        intended_audience=intended_audience,
        sound_image_relation=sound_image_relation,
        concept_tags=tuple(concept_tags),
        cultural_aesthetic_context=cultural_aesthetic_context,
        source_license_provenance=dict(source_license_provenance),
        accessibility=fields["accessibility"],
        editor_name_version=fields["editor_name_version"],
        source_assets=tuple(fields["source_assets"]),
        edit_decisions=fields["edit_decisions"],
        assistance_disclosure=fields["assistance_disclosure"],
        export_preset_version=fields["export_preset_version"],
        change_rationale=change_rationale,
        processing_boundary=processing_boundary,
    )


def dump_artifact_version(artifact: ArtifactVersion) -> bytes:
    return canonical_json_bytes(artifact.to_dict())


def load_artifact_version(payload: bytes) -> ArtifactVersion:
    try:
        data = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is not valid UTF-8 JSON") from exc
    if not isinstance(data, Mapping):
        raise ValueError("artifact manifest root must be an object")
    return ArtifactVersion.from_dict(data)
