"""Deterministic identities and pure signal transforms for controlled variants.

Presented-signal controls are media operations.  Processor modality omission and
direct attention-edge knockout are deliberately recognized as distinct operation
families but are never executed by this module.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


STIMULUS_VARIANT_SCHEMA = "stimulus-variant/1.0.0"
STIMULUS_MANIFEST_SCHEMA = "stimulus-manifest/1.0.0"
TRANSFORM_SCHEMA = "stimulus-transform/1.0.0"

MEDIA_OPERATIONS = frozenset(
    {
        "original_reference",
        "audio_swap_duration_matched",
        "temporal_offset",
        "audio_silence_control",
        "video_neutral_control",
    }
)
SIGNAL_CONTROL_OPERATIONS = frozenset(
    {"audio_silence_control", "video_neutral_control"}
)
MODALITY_OMISSION_OPERATIONS = frozenset(
    {"audio_omitted_model_input", "video_omitted_model_input"}
)
MODEL_INTERVENTION_OPERATIONS = frozenset({"direct_attention_edge_knockout"})
KNOWN_OPERATIONS = (
    MEDIA_OPERATIONS | MODALITY_OMISSION_OPERATIONS | MODEL_INTERVENTION_OPERATIONS
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_STIMULUS_ID_RE = re.compile(r"^stimulus:sha256:[0-9a-f]{64}$")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("recipe mapping keys must be strings")
        return {key: _normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("recipe numbers must be finite")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError("recipe values must be JSON-compatible")


def normalize_recipe(recipe: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise TypeError("recipe must be a mapping")
    normalized = _normalize_json(recipe)
    if not isinstance(normalized, dict):
        raise TypeError("recipe must normalize to an object")
    return normalized


def canonical_recipe_bytes(recipe: Mapping[str, Any]) -> bytes:
    return json.dumps(
        normalize_recipe(recipe),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stimulus_id_for(
    parent_content_sha256: str,
    recipe: Mapping[str, Any],
    transform_schema_version: str = TRANSFORM_SCHEMA,
) -> str:
    if not isinstance(parent_content_sha256, str) or not _SHA256_RE.fullmatch(
        parent_content_sha256
    ):
        raise ValueError("parent_content_sha256 must be sha256:<64 lowercase hex>")
    identity = {
        "parent_content_sha256": parent_content_sha256,
        "recipe": normalize_recipe(recipe),
        "transform_schema_version": transform_schema_version,
    }
    digest = sha256(canonical_recipe_bytes(identity)).hexdigest()
    return "stimulus:sha256:" + digest


def duration_match_samples(samples: Sequence[float], target_sample_count: int) -> Tuple[float, ...]:
    """Deterministically trim or zero-pad a mono sample sequence."""

    if (
        isinstance(target_sample_count, bool)
        or not isinstance(target_sample_count, int)
        or target_sample_count < 0
    ):
        raise ValueError("target_sample_count must be a non-negative integer")
    values = tuple(samples)
    if len(values) >= target_sample_count:
        return values[:target_sample_count]
    return values + (0,) * (target_sample_count - len(values))


def offset_samples_zero_fill(
    samples: Sequence[float],
    offset_sample_count: int,
    target_sample_count: Optional[int] = None,
) -> Tuple[float, ...]:
    """Apply a signed offset with zero-fill; samples never circularly wrap."""

    if isinstance(offset_sample_count, bool) or not isinstance(offset_sample_count, int):
        raise TypeError("offset_sample_count must be an integer")
    values = tuple(samples)
    target = len(values) if target_sample_count is None else target_sample_count
    matched = duration_match_samples(values, target)
    if offset_sample_count >= 0:
        return duration_match_samples((0,) * offset_sample_count + matched, target)
    drop = min(-offset_sample_count, target)
    return duration_match_samples(matched[drop:], target)


def resample_pcm_mono(
    samples: Sequence[float], input_sample_rate_hz: int, output_sample_rate_hz: int = 16000
) -> Tuple[float, ...]:
    """Linearly resample small decoded mono fixtures with deterministic length."""

    if input_sample_rate_hz <= 0 or output_sample_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    values = tuple(float(sample) for sample in samples)
    if not values or input_sample_rate_hz == output_sample_rate_hz:
        return values
    output_length = int(round(len(values) * output_sample_rate_hz / input_sample_rate_hz))
    if output_length <= 1 or len(values) == 1:
        return (values[0],) * max(output_length, 1)
    scale = (len(values) - 1) / float(output_length - 1)
    output = []
    for index in range(output_length):
        position = index * scale
        left = int(math.floor(position))
        right = min(left + 1, len(values) - 1)
        fraction = position - left
        output.append(values[left] * (1.0 - fraction) + values[right] * fraction)
    return tuple(output)


def duration_matched_audio_swap(
    donor_samples: Sequence[float],
    donor_sample_rate_hz: int,
    target_sample_count: int,
    processor_sample_rate_hz: int = 16000,
) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    normalized = resample_pcm_mono(
        donor_samples, donor_sample_rate_hz, processor_sample_rate_hz
    )
    matched = duration_match_samples(normalized, target_sample_count)
    if len(normalized) > target_sample_count:
        duration_match = "trim"
    elif len(normalized) < target_sample_count:
        duration_match = "zero_pad"
    else:
        duration_match = "unchanged"
    return matched, {
        "input_sample_rate_hz": donor_sample_rate_hz,
        "processor_sample_rate_hz": processor_sample_rate_hz,
        "resampled_sample_count": len(normalized),
        "target_sample_count": target_sample_count,
        "duration_match": duration_match,
    }


def audio_silence_control(target_sample_count: int) -> Tuple[int, ...]:
    if (
        isinstance(target_sample_count, bool)
        or not isinstance(target_sample_count, int)
        or target_sample_count < 0
    ):
        raise ValueError("target_sample_count must be a non-negative integer")
    return tuple(0 for _ in range(target_sample_count))


def neutral_video_control(frame_count: int, neutral_value: Any = 0) -> Tuple[Any, ...]:
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 0
    ):
        raise ValueError("frame_count must be a non-negative integer")
    return tuple(neutral_value for _ in range(frame_count))


def _canonical_json_bytes(value: Any) -> bytes:
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
        raise ValueError("invalid stimulus fields: missing={} unknown={}".format(missing, unknown))


@dataclass(frozen=True)
class StimulusVariant:
    schema_version: str
    stimulus_id: str
    pair_id: str
    parent_content_sha256: str
    condition_code: str
    technical_operation: str
    congruence_level: str
    congruence_coding_provenance: Mapping[str, Any]
    pairing_block: str
    manipulation_check: Mapping[str, Any]
    recipe: Mapping[str, Any]
    donor_stimulus_id: Optional[str]
    offset_ms: Optional[int]
    duration_ms: int
    audio_facts: Mapping[str, Any]
    video_facts: Mapping[str, Any]
    processor_path: str
    token_layout_fingerprint: str
    provenance: Mapping[str, Any]
    checksum: str

    FIELDS = (
        "schema_version", "stimulus_id", "pair_id", "parent_content_sha256",
        "condition_code", "technical_operation", "congruence_level",
        "congruence_coding_provenance", "pairing_block", "manipulation_check",
        "recipe", "donor_stimulus_id", "offset_ms", "duration_ms",
        "audio_facts", "video_facts", "processor_path",
        "token_layout_fingerprint", "provenance", "checksum",
    )

    def __post_init__(self) -> None:
        if self.schema_version != STIMULUS_VARIANT_SCHEMA:
            raise ValueError("unsupported stimulus variant schema version")
        if not _STIMULUS_ID_RE.fullmatch(self.stimulus_id or ""):
            raise ValueError("stimulus_id has an invalid format")
        if not _SHA256_RE.fullmatch(self.parent_content_sha256 or ""):
            raise ValueError("parent_content_sha256 has an invalid format")
        if self.technical_operation not in KNOWN_OPERATIONS:
            raise ValueError("unknown technical_operation")
        for name in (
            "pair_id", "condition_code", "congruence_level", "pairing_block",
            "processor_path", "token_layout_fingerprint",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("{} must be non-empty text".format(name))
        if not self.congruence_coding_provenance or not self.manipulation_check:
            raise ValueError("congruence provenance and manipulation check are required")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.technical_operation == "audio_swap_duration_matched" and not self.donor_stimulus_id:
            raise ValueError("audio swap requires donor_stimulus_id")
        if self.technical_operation == "temporal_offset" and self.offset_ms is None:
            raise ValueError("temporal_offset requires signed offset_ms")
        if self.technical_operation in SIGNAL_CONTROL_OPERATIONS and "omitted" in self.processor_path:
            raise ValueError("signal controls cannot use a modality-omission processor path")
        recipe = normalize_recipe(self.recipe)
        if recipe.get("technical_operation") != self.technical_operation:
            raise ValueError("recipe technical_operation does not match the record")
        for field_name, expected in (
            ("condition_code", self.condition_code),
            ("donor_stimulus_id", self.donor_stimulus_id),
            ("offset_ms", self.offset_ms),
        ):
            if recipe.get(field_name) != expected:
                raise ValueError("recipe {} does not match the record".format(field_name))
        transform_schema = recipe.get("transform_schema_version")
        if not isinstance(transform_schema, str) or not transform_schema.strip():
            raise ValueError("recipe requires transform_schema_version")
        expected_id = stimulus_id_for(
            self.parent_content_sha256, recipe, transform_schema
        )
        if self.stimulus_id != expected_id:
            raise ValueError("stimulus_id does not match source, recipe, and schema")
        if self.checksum != variant_checksum(self):
            raise ValueError("stimulus checksum mismatch")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stimulus_id": self.stimulus_id,
            "pair_id": self.pair_id,
            "parent_content_sha256": self.parent_content_sha256,
            "condition_code": self.condition_code,
            "technical_operation": self.technical_operation,
            "congruence_level": self.congruence_level,
            "congruence_coding_provenance": dict(self.congruence_coding_provenance),
            "pairing_block": self.pairing_block,
            "manipulation_check": dict(self.manipulation_check),
            "recipe": dict(self.recipe),
            "donor_stimulus_id": self.donor_stimulus_id,
            "offset_ms": self.offset_ms,
            "duration_ms": self.duration_ms,
            "audio_facts": dict(self.audio_facts),
            "video_facts": dict(self.video_facts),
            "processor_path": self.processor_path,
            "token_layout_fingerprint": self.token_layout_fingerprint,
            "provenance": dict(self.provenance),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StimulusVariant":
        _require_exact_fields(data, cls.FIELDS)
        return cls(
            schema_version=data["schema_version"],
            stimulus_id=data["stimulus_id"],
            pair_id=data["pair_id"],
            parent_content_sha256=data["parent_content_sha256"],
            condition_code=data["condition_code"],
            technical_operation=data["technical_operation"],
            congruence_level=data["congruence_level"],
            congruence_coding_provenance=dict(data["congruence_coding_provenance"]),
            pairing_block=data["pairing_block"],
            manipulation_check=dict(data["manipulation_check"]),
            recipe=dict(data["recipe"]),
            donor_stimulus_id=data["donor_stimulus_id"],
            offset_ms=data["offset_ms"],
            duration_ms=data["duration_ms"],
            audio_facts=dict(data["audio_facts"]),
            video_facts=dict(data["video_facts"]),
            processor_path=data["processor_path"],
            token_layout_fingerprint=data["token_layout_fingerprint"],
            provenance=dict(data["provenance"]),
            checksum=data["checksum"],
        )


def variant_checksum(variant: StimulusVariant) -> str:
    data = variant.to_dict()
    data.pop("checksum", None)
    return "sha256:" + sha256(_canonical_json_bytes(data)).hexdigest()


def build_stimulus_variant(
    *,
    parent_content_sha256: str,
    pair_id: str,
    condition_code: str,
    technical_operation: str,
    congruence_level: str,
    congruence_coding_provenance: Mapping[str, Any],
    pairing_block: str,
    manipulation_check: Mapping[str, Any],
    parameters: Mapping[str, Any],
    donor_stimulus_id: Optional[str],
    offset_ms: Optional[int],
    duration_ms: int,
    audio_facts: Mapping[str, Any],
    video_facts: Mapping[str, Any],
    processor_path: str,
    token_layout_fingerprint: str,
    provenance: Mapping[str, Any],
    transform_schema_version: str = TRANSFORM_SCHEMA,
) -> StimulusVariant:
    if technical_operation not in KNOWN_OPERATIONS:
        raise ValueError("unknown technical_operation")
    recipe = normalize_recipe(
        {
            "transform_schema_version": transform_schema_version,
            "technical_operation": technical_operation,
            "condition_code": condition_code,
            "donor_stimulus_id": donor_stimulus_id,
            "offset_ms": offset_ms,
            "parameters": parameters,
        }
    )
    stimulus_id = stimulus_id_for(
        parent_content_sha256, recipe, transform_schema_version
    )
    values = {
        "schema_version": STIMULUS_VARIANT_SCHEMA,
        "stimulus_id": stimulus_id,
        "pair_id": pair_id,
        "parent_content_sha256": parent_content_sha256,
        "condition_code": condition_code,
        "technical_operation": technical_operation,
        "congruence_level": congruence_level,
        "congruence_coding_provenance": dict(congruence_coding_provenance),
        "pairing_block": pairing_block,
        "manipulation_check": dict(manipulation_check),
        "recipe": recipe,
        "donor_stimulus_id": donor_stimulus_id,
        "offset_ms": offset_ms,
        "duration_ms": duration_ms,
        "audio_facts": dict(audio_facts),
        "video_facts": dict(video_facts),
        "processor_path": processor_path,
        "token_layout_fingerprint": token_layout_fingerprint,
        "provenance": dict(provenance),
        "checksum": "",
    }
    draft = object.__new__(StimulusVariant)
    for key, value in values.items():
        object.__setattr__(draft, key, value)
    checksum = variant_checksum(draft)
    return StimulusVariant(
        schema_version=STIMULUS_VARIANT_SCHEMA,
        stimulus_id=stimulus_id,
        pair_id=pair_id,
        parent_content_sha256=parent_content_sha256,
        condition_code=condition_code,
        technical_operation=technical_operation,
        congruence_level=congruence_level,
        congruence_coding_provenance=dict(congruence_coding_provenance),
        pairing_block=pairing_block,
        manipulation_check=dict(manipulation_check),
        recipe=recipe,
        donor_stimulus_id=donor_stimulus_id,
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        audio_facts=dict(audio_facts),
        video_facts=dict(video_facts),
        processor_path=processor_path,
        token_layout_fingerprint=token_layout_fingerprint,
        provenance=dict(provenance),
        checksum=checksum,
    )


@dataclass(frozen=True)
class StimulusManifest:
    manifest_schema_version: str
    release_status: str
    variants: Tuple[StimulusVariant, ...]
    checksum: str

    FIELDS = ("manifest_schema_version", "release_status", "variants", "checksum")

    def __post_init__(self) -> None:
        if self.manifest_schema_version != STIMULUS_MANIFEST_SCHEMA:
            raise ValueError("unsupported stimulus manifest schema version")
        if self.release_status not in {"draft", "blocked", "course_authorized", "research_authorized"}:
            raise ValueError("invalid release_status")
        ids = [variant.stimulus_id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("stimulus IDs must be unique")
        if self.checksum != manifest_checksum(self):
            raise ValueError("stimulus manifest checksum mismatch")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "release_status": self.release_status,
            "variants": [variant.to_dict() for variant in self.variants],
            "checksum": self.checksum,
        }

    @classmethod
    def build(cls, variants: Sequence[StimulusVariant], release_status: str = "blocked") -> "StimulusManifest":
        variant_tuple = tuple(variants)
        values = {
            "manifest_schema_version": STIMULUS_MANIFEST_SCHEMA,
            "release_status": release_status,
            "variants": variant_tuple,
            "checksum": "",
        }
        draft = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(draft, key, value)
        checksum = manifest_checksum(draft)
        return cls(
            manifest_schema_version=STIMULUS_MANIFEST_SCHEMA,
            release_status=release_status,
            variants=variant_tuple,
            checksum=checksum,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StimulusManifest":
        _require_exact_fields(data, cls.FIELDS)
        return cls(
            manifest_schema_version=data["manifest_schema_version"],
            release_status=data["release_status"],
            variants=tuple(StimulusVariant.from_dict(item) for item in data["variants"]),
            checksum=data["checksum"],
        )


def manifest_checksum(manifest: StimulusManifest) -> str:
    data = manifest.to_dict()
    data.pop("checksum", None)
    return "sha256:" + sha256(_canonical_json_bytes(data)).hexdigest()


def dump_stimulus_manifest(manifest: StimulusManifest) -> bytes:
    return _canonical_json_bytes(manifest.to_dict())


def load_stimulus_manifest(payload: bytes) -> StimulusManifest:
    try:
        data = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("stimulus manifest is not valid UTF-8 JSON") from exc
    if not isinstance(data, Mapping):
        raise ValueError("stimulus manifest root must be an object")
    return StimulusManifest.from_dict(data)
