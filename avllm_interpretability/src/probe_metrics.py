"""Compact, versioned distribution summaries for AVLLM probes.

The functions in this module deliberately return JSON-safe summaries rather than
full-vocabulary logits.  Intermediate logit-lens measurements and final
teacher-forced answer distributions carry different measurement kinds and
caveats; neither is a calibration or free-generation uncertainty estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


PROBE_METRIC_SCHEMA_VERSION = "probe-metric/1.0.0"
DISTRIBUTION_SCHEMA_VERSION = "compact-distribution/1.0.0"
CACHE_SCHEMA_VERSION = "analysis-cache/1.0.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MeasurementKind(str, Enum):
    """Honest labels for the two supported distribution families."""

    RAW_PROBE_SCORE_DISPERSION = "raw_probe_score_dispersion"
    TEACHER_FORCED_ANSWER_DISTRIBUTION_DISPERSION = (
        "teacher_forced_answer_distribution_dispersion"
    )
    LEGACY_ARGMAX_ONLY = "legacy_argmax_only"


MEASUREMENT_CAVEATS = {
    MeasurementKind.RAW_PROBE_SCORE_DISPERSION: (
        "Raw probe score dispersion from the language head without the model's "
        "final normalization; it is not calibrated belief or free-generation uncertainty."
    ),
    MeasurementKind.TEACHER_FORCED_ANSWER_DISTRIBUTION_DISPERSION: (
        "Teacher-forced answer-distribution dispersion (uncalibrated proxy); it is "
        "not free-generation uncertainty or calibration."
    ),
    MeasurementKind.LEGACY_ARGMAX_ONLY: (
        "Legacy argmax-only artifact; entropy, margin, and target rank were not measured."
    ),
}


def _measurement_kind(value: MeasurementKind | str) -> MeasurementKind:
    try:
        return value if isinstance(value, MeasurementKind) else MeasurementKind(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported measurement kind: {value!r}") from exc


def metric_version(
    measurement_kind: MeasurementKind | str,
    *,
    accumulation_dtype: str = "float32",
    target_token_set_version: str | None = None,
) -> dict[str, Any]:
    """Return the complete metric-definition record required by the PRD."""

    kind = _measurement_kind(measurement_kind)
    probe_transform = (
        "lm_head_without_final_normalization"
        if kind is MeasurementKind.RAW_PROBE_SCORE_DISPERSION
        else "teacher_forced_final_answer_logits"
        if kind is MeasurementKind.TEACHER_FORCED_ANSWER_DISTRIBUTION_DISPERSION
        else "legacy_argmax_only"
    )
    claim_scope = (
        "score_dispersion"
        if kind is MeasurementKind.RAW_PROBE_SCORE_DISPERSION
        else "uncalibrated_proxy"
        if kind is MeasurementKind.TEACHER_FORCED_ANSWER_DISTRIBUTION_DISPERSION
        else "not_measured"
    )
    version = {
        "schema_version": PROBE_METRIC_SCHEMA_VERSION,
        "measurement_kind": kind.value,
        "entropy_units": "nats",
        "entropy_normalization": "H/log(vocab_size)",
        "margin_definition": "top1_minus_top2_log_probability",
        "tie_policy": "log_probability_desc_then_token_id_asc",
        "rank_policy": "one_indexed_deterministic_order",
        "accumulation_dtype": accumulation_dtype,
        "vocabulary_support": "full_vocabulary",
        "probe_transform": probe_transform,
        "target_token_set_version": target_token_set_version,
        "claim_scope": claim_scope,
    }
    validate_metric_version(version)
    return version


def validate_metric_version(version: Mapping[str, Any]) -> None:
    """Validate complete definitions and reject unsupported inference claims."""

    required = {
        "schema_version",
        "measurement_kind",
        "entropy_units",
        "entropy_normalization",
        "margin_definition",
        "tie_policy",
        "rank_policy",
        "accumulation_dtype",
        "vocabulary_support",
        "probe_transform",
        "target_token_set_version",
        "claim_scope",
    }
    missing = sorted(required.difference(version))
    if missing:
        raise ValueError(f"Metric version is missing required fields: {missing}")
    if version["schema_version"] != PROBE_METRIC_SCHEMA_VERSION:
        raise ValueError(f"Unsupported metric schema: {version['schema_version']!r}")
    kind = _measurement_kind(str(version["measurement_kind"]))
    allowed_claim = {
        MeasurementKind.RAW_PROBE_SCORE_DISPERSION: "score_dispersion",
        MeasurementKind.TEACHER_FORCED_ANSWER_DISTRIBUTION_DISPERSION: (
            "uncalibrated_proxy"
        ),
        MeasurementKind.LEGACY_ARGMAX_ONLY: "not_measured",
    }[kind]
    claim_scope = str(version["claim_scope"])
    if claim_scope != allowed_claim:
        raise ValueError(
            f"Unsupported {claim_scope!r} claim for {kind.value}; calibration, "
            "free-generation uncertainty, and truncated-support JS claims are not available."
        )
    expected = {
        "entropy_units": "nats",
        "entropy_normalization": "H/log(vocab_size)",
        "margin_definition": "top1_minus_top2_log_probability",
        "tie_policy": "log_probability_desc_then_token_id_asc",
        "rank_policy": "one_indexed_deterministic_order",
        "vocabulary_support": "full_vocabulary",
    }
    mismatches = {
        key: version.get(key) for key, value in expected.items() if version.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported metric-definition values: {mismatches}")
    if not str(version["accumulation_dtype"]).strip():
        raise ValueError("accumulation_dtype must be recorded")
    if not str(version["probe_transform"]).strip():
        raise ValueError("probe_transform must be recorded")


@dataclass(frozen=True)
class TokenScore:
    token_id: int
    token_text: str | None
    log_probability: float
    probability: float


@dataclass(frozen=True)
class TargetTokenScore:
    token_id: int
    found: bool
    rank: int | None
    log_probability: float | None


@dataclass(frozen=True)
class DistributionSummary:
    schema_version: str
    measurement_kind: str
    caveat: str
    vocabulary_size: int
    top_tokens: tuple[TokenScore, ...]
    entropy_nats: float
    normalized_entropy: float | None
    log_probability_margin: float | None
    targets: tuple[TargetTokenScore, ...]
    metric_version: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_logits(
    logits: torch.Tensor | Sequence[float],
    *,
    top_k: int = 5,
    target_token_ids: Iterable[int] | None = None,
    measurement_kind: MeasurementKind | str = MeasurementKind.RAW_PROBE_SCORE_DISPERSION,
    token_decoder: Callable[[int], str] | None = None,
    version: Mapping[str, Any] | None = None,
) -> DistributionSummary:
    """Summarize one full-vocabulary logit row without retaining the row."""

    row = torch.as_tensor(logits).detach()
    if row.ndim != 1:
        raise ValueError(f"Expected one logit row, got shape {tuple(row.shape)}")
    if row.numel() == 0:
        raise ValueError("Logit row must contain at least one vocabulary item")
    row = row.float()
    if not bool(torch.isfinite(row).all()):
        raise ValueError("All logits must be finite")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    kind = _measurement_kind(measurement_kind)
    version_record = dict(version or metric_version(kind))
    if version_record.get("measurement_kind") != kind.value:
        raise ValueError("Metric version measurement_kind does not match the summary")
    validate_metric_version(version_record)

    log_probabilities = torch.log_softmax(row, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = float(-(probabilities * log_probabilities).sum().item())
    vocab_size = int(row.numel())
    normalized_entropy = entropy / math.log(vocab_size) if vocab_size > 1 else None

    # Stable descending argsort keeps the original token-id order for exact ties.
    # Sorting and rank comparisons stay on-device; only compact top-k/target values
    # cross to CPU.
    ordered_ids = torch.argsort(log_probabilities, descending=True, stable=True)
    selected_ids = ordered_ids[: min(top_k, vocab_size)].cpu().tolist()
    selected_log_probabilities = log_probabilities[selected_ids].cpu().tolist()
    top_tokens = tuple(
        TokenScore(
            token_id=token_id,
            token_text=token_decoder(token_id) if token_decoder else None,
            log_probability=float(log_probability),
            probability=float(math.exp(float(log_probability))),
        )
        for token_id, log_probability in zip(selected_ids, selected_log_probabilities)
    )
    margin = (
        float(
            (
                log_probabilities[ordered_ids[0]]
                - log_probabilities[ordered_ids[1]]
            ).item()
        )
        if vocab_size > 1
        else None
    )
    token_ids = torch.arange(vocab_size, device=log_probabilities.device)
    target_scores = []
    for raw_token_id in (() if target_token_ids is None else target_token_ids):
        token_id = int(raw_token_id)
        found = 0 <= token_id < vocab_size
        if not found:
            target_scores.append(TargetTokenScore(token_id, False, None, None))
            continue
        target_log_probability = log_probabilities[token_id]
        greater = (log_probabilities > target_log_probability).sum()
        tied_before = (
            (log_probabilities == target_log_probability) & (token_ids < token_id)
        ).sum()
        target_scores.append(
            TargetTokenScore(
                token_id=token_id,
                found=True,
                rank=1 + int(greater.item()) + int(tied_before.item()),
                log_probability=float(target_log_probability.item()),
            )
        )
    targets = tuple(target_scores)

    return DistributionSummary(
        schema_version=DISTRIBUTION_SCHEMA_VERSION,
        measurement_kind=kind.value,
        caveat=MEASUREMENT_CAVEATS[kind],
        vocabulary_size=vocab_size,
        top_tokens=top_tokens,
        entropy_nats=entropy,
        normalized_entropy=normalized_entropy,
        log_probability_margin=margin,
        targets=targets,
        metric_version=version_record,
    )


def map_position_modalities(
    sequence_length: int,
    *,
    audio_positions: Iterable[int] = (),
    video_positions: Iterable[int] = (),
    answer_positions: Iterable[int] = (),
) -> list[str]:
    """Map processor-declared positions, leaving absent/overlapping facts unknown."""

    if sequence_length < 0:
        raise ValueError("sequence_length must be >= 0")
    claims: list[set[str]] = [set() for _ in range(sequence_length)]
    for modality, positions in (
        ("audio", audio_positions),
        ("video", video_positions),
        ("answer", answer_positions),
    ):
        for raw_position in positions:
            position = int(raw_position)
            if not 0 <= position < sequence_length:
                raise ValueError(
                    f"{modality} position {position} outside sequence length {sequence_length}"
                )
            claims[position].add(modality)
    return [next(iter(item)) if len(item) == 1 else "unknown" for item in claims]


@dataclass(frozen=True)
class TrajectoryAlignment:
    allowed: bool
    mode: str
    warning: str
    pairs: tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]


def _trajectory_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(row["layer"]), int(row["position"]), str(row.get("modality", "unknown"))


def align_trajectories(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    left_fingerprint: str,
    right_fingerprint: str,
    *,
    requested_mode: str = "position",
) -> TrajectoryAlignment:
    """Align exact positions only for identical processor/token layouts."""

    if requested_mode not in {"position", "aggregate", "normalized_bin"}:
        raise ValueError(f"Unsupported trajectory alignment mode: {requested_mode!r}")
    same_layout = bool(left_fingerprint) and left_fingerprint == right_fingerprint
    if requested_mode == "position" and not same_layout:
        return TrajectoryAlignment(
            allowed=False,
            mode="blocked",
            warning=(
                "Token-layout fingerprints differ; position-wise comparison is blocked. "
                "Use a preregistered aggregate or normalized-bin estimand."
            ),
            pairs=(),
        )

    left_by_key = {_trajectory_key(item): item for item in left}
    right_by_key = {_trajectory_key(item): item for item in right}
    if requested_mode == "position":
        if set(left_by_key) != set(right_by_key):
            return TrajectoryAlignment(
                allowed=False,
                mode="blocked",
                warning="Matching fingerprints have different recorded position identities.",
                pairs=(),
            )
        keys = sorted(left_by_key)
        pairs = tuple((left_by_key[key], right_by_key[key]) for key in keys)
        return TrajectoryAlignment(True, "position", "", pairs)

    left_rows = sorted(left, key=_trajectory_key)
    right_rows = sorted(right, key=_trajectory_key)
    warning = (
        "Token-layout fingerprints differ; position-wise comparison is unavailable and "
        f"the {requested_mode} estimand must remain preregistered."
        if not same_layout
        else ""
    )
    return TrajectoryAlignment(
        allowed=True,
        mode=requested_mode,
        warning=warning,
        pairs=tuple(zip(left_rows, right_rows)),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CacheIdentity:
    schema_version: str
    key: str
    semantic_inputs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_analysis_cache_identity(
    *,
    content_sha256: str,
    normalized_recipe: Mapping[str, Any],
    processor_revision: str,
    model_revision: str,
    prompt: str,
    frame_settings: Mapping[str, Any],
    analysis_parameters: Mapping[str, Any],
    display_labels: Mapping[str, Any] | None = None,
) -> CacheIdentity:
    """Hash every semantic input while intentionally ignoring display labels."""

    del display_labels
    content_sha256 = content_sha256.lower()
    if not _SHA256.fullmatch(content_sha256):
        raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest")
    semantic_inputs = {
        "content_sha256": content_sha256,
        "normalized_recipe": normalized_recipe,
        "processor_revision": str(processor_revision),
        "model_revision": str(model_revision),
        "prompt": str(prompt),
        "frame_settings": frame_settings,
        "analysis_parameters": analysis_parameters,
    }
    empty = [
        field
        for field in ("processor_revision", "model_revision", "prompt")
        if not str(semantic_inputs[field]).strip()
    ]
    if empty:
        raise ValueError(f"Cache identity requires non-empty semantic inputs: {empty}")
    digest = hashlib.sha256(_canonical_json(semantic_inputs).encode("utf-8")).hexdigest()
    return CacheIdentity(CACHE_SCHEMA_VERSION, digest, semantic_inputs)


def build_token_layout_fingerprint(
    *,
    input_ids: Sequence[int],
    modality_by_position: Sequence[str],
    processor_revision: str,
    model_revision: str,
    template_version: str,
    input_shapes: Mapping[str, Sequence[int]],
) -> str:
    """Hash exact token/modality layout facts without serializing feature values."""

    if len(input_ids) != len(modality_by_position):
        raise ValueError("input_ids and modality_by_position must have equal length")
    payload = {
        "input_ids": [int(value) for value in input_ids],
        "modality_by_position": list(modality_by_position),
        "processor_revision": processor_revision,
        "model_revision": model_revision,
        "template_version": template_version,
        "input_shapes": {key: list(value) for key, value in sorted(input_shapes.items())},
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
