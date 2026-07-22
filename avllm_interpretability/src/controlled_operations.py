"""Execute controlled operations without collapsing their semantic boundaries.

Media transforms, presented-signal controls, processor modality omission, and
attention-edge knockout are distinct intervention families.  This dispatcher
provides one integration point while preserving those distinctions in both the
returned payload and deterministic provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Callable, Mapping, Sequence

from .stimulus_variants import (
    KNOWN_OPERATIONS,
    audio_silence_control,
    canonical_recipe_bytes,
    duration_matched_audio_swap,
    neutral_video_control,
    offset_samples_zero_fill,
)


CONTROLLED_OPERATION_SCHEMA = "controlled-operation-dispatch/1.0.0"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_TYPES = frozenset(
    {"query_text", "audio", "video", "image", "answer", "generated"}
)


@dataclass(frozen=True)
class AttentionEdgeExecution:
    """Validated rules that enter the existing hook boundary only on ``context``."""

    knockout_rules: tuple[tuple[str, str, int, int], ...]

    def context(
        self,
        model: Any,
        *,
        token_types: Sequence[str],
        original_input_len: int,
        track_attention: bool = False,
        capture_layer_range: tuple[int, int] | None = None,
    ) -> Any:
        from .attention_knockout_experiment import block_attention

        return block_attention(
            model,
            list(self.knockout_rules),
            list(token_types),
            original_input_len,
            track_attention=track_attention,
            capture_layer_range=capture_layer_range,
        )


@dataclass(frozen=True)
class ControlledOperationRequest:
    """Inputs for exactly one declared technical operation.

    ``source_reference`` is a private local processor reference.  It is used for
    true-omission requests but deliberately excluded from stable provenance.
    Content identity comes from ``source_content_sha256``, never a filename.
    """

    technical_operation: str
    source_content_sha256: str
    audio_samples: Sequence[float]
    video_frames: Sequence[Any]
    sample_rate_hz: int
    source_reference: str
    token_layout_fingerprint: str = "pending:pre-model"
    donor_audio_samples: Sequence[float] | None = None
    donor_sample_rate_hz: int | None = None
    donor_content_sha256: str | None = None
    offset_ms: int | None = None
    neutral_video_value: Any = 0
    omission_nframes: int | None = None
    knockout_rules: Sequence[tuple[str, str, int, int]] = ()


@dataclass(frozen=True)
class ControlledOperationResult:
    technical_operation: str
    intervention_kind: str
    audio_samples: tuple[float, ...] | None
    video_frames: tuple[Any, ...] | None
    modality_presence: Mapping[str, bool]
    omission_request: Any | None
    knockout_rules: tuple[tuple[str, str, int, int], ...]
    edge_execution: Any | None
    token_layout_fingerprint: str | None
    result_digest: str
    provenance: Mapping[str, Any]


def _require_hash(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _normalize_knockout_rules(
    rules: Sequence[tuple[str, str, int, int]],
) -> tuple[tuple[str, str, int, int], ...]:
    normalized = []
    for rule in rules:
        if not isinstance(rule, (tuple, list)) or len(rule) != 4:
            raise ValueError(
                "knockout_rules entries must be (source, target, start_layer, end_layer)"
            )
        source, target, start, end = rule
        if source not in _TOKEN_TYPES or target not in _TOKEN_TYPES:
            raise ValueError(
                f"knockout_rules token types must be one of {sorted(_TOKEN_TYPES)}"
            )
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(
                "knockout_rules layer bounds must satisfy 0 <= start_layer < end_layer"
            )
        normalized.append((source, target, start, end))
    return tuple(normalized)


def _provenance(
    request: ControlledOperationRequest,
    *,
    intervention_kind: str,
    audio_count: int,
    video_count: int,
    knockout_rules: tuple[tuple[str, str, int, int], ...],
) -> dict[str, Any]:
    semantic_inputs = {
        "schema_version": CONTROLLED_OPERATION_SCHEMA,
        "technical_operation": request.technical_operation,
        "intervention_kind": intervention_kind,
        "source_content_sha256": request.source_content_sha256,
        "donor_content_sha256": request.donor_content_sha256,
        "requested_token_layout_fingerprint": request.token_layout_fingerprint,
        "sample_rate_hz": request.sample_rate_hz,
        "source_audio_sample_count": audio_count,
        "source_video_frame_count": video_count,
        "donor_sample_rate_hz": request.donor_sample_rate_hz,
        "donor_audio_sample_count": (
            None
            if request.donor_audio_samples is None
            else len(tuple(request.donor_audio_samples))
        ),
        "offset_ms": request.offset_ms,
        "neutral_video_value": request.neutral_video_value,
        "omission_nframes": request.omission_nframes,
        "knockout_rules": knockout_rules,
    }
    digest = sha256(canonical_recipe_bytes(semantic_inputs)).hexdigest()
    return {
        **semantic_inputs,
        "dispatch_digest": "sha256:" + digest,
        "evidence_scope": "local_only",
        "target_runtime_verified": False,
    }


def _default_omission_executor(
    operation: str, request: ControlledOperationRequest
) -> Any:
    from .modality_omission import build_modality_omission_request

    return build_modality_omission_request(
        operation,
        source=request.source_reference,
        nframes=(
            request.omission_nframes
            if operation == "audio_omitted_model_input"
            else None
        ),
        evidence_scope="local_only",
        target_runtime_verified=False,
    )


def _default_edge_executor(
    rules: tuple[tuple[str, str, int, int], ...],
    _request: ControlledOperationRequest,
) -> AttentionEdgeExecution:
    return AttentionEdgeExecution(rules)


def _dispatch(
    request: ControlledOperationRequest,
    *,
    omission_executor: Callable[[str, ControlledOperationRequest], Any],
    edge_executor: Callable[
        [tuple[tuple[str, str, int, int], ...], ControlledOperationRequest], Any
    ],
) -> ControlledOperationResult:
    if request.technical_operation not in KNOWN_OPERATIONS:
        raise ValueError(f"unknown technical_operation: {request.technical_operation!r}")
    _require_hash("source_content_sha256", request.source_content_sha256)
    _require_positive_integer("sample_rate_hz", request.sample_rate_hz)
    if (
        not isinstance(request.token_layout_fingerprint, str)
        or not request.token_layout_fingerprint.strip()
    ):
        raise ValueError("token_layout_fingerprint must be non-empty text")

    audio = tuple(request.audio_samples)
    video = tuple(request.video_frames)
    operation = request.technical_operation
    omission_request = None
    knockout_rules: tuple[tuple[str, str, int, int], ...] = ()
    edge_execution = None

    if operation == "original_reference":
        intervention_kind = "media_transform"
        output_audio, output_video = audio, video
    elif operation == "audio_swap_duration_matched":
        intervention_kind = "media_transform"
        if request.donor_audio_samples is None:
            raise ValueError("audio swap requires donor_audio_samples")
        if request.donor_sample_rate_hz is None:
            raise ValueError("audio swap requires donor_sample_rate_hz")
        _require_positive_integer("donor_sample_rate_hz", request.donor_sample_rate_hz)
        _require_hash("donor_content_sha256", request.donor_content_sha256)
        output_audio, _normalization = duration_matched_audio_swap(
            request.donor_audio_samples,
            request.donor_sample_rate_hz,
            len(audio),
            processor_sample_rate_hz=request.sample_rate_hz,
        )
        output_video = video
    elif operation == "temporal_offset":
        intervention_kind = "media_transform"
        if request.offset_ms is None:
            raise ValueError("temporal_offset requires signed offset_ms")
        if isinstance(request.offset_ms, bool) or not isinstance(request.offset_ms, int):
            raise ValueError("offset_ms must be an integer")
        offset_samples = int(round(request.offset_ms * request.sample_rate_hz / 1000.0))
        output_audio = offset_samples_zero_fill(audio, offset_samples)
        output_video = video
    elif operation == "audio_silence_control":
        intervention_kind = "signal_control"
        output_audio = audio_silence_control(len(audio))
        output_video = video
    elif operation == "video_neutral_control":
        intervention_kind = "signal_control"
        output_audio = audio
        output_video = neutral_video_control(
            len(video), request.neutral_video_value
        )
    elif operation in {
        "audio_omitted_model_input",
        "video_omitted_model_input",
    }:
        intervention_kind = "modality_omission"
        if not isinstance(request.source_reference, str) or not request.source_reference.strip():
            raise ValueError("true modality omission requires source_reference")
        nframes = (
            request.omission_nframes
            if operation == "audio_omitted_model_input"
            else None
        )
        if nframes is not None and (
            isinstance(nframes, bool) or not isinstance(nframes, int) or nframes < 1
        ):
            raise ValueError("omission_nframes must be >= 1 when supplied")
        if operation == "audio_omitted_model_input":
            if not video:
                raise ValueError("audio omission requires a present video signal")
            output_audio, output_video = None, video
        else:
            if not audio:
                raise ValueError("video omission requires a present audio signal")
            output_audio, output_video = audio, None
        omission_request = omission_executor(operation, request)
    elif operation == "direct_attention_edge_knockout":
        intervention_kind = "model_intervention"
        knockout_rules = _normalize_knockout_rules(request.knockout_rules)
        if not knockout_rules:
            raise ValueError("direct attention edge knockout requires knockout_rules")
        output_audio, output_video = audio, video
        edge_execution = edge_executor(knockout_rules, request)
    else:  # pragma: no cover - guarded by KNOWN_OPERATIONS
        raise AssertionError(f"unhandled operation: {operation}")

    provenance = _provenance(
        request,
        intervention_kind=intervention_kind,
        audio_count=len(audio),
        video_count=len(video),
        knockout_rules=knockout_rules,
    )
    layout_fingerprint = (
        None
        if intervention_kind == "modality_omission"
        else request.token_layout_fingerprint
    )
    provenance["token_layout_fingerprint"] = layout_fingerprint
    provenance["layout_status"] = (
        "requires_processor_attestation"
        if layout_fingerprint is None
        else "preserved_from_request"
    )
    return ControlledOperationResult(
        technical_operation=operation,
        intervention_kind=intervention_kind,
        audio_samples=output_audio,
        video_frames=output_video,
        modality_presence={
            "audio": output_audio is not None,
            "video": output_video is not None,
        },
        omission_request=omission_request,
        knockout_rules=knockout_rules,
        edge_execution=edge_execution,
        token_layout_fingerprint=layout_fingerprint,
        result_digest=provenance["dispatch_digest"],
        provenance=provenance,
    )


def dispatch_controlled_operation(
    request: ControlledOperationRequest,
    *,
    omission_executor: Callable[[str, ControlledOperationRequest], Any] | None = None,
    edge_executor: Callable[
        [tuple[tuple[str, str, int, int], ...], ControlledOperationRequest], Any
    ]
    | None = None,
    on_failure_cleanup: Callable[[], None] | None = None,
) -> ControlledOperationResult:
    """Execute a request atomically and invoke caller cleanup on failure.

    The dispatcher has no filesystem side effects itself.  ``on_failure_cleanup``
    lets an integrating notebook remove caller-owned decode/render resources if an
    operation fails before a result is committed.
    """

    if not isinstance(request, ControlledOperationRequest):
        raise TypeError("request must be a ControlledOperationRequest")
    try:
        return _dispatch(
            request,
            omission_executor=omission_executor or _default_omission_executor,
            edge_executor=edge_executor or _default_edge_executor,
        )
    except Exception as exc:
        if on_failure_cleanup is not None:
            try:
                on_failure_cleanup()
            except Exception as cleanup_exc:  # noqa: BLE001
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "on_failure_cleanup also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
        raise


def build_result_link_command(
    result: ControlledOperationResult,
    *,
    run_id: str,
    command_nonce: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the neutral exactly-once ``attach_result`` command.

    ``curriculum_common.session_records.reduce_command`` owns idempotency.  This
    helper supplies its stable result digest and linkage fields without mutating a
    session log or creating a second state store.
    """

    if not isinstance(result, ControlledOperationResult):
        raise TypeError("result must be a ControlledOperationResult")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty text")
    if not isinstance(command_nonce, str) or not command_nonce.strip():
        raise ValueError("command_nonce must be non-empty text")
    linked_metrics = {
        "technical_operation": result.technical_operation,
        "intervention_kind": result.intervention_kind,
        "dispatch_digest": result.result_digest,
        "token_layout_fingerprint": result.token_layout_fingerprint,
        **dict(metrics or {}),
    }
    return {
        "kind": "attach_result",
        "command_nonce": command_nonce.strip(),
        "run_id": run_id.strip(),
        "result_digest": result.result_digest,
        "metrics": linked_metrics,
        "status": "completed",
    }
