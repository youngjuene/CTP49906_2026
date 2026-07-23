"""True processor/model modality-omission contracts.

Signal controls transform supplied media and attention-edge knockout changes model
information flow.  Neither may enter this API.  These helpers build distinct
processor paths in which one modality is not supplied at all and record a token-
layout fingerprint for compatible aggregate/final-output comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .probe_metrics import build_token_layout_fingerprint


OMISSION_SCHEMA_VERSION = "modality-omission/1.0.0"
_OPERATIONS = {
    "audio_omitted_model_input": {
        "absent": "audio",
        "present": "video",
        "content_type": "video",
        "processor_path": "processor.video_without_audio",
    },
    "video_omitted_model_input": {
        "absent": "video",
        "present": "audio",
        "content_type": "audio",
        "processor_path": "processor.audio_only",
    },
}
_EVIDENCE_SCOPES = {"local_only", "target_runtime"}


@dataclass(frozen=True)
class ModalityOmissionRequest:
    schema_version: str
    technical_operation: str
    source: str
    absent_modality: str
    present_modality: str
    processor_path: str
    nframes: int | None
    evidence_scope: str
    target_runtime_verified: bool


@dataclass(frozen=True)
class PreparedModalityOmission:
    request: ModalityOmissionRequest
    inputs: Mapping[str, Any]
    processor_path: str
    modality_presence: Mapping[str, bool]
    token_layout_fingerprint: str
    research_ready: bool
    comparison_scope: str


def build_modality_omission_request(
    technical_operation: str,
    *,
    source: str,
    nframes: int | None = None,
    evidence_scope: str = "local_only",
    target_runtime_verified: bool = False,
) -> ModalityOmissionRequest:
    """Create only a true omission request, never a signal/model intervention."""

    if technical_operation not in _OPERATIONS:
        raise ValueError(
            f"{technical_operation!r} is not a true modality-omission operation; "
            "signal controls and attention-edge knockout require separate APIs."
        )
    if not str(source).strip():
        raise ValueError("source is required for the modality that remains present")
    if nframes is not None and nframes < 1:
        raise ValueError("nframes must be >= 1 when supplied")
    if evidence_scope not in _EVIDENCE_SCOPES:
        raise ValueError(f"Unsupported omission evidence scope: {evidence_scope!r}")
    facts = _OPERATIONS[technical_operation]
    return ModalityOmissionRequest(
        schema_version=OMISSION_SCHEMA_VERSION,
        technical_operation=technical_operation,
        source=str(source),
        absent_modality=facts["absent"],
        present_modality=facts["present"],
        processor_path=facts["processor_path"],
        nframes=nframes,
        evidence_scope=evidence_scope,
        target_runtime_verified=bool(target_runtime_verified),
    )


def _content_for(request: ModalityOmissionRequest) -> dict[str, Any]:
    facts = _OPERATIONS[request.technical_operation]
    content: dict[str, Any] = {
        "type": facts["content_type"],
        facts["content_type"]: request.source,
    }
    if facts["content_type"] == "video" and request.nframes is not None:
        content["nframes"] = request.nframes
    return content


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def _input_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else []


def prepare_modality_omission_inputs(
    request: ModalityOmissionRequest,
    *,
    prompt: str,
    processor: Any,
    process_mm_info: Callable[..., tuple[Any, Any, Any]],
    processor_revision: str,
    model_revision: str,
    template_version: str = "qwen-chat-template/unknown",
    audio_token_index: int | None = None,
    video_token_index: int | None = None,
) -> PreparedModalityOmission:
    """Run the exact audio-absent or video-absent processor path and attest it."""

    if request.technical_operation not in _OPERATIONS:
        raise ValueError("request is not a supported true modality-omission contract")
    if not prompt.strip():
        raise ValueError("prompt is required")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                _content_for(request),
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=False,
    )
    if request.absent_modality == "audio" and _present(audios):
        raise ValueError("Audio-omitted path unexpectedly produced audio features")
    if request.absent_modality == "video" and _present(videos):
        raise ValueError("Video-omitted path unexpectedly produced video features")
    if request.present_modality == "audio" and not _present(audios):
        raise ValueError("Video-omitted path did not produce the required audio input")
    if request.present_modality == "video" and not _present(videos):
        raise ValueError("Audio-omitted path did not produce the required video input")

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    input_keys = set(inputs)
    if request.absent_modality == "audio":
        forbidden = {
            key
            for key in input_keys
            if "audio" in key or key in {"input_features", "feature_attention_mask"}
        }
    else:
        forbidden = {key for key in input_keys if "video" in key}
    if forbidden:
        raise ValueError(
            f"{request.absent_modality.title()}-omitted processor inputs contain "
            f"forbidden modality fields: {sorted(forbidden)}"
        )

    input_ids_tensor = torch.as_tensor(inputs["input_ids"]).detach().cpu().reshape(-1)
    input_ids = [int(item) for item in input_ids_tensor.tolist()]
    modality_by_position = ["unknown"] * len(input_ids)
    for position, token_id in enumerate(input_ids):
        if audio_token_index is not None and token_id == int(audio_token_index):
            modality_by_position[position] = "audio"
        if video_token_index is not None and token_id == int(video_token_index):
            if modality_by_position[position] != "unknown":
                modality_by_position[position] = "unknown"
            else:
                modality_by_position[position] = "video"

    if request.absent_modality in modality_by_position:
        raise ValueError(
            f"{request.absent_modality.title()} token remains in a true omission path"
        )
    if request.present_modality not in modality_by_position:
        raise ValueError(
            f"Processor metadata did not expose the required {request.present_modality} token"
        )

    fingerprint = build_token_layout_fingerprint(
        input_ids=input_ids,
        modality_by_position=modality_by_position,
        processor_revision=processor_revision,
        model_revision=model_revision,
        template_version=template_version,
        input_shapes={key: _input_shape(value) for key, value in inputs.items()},
    )
    modality_presence = {
        "audio": request.present_modality == "audio",
        "video": request.present_modality == "video",
    }
    research_ready = (
        request.evidence_scope == "target_runtime" and request.target_runtime_verified
    )
    return PreparedModalityOmission(
        request=request,
        inputs=inputs,
        processor_path=request.processor_path,
        modality_presence=modality_presence,
        token_layout_fingerprint=fingerprint,
        research_ready=research_ready,
        comparison_scope="aggregate_or_final_output_only",
    )
