"""Processor-boundary tests for true audio/video omission."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modality_omission import (  # noqa: E402
    build_modality_omission_request,
    prepare_modality_omission_inputs,
)


class _Processor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        return "rendered prompt"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("videos"):
            return {
                "input_ids": torch.tensor([[1, 20, 2]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values_videos": torch.zeros(1, 2),
                "video_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }
        return {
            "input_ids": torch.tensor([[1, 10, 2]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "input_features": torch.zeros(1, 2),
            "feature_attention_mask": torch.ones(1, 2, dtype=torch.long),
        }


def _process_mm_info(conversation, *, use_audio_in_video):
    assert use_audio_in_video is False
    kind = conversation[0]["content"][1]["type"]
    if kind == "video":
        return None, None, ["video-pixels"]
    if kind == "audio":
        return ["audio-features"], None, None
    raise AssertionError(kind)


def test_audio_omission_supplies_video_but_no_audio_path() -> None:
    request = build_modality_omission_request(
        "audio_omitted_model_input",
        source="clip.mp4",
        nframes=8,
        evidence_scope="local_only",
        target_runtime_verified=False,
    )
    processor = _Processor()
    prepared = prepare_modality_omission_inputs(
        request,
        prompt="Describe the video",
        processor=processor,
        process_mm_info=_process_mm_info,
        processor_revision="processor-rev",
        model_revision="model-rev",
        audio_token_index=10,
        video_token_index=20,
    )

    call = processor.calls[0]
    assert call["audio"] is None
    assert call["videos"] == ["video-pixels"]
    assert call["use_audio_in_video"] is False
    assert prepared.modality_presence == {"audio": False, "video": True}
    assert prepared.research_ready is False
    assert prepared.processor_path == "processor.video_without_audio"
    assert len(prepared.token_layout_fingerprint) == 64


def test_video_omission_supplies_audio_but_no_video_path() -> None:
    request = build_modality_omission_request(
        "video_omitted_model_input",
        source="clip-audio.wav",
        evidence_scope="target_runtime",
        target_runtime_verified=True,
    )
    processor = _Processor()
    prepared = prepare_modality_omission_inputs(
        request,
        prompt="Describe the sound",
        processor=processor,
        process_mm_info=_process_mm_info,
        processor_revision="processor-rev",
        model_revision="model-rev",
        audio_token_index=10,
        video_token_index=20,
    )

    call = processor.calls[0]
    assert call["audio"] == ["audio-features"]
    assert call["videos"] is None
    assert prepared.modality_presence == {"audio": True, "video": False}
    assert prepared.research_ready is True
    assert prepared.processor_path == "processor.audio_only"


def test_signal_controls_and_edge_knockout_cannot_enter_omission_api() -> None:
    for operation in [
        "audio_silence_control",
        "video_neutral_control",
        "direct_attention_edge_knockout",
    ]:
        try:
            build_modality_omission_request(operation, source="fixture.mp4")
            raise AssertionError(f"{operation} must remain outside the omission API")
        except ValueError as exc:
            assert "omission" in str(exc).lower()
