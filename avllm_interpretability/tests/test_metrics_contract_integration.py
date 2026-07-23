"""Lane-2 integration: omission identity, layout limits, and semantic caches."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modality_omission import (  # noqa: E402
    build_modality_omission_request,
    prepare_modality_omission_inputs,
)
from src.probe_metrics import (  # noqa: E402
    align_trajectories,
    build_analysis_cache_identity,
)


class _Processor:
    def apply_chat_template(self, conversation, **kwargs):
        return "prompt-template"

    def __call__(self, **kwargs):
        if kwargs["videos"]:
            return {
                "input_ids": torch.tensor([[1, 20, 2]]),
                "attention_mask": torch.ones(1, 3),
                "pixel_values_videos": torch.zeros(1, 3, 2),
                "video_grid_thw": torch.ones(1, 3),
            }
        return {
            "input_ids": torch.tensor([[1, 10, 10, 2]]),
            "attention_mask": torch.ones(1, 4),
            "input_features": torch.zeros(1, 4, 2),
            "feature_attention_mask": torch.ones(1, 4),
        }


def _mm(conversation, *, use_audio_in_video):
    assert use_audio_in_video is False
    kind = conversation[0]["content"][1]["type"]
    return (["audio"], None, None) if kind == "audio" else (None, None, ["video"])


def _prepare(operation, source):
    return prepare_modality_omission_inputs(
        build_modality_omission_request(operation, source=source),
        prompt="Describe the available modality",
        processor=_Processor(),
        process_mm_info=_mm,
        processor_revision="processor-rev",
        model_revision="model-rev",
        audio_token_index=10,
        video_token_index=20,
    )


def test_true_omission_changes_layout_and_downgrades_comparison() -> None:
    audio_omitted = _prepare("audio_omitted_model_input", "video.mp4")
    video_omitted = _prepare("video_omitted_model_input", "audio.wav")

    assert audio_omitted.modality_presence != video_omitted.modality_presence
    assert audio_omitted.token_layout_fingerprint != video_omitted.token_layout_fingerprint
    rows = [{"layer": 0, "position": 1, "modality": "unknown", "value": 0.5}]
    exact = align_trajectories(
        rows,
        rows,
        audio_omitted.token_layout_fingerprint,
        video_omitted.token_layout_fingerprint,
    )
    aggregate = align_trajectories(
        rows,
        rows,
        audio_omitted.token_layout_fingerprint,
        video_omitted.token_layout_fingerprint,
        requested_mode="aggregate",
    )
    assert exact.allowed is False
    assert aggregate.allowed is True
    assert aggregate.warning

    shared = {
        "content_sha256": "c" * 64,
        "normalized_recipe": {"operation": "processor_input"},
        "processor_revision": "processor-rev",
        "model_revision": "model-rev",
        "prompt": "Describe the available modality",
        "frame_settings": {"nframes": 8},
    }
    audio_key = build_analysis_cache_identity(
        **shared,
        analysis_parameters={
            "technical_operation": "audio_omitted_model_input",
            "token_layout_fingerprint": audio_omitted.token_layout_fingerprint,
        },
    )
    video_key = build_analysis_cache_identity(
        **shared,
        analysis_parameters={
            "technical_operation": "video_omitted_model_input",
            "token_layout_fingerprint": video_omitted.token_layout_fingerprint,
        },
    )
    assert audio_key.key != video_key.key
