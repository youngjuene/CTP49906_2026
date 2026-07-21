"""Lane-1 tests for deterministic controlled-variant contracts."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stimulus_variants import (  # noqa: E402
    StimulusManifest,
    audio_silence_control,
    build_stimulus_variant,
    canonical_recipe_bytes,
    duration_match_samples,
    duration_matched_audio_swap,
    dump_stimulus_manifest,
    load_stimulus_manifest,
    neutral_video_control,
    normalize_recipe,
    offset_samples_zero_fill,
    stimulus_id_for,
)


SOURCE = "sha256:" + "1" * 64


def _variant(**changes):
    values = {
        "parent_content_sha256": SOURCE,
        "pair_id": "pair-a",
        "condition_code": "original-a",
        "technical_operation": "original_reference",
        "congruence_level": "congruent",
        "congruence_coding_provenance": {"coder_protocol": "fixture-v1"},
        "pairing_block": "block-a",
        "manipulation_check": {"status": "pass"},
        "parameters": {"equality_policy": "decoded_signal"},
        "donor_stimulus_id": None,
        "offset_ms": None,
        "duration_ms": 1000,
        "audio_facts": {"sample_rate_hz": 16000, "sample_count": 16000},
        "video_facts": {"frame_count": 25},
        "processor_path": "standard_av_processor",
        "token_layout_fingerprint": "pending:pre-model",
        "provenance": {"license_status": "synthetic_fixture"},
    }
    values.update(changes)
    return build_stimulus_variant(**values)


def test_recipe_key_order_and_negative_zero_are_canonical():
    first = {"b": [1, -0.0], "a": {"y": 2, "x": 1}}
    second = {"a": {"x": 1, "y": 2}, "b": [1, 0.0]}
    assert normalize_recipe(first) == normalize_recipe(second)
    assert canonical_recipe_bytes(first) == canonical_recipe_bytes(second)
    assert stimulus_id_for(SOURCE, first) == stimulus_id_for(SOURCE, second)
    with pytest.raises(ValueError, match="finite"):
        normalize_recipe({"bad": float("nan")})


def test_semantic_recipe_inputs_invalidate_stimulus_identity():
    base = _variant()
    for changed in (
        _variant(condition_code="original-b"),
        _variant(
            technical_operation="temporal_offset",
            condition_code="offset-a",
            offset_ms=40,
        ),
        _variant(
            technical_operation="audio_swap_duration_matched",
            condition_code="swap-a",
            donor_stimulus_id="donor-a",
        ),
        _variant(transform_schema_version="stimulus-transform/2.0.0"),
    ):
        assert changed.stimulus_id != base.stimulus_id


def test_duration_match_and_signed_offset_zero_fill_without_wrap():
    assert duration_match_samples((1, 2), 4) == (1, 2, 0, 0)
    assert duration_match_samples((1, 2, 3), 2) == (1, 2)
    assert offset_samples_zero_fill((1, 2, 3, 4), 2) == (0, 0, 1, 2)
    assert offset_samples_zero_fill((1, 2, 3, 4), -2) == (3, 4, 0, 0)


def test_swap_resamples_to_processor_contract_and_records_normalization():
    swapped, record = duration_matched_audio_swap((0, 1, 0), 8000, 8)
    assert len(swapped) == 8
    assert record["processor_sample_rate_hz"] == 16000
    assert record["input_sample_rate_hz"] == 8000
    assert record["target_sample_count"] == 8
    assert record["duration_match"] == "zero_pad"


def test_signal_controls_are_media_not_modality_omission():
    assert audio_silence_control(4) == (0, 0, 0, 0)
    assert neutral_video_control(3, "black") == ("black", "black", "black")
    silence = _variant(
        technical_operation="audio_silence_control",
        condition_code="silence-a",
    )
    assert silence.technical_operation == "audio_silence_control"
    assert "omitted" not in silence.processor_path
    with pytest.raises(ValueError, match="modality-omission"):
        _variant(
            technical_operation="audio_silence_control",
            condition_code="silence-b",
            processor_path="audio_omitted_model_input",
        )


def test_congruence_and_operation_are_independently_required():
    congruent = _variant(congruence_level="congruent")
    incongruent = _variant(congruence_level="incongruent")
    assert congruent.technical_operation == incongruent.technical_operation
    assert congruent.congruence_level != incongruent.congruence_level
    with pytest.raises(ValueError, match="provenance"):
        _variant(congruence_coding_provenance={})


def test_manifest_round_trip_is_deterministic_and_strict():
    manifest = StimulusManifest.build((_variant(),), release_status="blocked")
    payload = dump_stimulus_manifest(manifest)
    assert dump_stimulus_manifest(load_stimulus_manifest(payload)) == payload
    changed_order = json.loads(payload)
    assert load_stimulus_manifest(
        json.dumps(changed_order, indent=2, sort_keys=False).encode()
    ) == manifest
    changed_order["future_field"] = True
    with pytest.raises(ValueError, match="unknown"):
        load_stimulus_manifest(json.dumps(changed_order).encode())
    tampered = json.loads(payload)
    tampered["variants"][0]["duration_ms"] += 1
    with pytest.raises(ValueError, match="checksum"):
        load_stimulus_manifest(json.dumps(tampered).encode())


def test_top_level_condition_cannot_diverge_from_checksummed_recipe():
    variant = _variant()
    changed = variant.to_dict()
    changed["condition_code"] = "retargeted-condition"
    with pytest.raises(ValueError, match="condition_code"):
        type(variant).from_dict(changed)
