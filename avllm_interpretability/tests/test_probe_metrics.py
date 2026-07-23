"""CPU contracts for compact distributions, alignment, and cache identity."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probe_metrics import (  # noqa: E402
    MeasurementKind,
    align_trajectories,
    build_analysis_cache_identity,
    map_position_modalities,
    metric_version,
    summarize_logits,
    validate_metric_version,
)


def test_known_distribution_entropy_margin_topk_and_normalization() -> None:
    probabilities = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    summary = summarize_logits(
        probabilities.log(),
        top_k=3,
        target_token_ids=[1],
        measurement_kind=MeasurementKind.RAW_PROBE_SCORE_DISPERSION,
    )

    expected_entropy = -sum(float(p) * math.log(float(p)) for p in probabilities)
    assert math.isclose(summary.entropy_nats, expected_entropy, rel_tol=1e-6)
    assert math.isclose(
        summary.normalized_entropy,
        expected_entropy / math.log(3),
        rel_tol=1e-6,
    )
    assert math.isclose(summary.log_probability_margin, math.log(0.5 / 0.3), rel_tol=1e-6)
    assert [item.token_id for item in summary.top_tokens] == [0, 1, 2]
    assert summary.targets[0].rank == 2
    assert summary.targets[0].found is True
    assert summary.measurement_kind == MeasurementKind.RAW_PROBE_SCORE_DISPERSION.value
    assert "not calibrated" in summary.caveat.lower()


def test_constant_shift_extremes_ties_and_missing_targets_are_stable() -> None:
    base = torch.tensor([10000.0, 10000.0, -10000.0], dtype=torch.float16)
    shifted = base.float() - 1234.5
    left = summarize_logits(base, top_k=3, target_token_ids=[0, 9])
    right = summarize_logits(shifted, top_k=3, target_token_ids=[0, 9])

    assert math.isfinite(left.entropy_nats)
    assert left.log_probability_margin == 0.0
    assert [item.token_id for item in left.top_tokens[:2]] == [0, 1]
    assert math.isclose(left.entropy_nats, right.entropy_nats, rel_tol=1e-6)
    assert left.targets[0].rank == 1
    assert left.targets[1].found is False
    assert left.targets[1].rank is None
    assert left.to_dict().keys().isdisjoint({"logits", "probabilities"})


def test_nonfinite_logits_and_unsupported_measurement_claims_fail_loudly() -> None:
    try:
        summarize_logits(torch.tensor([0.0, float("nan")]))
        raise AssertionError("non-finite logits must be rejected")
    except ValueError as exc:
        assert "finite" in str(exc).lower()

    version = metric_version(
        MeasurementKind.RAW_PROBE_SCORE_DISPERSION,
        target_token_set_version="course-targets/1.0.0",
    )
    validate_metric_version(version)
    invalid = copy.deepcopy(version)
    invalid["claim_scope"] = "calibration"
    try:
        validate_metric_version(invalid)
        raise AssertionError("calibration claims need a labeled evaluation protocol")
    except ValueError as exc:
        assert "calibration" in str(exc).lower()


def test_modality_mapping_marks_overlap_and_unmapped_positions_unknown() -> None:
    mapped = map_position_modalities(
        7,
        audio_positions=[1, 2],
        video_positions=[3, 4],
        answer_positions=[5, 6],
    )
    assert mapped == ["unknown", "audio", "audio", "video", "video", "answer", "answer"]

    ambiguous = map_position_modalities(
        3,
        audio_positions=[1],
        video_positions=[1],
    )
    assert ambiguous[1] == "unknown"


def test_trajectory_alignment_requires_matching_layout_for_positions() -> None:
    rows = [
        {"layer": 2, "position": 5, "modality": "audio", "value": 0.2},
        {"layer": 0, "position": 5, "modality": "audio", "value": 0.1},
    ]
    exact = align_trajectories(rows, list(reversed(rows)), "layout-a", "layout-a")
    assert exact.allowed is True
    assert exact.mode == "position"
    assert [item[0]["layer"] for item in exact.pairs] == [0, 2]

    blocked = align_trajectories(rows, rows, "layout-a", "layout-b")
    assert blocked.allowed is False
    assert blocked.mode == "blocked"
    assert "fingerprint" in blocked.warning.lower()

    aggregate = align_trajectories(
        rows,
        rows,
        "layout-a",
        "layout-b",
        requested_mode="aggregate",
    )
    assert aggregate.allowed is True
    assert aggregate.mode == "aggregate"
    assert "position-wise" in aggregate.warning.lower()


def test_cache_identity_uses_semantics_but_ignores_display_labels() -> None:
    base = {
        "content_sha256": "a" * 64,
        "normalized_recipe": {"operation": "original_reference", "parameters": {}},
        "processor_revision": "processor-rev",
        "model_revision": "model-rev",
        "prompt": "Describe the sound",
        "frame_settings": {"nframes": 8, "fps": None},
        "analysis_parameters": {"layers": [0, 4], "top_k": 5},
        "display_labels": {"title": "First label"},
    }
    original = build_analysis_cache_identity(**base)

    renamed = dict(base)
    renamed["display_labels"] = {"title": "A different classroom label"}
    assert build_analysis_cache_identity(**renamed).key == original.key

    for field, replacement in [
        ("content_sha256", "b" * 64),
        ("processor_revision", "processor-rev-2"),
        ("model_revision", "model-rev-2"),
        ("prompt", "Describe the image"),
        ("frame_settings", {"nframes": 16, "fps": None}),
        ("analysis_parameters", {"layers": [0, 8], "top_k": 5}),
        ("normalized_recipe", {"operation": "temporal_offset", "parameters": {"offset_ms": 250}}),
    ]:
        changed = dict(base)
        changed[field] = replacement
        assert build_analysis_cache_identity(**changed).key != original.key, field
