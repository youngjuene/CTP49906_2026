from __future__ import annotations

from pathlib import Path

import pytest

from curriculum_common.pilot_profile import (
    final_release_issues,
    load_pilot_profile,
    require_final_release,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "study_materials" / "wp0" / "course_release_profile.json"
BASELINE = "495875bdb297b809c82dccc6e631616444f47229"


def test_candidate_profile_is_neutral_exact_and_fail_closed() -> None:
    profile = load_pilot_profile(PROFILE)

    assert profile.baseline_commit == BASELINE
    assert profile.protected_tag_ref == "refs/tags/teaching-pilot-candidate"
    assert profile.protected_tag_published is False
    assert profile.course_release_id == f"teaching-pilot-candidate/{BASELINE}"
    assert profile.candidate_observation_only is True
    assert profile.immutable is False
    assert profile.research_enabled is False
    assert profile.default_execution_mode == "saved_replay"
    assert profile.automatic_student_data_egress is False
    assert profile.student_media_ownership == "student_retained_private_media"
    assert profile.peer_exchange_required is False
    assert profile.peer_permission_required is True
    assert profile.private_equivalent_enabled is True
    assert profile.private_equivalent_no_penalty is True
    assert set(profile.private_evidence_sources) == {
        "synthetic_example",
        "instructor_example",
    }
    assert all(
        (
            profile.treatment_route_id,
            profile.comparison_route_id,
            profile.audience_route_id,
        )
    )


def test_candidate_profile_rejects_any_research_enablement(tmp_path: Path) -> None:
    text = PROFILE.read_text(encoding="utf-8").replace(
        '"research_enabled": false',
        '"research_enabled": true',
    )
    changed = tmp_path / "profile.json"
    changed.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Research"):
        load_pilot_profile(changed)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (
            '"automatic_student_data_egress": false',
            '"automatic_student_data_egress": "false"',
            "automatic student-data egress",
        ),
        (
            '"default_execution_mode": "saved_replay"',
            '"default_execution_mode": "live_model"',
            "saved replay",
        ),
        (
            '"student_media_ownership": "student_retained_private_media"',
            '"student_media_ownership": "instructor_collected"',
            "student-retained",
        ),
    ],
)
def test_candidate_profile_rejects_malformed_teaching_boundaries(
    tmp_path: Path,
    original: str,
    replacement: str,
    message: str,
) -> None:
    changed = tmp_path / "profile.json"
    changed.write_text(
        PROFILE.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_pilot_profile(changed)


def test_final_release_validation_explicitly_rejects_candidate_profile() -> None:
    profile = load_pilot_profile(PROFILE)

    issues = final_release_issues(profile)
    assert any("candidate" in issue for issue in issues)
    assert any("immutable" in issue for issue in issues)
    assert any("incomplete" in issue.lower() for issue in issues)
    assert any("protected pilot tag" in issue for issue in issues)
    with pytest.raises(ValueError, match="final release"):
        require_final_release(profile)
