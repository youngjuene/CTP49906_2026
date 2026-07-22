from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "495875bdb297b809c82dccc6e631616444f47229"
PROFILE_PATH = ROOT / "study_materials" / "wp0" / "course_release_profile.json"
TREATMENT_NOTEBOOK = (
    ROOT / "avllm_interpretability" / "CTP49906_avllm_molab.py"
)
COMPARISON_NOTEBOOK = (
    ROOT / "digital_storytelling" / "CTP49906_storytelling_molab.py"
)


def _student_markdown_literals(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "md"
            and isinstance(function.value, ast.Name)
            and function.value.id == "mo"
        ):
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            chunks.append(value.value)
    return "\n".join(chunks)


def test_candidate_profile_bootstraps_exact_baseline_and_disables_research() -> None:
    from curriculum_common.pilot_profile import load_pilot_profile

    profile = load_pilot_profile(PROFILE_PATH)

    assert profile.baseline_commit == BASELINE_COMMIT
    assert profile.course_release_id == f"teaching-pilot-candidate/{BASELINE_COMMIT}"
    assert profile.candidate_observation_only is True
    assert profile.immutable is False
    assert profile.research_enabled is False
    assert profile.default_execution_mode == "saved_replay"
    assert profile.automatic_student_data_egress is False
    assert profile.treatment_route_id
    assert profile.comparison_route_id
    assert profile.audience_route_id


def test_both_student_routes_consume_the_same_neutral_pilot_profile() -> None:
    treatment = TREATMENT_NOTEBOOK.read_text(encoding="utf-8")
    comparison = COMPARISON_NOTEBOOK.read_text(encoding="utf-8")

    for source in (treatment, comparison):
        assert "from curriculum_common.pilot_profile import" in source
        assert "load_pilot_profile" in source
        assert ".course_release_id" in source
        assert "research_enabled" in source
    assert 'COURSE_RELEASE_ID = "counterpoint-lens-classroom-2026-07-22"' not in treatment


def test_peer_exchange_requires_permission_but_private_equivalent_does_not() -> None:
    from curriculum_common.pilot_profile import (
        PEER_EXCHANGE_ROUTE,
        PRIVATE_EQUIVALENT_ROUTE,
        resolve_exchange_route,
    )

    with pytest.raises(PermissionError, match="permission"):
        resolve_exchange_route(
            PEER_EXCHANGE_ROUTE,
            permission_confirmed=False,
        )

    peer = resolve_exchange_route(
        PEER_EXCHANGE_ROUTE,
        permission_confirmed=True,
    )
    assert peer.audience_exchange_required is True
    assert peer.no_penalty is True
    assert peer.research_ready is False

    for evidence_source in ("synthetic_example", "instructor_example"):
        private = resolve_exchange_route(
            PRIVATE_EQUIVALENT_ROUTE,
            permission_confirmed=False,
            evidence_source=evidence_source,
        )
        assert private.audience_exchange_required is False
        assert private.no_penalty is True
        assert private.evidence_source == evidence_source
        assert private.research_ready is False
        assert private.checkpoint_id == peer.checkpoint_id


def test_notebooks_offer_optional_peer_and_equivalent_private_no_penalty_routes() -> None:
    for path in (TREATMENT_NOTEBOOK, COMPARISON_NOTEBOOK):
        source = path.read_text(encoding="utf-8")
        student_text = _student_markdown_literals(path).lower()
        assert "resolve_exchange_route" in source
        assert "peer_exchange" in source
        assert "private_equivalent" in source
        assert "optional" in student_text or "선택" in student_text
        assert "permission" in student_text or "권한" in student_text
        assert "private" in student_text or "비공개" in student_text
        assert "no penalty" in student_text or "불이익" in student_text
        assert "synthetic" in student_text or "합성" in student_text
        assert "instructor" in student_text or "교수" in student_text


def test_student_cells_remain_free_of_internal_repair_annotations() -> None:
    forbidden = (
        "candidate repair",
        "repair_required",
        "worker-1",
        "worker-2",
        "worker-3",
        "baseline commit",
        "internal planning",
    )
    for path in (TREATMENT_NOTEBOOK, COMPARISON_NOTEBOOK):
        student_text = _student_markdown_literals(path).lower()
        assert not any(term in student_text for term in forbidden)
