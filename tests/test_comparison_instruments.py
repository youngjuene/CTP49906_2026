from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from digital_storytelling.workflow import (
    ARTIFACT_RUBRIC_ID,
    FLEXIBILITY_RUBRIC_ID,
    POST_OUTCOME_ID,
    PRE_OUTCOME_ID,
)

INSTRUMENTS = ROOT / "study_materials" / "instruments"


def _load(name: str):
    return json.loads((INSTRUMENTS / name).read_text(encoding="utf-8"))


def test_instrument_catalog_records_all_honest_metadata() -> None:
    catalog = _load("instrument_catalog.json")
    assert catalog["condition_independent_scoring"] is True
    assert catalog["student_assignment_fields_permitted"] is False
    assert catalog["status"] == "candidate_not_validated_not_piloted_not_research_approved"
    instruments = {item["instrument_id"]: item for item in catalog["instruments"]}
    assert set(instruments) == {
        PRE_OUTCOME_ID,
        POST_OUTCOME_ID,
        ARTIFACT_RUBRIC_ID,
        FLEXIBILITY_RUBRIC_ID,
    }
    required = {
        "provenance",
        "adaptation_permission",
        "administration",
        "scoring",
        "missingness",
        "rater_training",
        "reliability",
        "adjudication",
        "translation_status",
        "pilot_status",
        "research_use",
    }
    for item in instruments.values():
        assert required <= set(item)
        assert item["pilot_status"] == "not_piloted"
        assert item["research_use"] == "not_approved"
    assert set(catalog["human_gates"].values()) == {"unresolved"}


def test_english_korean_instrument_content_has_exact_id_and_key_parity() -> None:
    english = _load("critical_ai_reasoning.en.json")
    korean = _load("critical_ai_reasoning.ko.json")
    assert english["administration_order"] == korean["administration_order"] == [
        PRE_OUTCOME_ID,
        POST_OUTCOME_ID,
    ]
    assert set(english["content"]) == set(korean["content"])
    assert all(value.strip() for value in english["content"].values())
    assert all(value.strip() for value in korean["content"].values())
    assert english["language"] == "en"
    assert korean["language"] == "ko"
    assert "human" in english["translation_status"]
    assert "human" in korean["translation_status"]


def test_rubrics_are_condition_independent_and_keep_missingness_explicit() -> None:
    rubrics = _load("artifact_rubrics.json")
    assert rubrics["condition_independent_scoring"] is True
    assert rubrics["allocation_metadata_visible_to_rater"] is False
    assert {item["rubric_id"] for item in rubrics["rubrics"]} == {
        ARTIFACT_RUBRIC_ID,
        FLEXIBILITY_RUBRIC_ID,
    }
    assert rubrics["missingness"]["empty_score_allowed"] is False
    assert rubrics["missingness"]["reason_required_when_not_observed"] is True
    assert rubrics["reliability_and_adjudication"]["status"].startswith("unresolved")
    serialized = json.dumps(rubrics, ensure_ascii=False, sort_keys=True).lower()
    assert "condition_code" not in serialized
    assert "condition_assignment" not in serialized


def test_fidelity_form_is_instructor_only_append_only_and_unresolved() -> None:
    fidelity = _load("matched_route_fidelity_form.json")
    assert fidelity["audience"] == "instructor_or_authorized_observer_only"
    assert fidelity["student_portfolio_member"] is False
    assert fidelity["research_ready"] is False
    assert fidelity["deviation_contract"]["append_only"] is True
    assert fidelity["deviation_contract"]["silent_repair_permitted"] is False
    assert fidelity["deviation_contract"]["criticality_threshold"] is None
    assert fidelity["two_audience_readings_semantics"].startswith(
        "minimum classroom activity"
    )
