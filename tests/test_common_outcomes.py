from __future__ import annotations

import json

from curriculum_common.outcome_records import (
    OUTCOME_RESPONSE_SCHEMA,
    POST_OUTCOME_ID,
    PRE_OUTCOME_ID,
    build_outcome_bundle,
    validate_outcome_bundle,
)
from digital_storytelling import workflow


def _response(prefix: str) -> dict[str, str]:
    return {
        "evidence": f"{prefix} evidence",
        "alternative": f"{prefix} alternative",
        "limit": f"{prefix} limit",
    }


def test_common_outcome_contract_is_neutral_and_reexported_by_comparison() -> None:
    bundle = build_outcome_bundle(
        session_pseudonym="shared-fixture",
        language="en",
        pre_response=_response("pre"),
        post_response=_response("post"),
    )

    assert bundle["schema_version"] == OUTCOME_RESPONSE_SCHEMA
    assert bundle["administration_order"] == [PRE_OUTCOME_ID, POST_OUTCOME_ID]
    assert validate_outcome_bundle(bundle) == ()
    assert workflow.OUTCOME_RESPONSE_SCHEMA == OUTCOME_RESPONSE_SCHEMA
    assert workflow.PRE_OUTCOME_ID == PRE_OUTCOME_ID
    assert workflow.POST_OUTCOME_ID == POST_OUTCOME_ID
    assert workflow.build_outcome_bundle is build_outcome_bundle
    assert "condition" not in json.dumps(bundle, ensure_ascii=False).lower()


def test_common_outcome_contract_rejects_incomplete_or_changed_envelopes() -> None:
    bundle = build_outcome_bundle(
        session_pseudonym="shared-fixture",
        language="ko",
        pre_response=_response("pre"),
        post_response=_response("post"),
    )
    bundle["administration_order"] = [POST_OUTCOME_ID, PRE_OUTCOME_ID]
    bundle["responses"]["post"]["limit"] = ""

    issues = validate_outcome_bundle(bundle)
    assert "common outcome administration order changed" in issues
    assert "post outcome response is incomplete" in issues
