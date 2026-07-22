"""Treatment-neutral common pre/post classroom outcome records."""

from __future__ import annotations

from typing import Any, Mapping


OUTCOME_RESPONSE_SCHEMA = "common-outcome-response/1.0.0"
PRE_OUTCOME_ID = "critical-ai-reasoning-pre/0.1.0-pilot"
POST_OUTCOME_ID = "critical-ai-reasoning-post/0.1.0-pilot"

_RESPONSE_FIELDS = frozenset({"evidence", "alternative", "limit"})
_SUPPORTED_LANGUAGES = frozenset({"en", "ko"})


def build_outcome_bundle(
    *,
    session_pseudonym: str,
    language: str,
    pre_response: Mapping[str, str],
    post_response: Mapping[str, str],
) -> dict[str, Any]:
    """Build the identical condition-neutral pre/post record for either route."""

    for stage, response in (("pre", pre_response), ("post", post_response)):
        if set(response) != _RESPONSE_FIELDS:
            raise ValueError(
                f"{stage} response fields must be {sorted(_RESPONSE_FIELDS)}"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in response.values()
        ):
            raise ValueError(f"{stage} response fields must contain non-empty text")
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError("language must be 'en' or 'ko'")
    if not isinstance(session_pseudonym, str) or not session_pseudonym.strip():
        raise ValueError("session_pseudonym must be non-empty text")
    return {
        "schema_version": OUTCOME_RESPONSE_SCHEMA,
        "classification": "private formative learning response; not research data",
        "session_pseudonym": session_pseudonym.strip(),
        "language": language,
        "administration_order": [PRE_OUTCOME_ID, POST_OUTCOME_ID],
        "responses": {
            "pre": {"instrument_id": PRE_OUTCOME_ID, **dict(pre_response)},
            "post": {"instrument_id": POST_OUTCOME_ID, **dict(post_response)},
        },
        "missingness": {"pre": "observed", "post": "observed"},
        "automatic_student_data_egress": False,
        "research_ready": False,
    }


def validate_outcome_bundle(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable validation messages for one common outcome envelope."""

    issues: list[str] = []
    if value.get("schema_version") != OUTCOME_RESPONSE_SCHEMA:
        issues.append("unsupported common outcome schema")
    if value.get("administration_order") != [PRE_OUTCOME_ID, POST_OUTCOME_ID]:
        issues.append("common outcome administration order changed")
    responses = value.get("responses")
    if not isinstance(responses, Mapping):
        issues.append("common outcome responses must be a mapping")
    else:
        for stage, instrument_id in (
            ("pre", PRE_OUTCOME_ID),
            ("post", POST_OUTCOME_ID),
        ):
            response = responses.get(stage)
            if (
                not isinstance(response, Mapping)
                or response.get("instrument_id") != instrument_id
            ):
                issues.append(f"{stage} outcome instrument identifier changed")
            elif any(
                not str(response.get(key, "")).strip()
                for key in ("evidence", "alternative", "limit")
            ):
                issues.append(f"{stage} outcome response is incomplete")
    return tuple(issues)


__all__ = [
    "OUTCOME_RESPONSE_SCHEMA",
    "POST_OUTCOME_ID",
    "PRE_OUTCOME_ID",
    "build_outcome_bundle",
    "validate_outcome_bundle",
]
