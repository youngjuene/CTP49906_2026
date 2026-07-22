"""Neutral, fail-closed teaching-pilot configuration shared by both course arms."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


PILOT_PROFILE_SCHEMA = "wp0.course-release-profile/1.0.0"
PEER_EXCHANGE_ROUTE = "peer_exchange"
PRIVATE_EQUIVALENT_ROUTE = "private_equivalent"
EXCHANGE_CHECKPOINT_ID = "compare_readings"

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_PROFILE_PATH = (
    _ROOT / "study_materials" / "wp0" / "course_release_profile.json"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PROTECTED_TAG_PATTERN = re.compile(r"^refs/tags/teaching-pilot-[a-z0-9._/-]+$")
_PRIVATE_EVIDENCE_SOURCES = frozenset(
    {"synthetic_example", "instructor_example"}
)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


@dataclass(frozen=True)
class TeachingPilotProfile:
    """Validated candidate identity and classroom-only policy."""

    schema_version: str
    profile_status: str
    baseline_commit: str
    protected_tag_ref: str
    protected_tag_published: bool
    course_release_id: str
    candidate_observation_only: bool
    immutable: bool
    research_enabled: bool
    default_execution_mode: str
    automatic_student_data_egress: bool
    student_media_ownership: str
    treatment_route_id: str
    comparison_route_id: str
    audience_route_id: str
    peer_exchange_required: bool
    peer_permission_required: bool
    private_equivalent_enabled: bool
    private_equivalent_no_penalty: bool
    private_evidence_sources: tuple[str, ...]

    @property
    def teaching_mode_reason(self) -> str:
        return (
            "Teaching is the default; this candidate profile disables Research "
            "and has no approved research destination"
        )


@dataclass(frozen=True)
class ExchangeRouteDecision:
    """One permission-aware route through the shared comparison checkpoint."""

    route_id: str
    checkpoint_id: str
    audience_exchange_required: bool
    permission_confirmed: bool
    evidence_source: str
    no_penalty: bool
    research_ready: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "checkpoint_id": self.checkpoint_id,
            "audience_exchange_required": self.audience_exchange_required,
            "permission_confirmed": self.permission_confirmed,
            "evidence_source": self.evidence_source,
            "no_penalty": self.no_penalty,
            "research_ready": self.research_ready,
        }

    def to_private_checkpoint_record(self) -> dict[str, object]:
        """Return neutral metadata safe for a student-retained local bundle."""

        return {
            "checkpoint_id": self.checkpoint_id,
            "evidence_source": self.evidence_source,
            "no_penalty": self.no_penalty,
            "research_ready": self.research_ready,
        }


def load_pilot_profile(
    path: str | Path = DEFAULT_PILOT_PROFILE_PATH,
) -> TeachingPilotProfile:
    """Load the checked-in candidate profile without upgrading release claims."""

    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load teaching-pilot profile: {exc}") from exc
    root = _mapping(raw, "profile")
    if root.get("schema_version") != PILOT_PROFILE_SCHEMA:
        raise ValueError("unsupported teaching-pilot profile schema")

    repository = _mapping(root.get("repository"), "repository")
    candidate = _mapping(root.get("pilot_candidate"), "pilot_candidate")
    shared = _mapping(root.get("shared_course_contract"), "shared_course_contract")
    human = _mapping(root.get("human_approved_versions"), "human_approved_versions")
    peer = _mapping(candidate.get("peer_exchange"), "pilot_candidate.peer_exchange")
    private = _mapping(
        candidate.get("private_equivalent"),
        "pilot_candidate.private_equivalent",
    )

    baseline_commit = _required_text(
        candidate.get("baseline_commit"), "pilot_candidate.baseline_commit"
    )
    if not _COMMIT_PATTERN.fullmatch(baseline_commit):
        raise ValueError("pilot_candidate.baseline_commit must be a full git commit")
    if repository.get("observed_head") != baseline_commit:
        raise ValueError("candidate baseline must match repository.observed_head")

    protected_tag_ref = _required_text(
        candidate.get("protected_tag_ref"), "pilot_candidate.protected_tag_ref"
    )
    if not _PROTECTED_TAG_PATTERN.fullmatch(protected_tag_ref):
        raise ValueError("pilot_candidate.protected_tag_ref must name a protected pilot tag")
    if candidate.get("protected_tag_published") is not False:
        raise ValueError("candidate profile must not claim that the protected tag is published")

    course_release_id = _required_text(
        candidate.get("course_release_id"), "pilot_candidate.course_release_id"
    )
    if baseline_commit not in course_release_id:
        raise ValueError("candidate course_release_id must retain the baseline identity")
    if root.get("candidate_observation_only") is not True:
        raise ValueError("the teaching-pilot profile must remain candidate-only")
    if root.get("immutable") is not False:
        raise ValueError("this profile is not a final immutable course release")
    if root.get("immutable_course_release_id") is not None:
        raise ValueError("candidate profile must not claim an immutable release ID")
    if candidate.get("research_enabled") is not False:
        raise ValueError("Research must remain disabled for the teaching pilot")
    if human.get("research_configuration") is not None:
        raise ValueError("candidate profile cannot include a research configuration")

    sources = private.get("evidence_sources")
    if not isinstance(sources, list) or set(sources) != _PRIVATE_EVIDENCE_SOURCES:
        raise ValueError(
            "private-equivalent evidence sources must be synthetic and instructor examples"
        )
    if peer.get("required") is not False or peer.get("permission_required") is not True:
        raise ValueError("peer exchange must be optional and permission-aware")
    if private.get("enabled") is not True or private.get("no_penalty") is not True:
        raise ValueError("the equivalent private route must be enabled without penalty")
    if candidate.get("automatic_student_data_egress") is not False:
        raise ValueError("automatic student-data egress must be exactly false")
    if candidate.get("default_execution_mode") != "saved_replay":
        raise ValueError("the teaching pilot must default to saved replay")
    if candidate.get("student_media_ownership") != "student_retained_private_media":
        raise ValueError("student-retained private media ownership is required")

    return TeachingPilotProfile(
        schema_version=PILOT_PROFILE_SCHEMA,
        profile_status=_required_text(root.get("profile_status"), "profile_status"),
        baseline_commit=baseline_commit,
        protected_tag_ref=protected_tag_ref,
        protected_tag_published=False,
        course_release_id=course_release_id,
        candidate_observation_only=True,
        immutable=False,
        research_enabled=False,
        default_execution_mode=_required_text(
            candidate.get("default_execution_mode"),
            "pilot_candidate.default_execution_mode",
        ),
        automatic_student_data_egress=False,
        student_media_ownership=_required_text(
            candidate.get("student_media_ownership"),
            "pilot_candidate.student_media_ownership",
        ),
        treatment_route_id=_required_text(
            shared.get("treatment_route_id"),
            "shared_course_contract.treatment_route_id",
        ),
        comparison_route_id=_required_text(
            shared.get("comparison_route_id"),
            "shared_course_contract.comparison_route_id",
        ),
        audience_route_id=_required_text(
            shared.get("audience_route_id"),
            "shared_course_contract.audience_route_id",
        ),
        peer_exchange_required=False,
        peer_permission_required=True,
        private_equivalent_enabled=True,
        private_equivalent_no_penalty=True,
        private_evidence_sources=tuple(sources),
    )


def final_release_issues(profile: TeachingPilotProfile) -> tuple[str, ...]:
    """Explain why a teaching candidate cannot be treated as a final release."""

    issues: list[str] = []
    if profile.candidate_observation_only:
        issues.append("candidate observation profile is not a final release")
    if not profile.immutable:
        issues.append("final release identity is not immutable")
    if profile.profile_status != "PASS_FINAL_RELEASE":
        issues.append(f"release profile remains incomplete: {profile.profile_status}")
    if not profile.protected_tag_published:
        issues.append("advertised protected pilot tag is not published")
    return tuple(issues)


def require_final_release(profile: TeachingPilotProfile) -> None:
    """Reject candidate/pilot state at any final-release validation boundary."""

    issues = final_release_issues(profile)
    if issues:
        raise ValueError("final release rejected: " + "; ".join(issues))


def resolve_exchange_route(
    route_id: str,
    *,
    permission_confirmed: bool,
    evidence_source: str | None = None,
) -> ExchangeRouteDecision:
    """Resolve peer or private evidence without treating either as research data."""

    if route_id == PEER_EXCHANGE_ROUTE:
        if not permission_confirmed:
            raise PermissionError("classroom peer exchange requires sharing permission")
        return ExchangeRouteDecision(
            route_id=route_id,
            checkpoint_id=EXCHANGE_CHECKPOINT_ID,
            audience_exchange_required=True,
            permission_confirmed=True,
            evidence_source="independent_peer_readings",
            no_penalty=True,
        )
    if route_id == PRIVATE_EQUIVALENT_ROUTE:
        source = evidence_source or "instructor_example"
        if source not in _PRIVATE_EVIDENCE_SOURCES:
            raise ValueError(
                "private-equivalent evidence_source must be synthetic_example or "
                "instructor_example"
            )
        return ExchangeRouteDecision(
            route_id=route_id,
            checkpoint_id=EXCHANGE_CHECKPOINT_ID,
            audience_exchange_required=False,
            permission_confirmed=False,
            evidence_source=source,
            no_penalty=True,
        )
    raise ValueError(f"unsupported learning-exchange route: {route_id!r}")
