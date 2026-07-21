from datetime import datetime, timedelta, timezone
import copy
import unittest

from curriculum_common.portfolio_export import (
    ProjectionError,
    build_private_portfolio,
    forbidden_research_paths,
    project_research,
    serialize_private_portfolio,
)
from curriculum_common.session_records import (
    new_session,
    reduce_command,
    resolve_operating_mode,
)


NOW = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)
PERMITTED = {
    "condition_code",
    "metrics",
    "metric_versions",
    "result_digest",
    "stimulus_id",
    "tags",
}


def research_decision(process_permission=True):
    config = {
        "mode": "Research",
        "approved": True,
        "approved_protocol_version": "protocol/1",
        "approved_wording_reference": "wording/1",
        "independent_permission_scopes": {
            "process_data_analysis": process_permission,
            "classroom_or_audience_sharing": False,
            "quotation_or_reproduction": False,
            "future_reuse": False,
        },
        "permitted_fields": sorted(PERMITTED),
        "destination": "https://approved.invalid/ingest",
        "retention_period": "P90D",
        "withdrawal_path": "mailto:withdraw@example.invalid",
        "contact": "research@example.invalid",
        "expiry": "2027-01-01T00:00:00Z",
        "compatible_schema_version": "append-only-process/1.0.0",
        "compatible_course_release_id": "course/1",
    }
    return resolve_operating_mode(
        config,
        expected_schema_version="append-only-process/1.0.0",
        expected_course_release_id="course/1",
        approved_field_allowlist=PERMITTED,
        now=NOW,
    )


def populated_log():
    log = new_session("pseudonym-7", course_release_id="course/1", now=NOW)
    run = reduce_command(
        log,
        {
            "kind": "commit_run",
            "command_nonce": "run-1",
            "artifact_id": "private-artifact-1",
            "stimulus_id": "stimulus-1",
            "condition_code": "baseline",
            "model_id": "model-1",
            "model_revision": "immutable",
            "prompt": "Private prompt",
            "parameters": {"temperature": 0.0, "device_metadata": "GPU serial"},
            "prediction": "Private prediction",
            "initial_explanation": "Private explanation",
        },
        now=NOW + timedelta(seconds=1),
        id_factory=lambda: "00000000-0000-4000-8000-000000000001",
    )
    log = run.log
    log = reduce_command(
        log,
        {
            "kind": "attach_result",
            "command_nonce": "result-1",
            "run_id": run.record_ids[0],
            "result_digest": "digest-1",
            "metrics": {"entropy": 1.5},
        },
        now=NOW + timedelta(seconds=2),
        id_factory=lambda: "00000000-0000-4000-8000-000000000002",
    ).log
    return log


class PortfolioTests(unittest.TestCase):
    def test_private_portfolio_retains_private_history_without_research_claim(self):
        private = build_private_portfolio(
            populated_log(),
            artifact_versions=[
                {
                    "artifact_id": "private-artifact-1",
                    "filename": "student-cut.mp4",
                    "content_sha256": "private-hash",
                }
            ],
        )
        self.assertIn("not research data", private["classification"])
        self.assertIsNone(private["research_destination"])
        self.assertEqual(private["artifacts"][0]["content_sha256"], "private-hash")
        self.assertIn("Molab", private["boundary_disclosure"])

    def test_research_projection_is_pure_minimized_and_identity_mapped(self):
        private = build_private_portfolio(
            populated_log(),
            artifact_versions=[
                {
                    "artifact_id": "private-artifact-1",
                    "filename": "student-cut.mp4",
                    "content_sha256": "private-hash",
                    "device_metadata": {"gpu": "serial"},
                }
            ],
        )
        original = serialize_private_portfolio(private)
        original_object = copy.deepcopy(private)
        projection = project_research(
            private,
            decision=research_decision(),
            identity_map={
                "private-artifact-1": "study-artifact-42",
                "private-hash": "study-artifact-42",
            },
        )
        data = projection.to_dict()

        self.assertEqual(serialize_private_portfolio(private), original)
        self.assertEqual(private, original_object)
        self.assertEqual(forbidden_research_paths(data), ())
        self.assertNotIn(b"Private prompt", projection.to_bytes())
        self.assertNotIn(b"pseudonym-7", projection.to_bytes())
        self.assertEqual(data["records"][0]["study_artifact_id"], "study-artifact-42")
        self.assertEqual(data["artifacts"][0]["study_artifact_id"], "study-artifact-42")
        self.assertIn("study_record_id", data["records"][0])
        self.assertTrue(data["redaction_report"])

    def test_teaching_or_revoked_analysis_permission_cannot_project(self):
        private = build_private_portfolio(populated_log())
        with self.assertRaises(ProjectionError):
            project_research(private, decision=research_decision(False), identity_map={})


if __name__ == "__main__":
    unittest.main()
