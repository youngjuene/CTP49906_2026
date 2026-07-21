from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import unittest

from curriculum_common.audience_packets import AudiencePacket
from curriculum_common.export_validation import (
    export_json_text,
    import_json_text,
    neutralize_spreadsheet_formula,
    safe_csv_text,
    safe_report_text,
    validate_audience_packet_export,
    validate_private_portfolio,
    validate_projection,
    validate_research_projection,
)
from curriculum_common.portfolio_export import (
    build_private_portfolio,
    project_research,
    serialize_private_portfolio,
)
from curriculum_common.session_records import (
    new_session,
    reduce_command,
    resolve_operating_mode,
)
from tests.test_audience_exchange import load_fixture


NOW = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
RUN_ID = "11111111-1111-4111-8111-111111111111"
ARTIFACT_ID = "artifact-v1"
CONTENT_SHA256 = "a" * 64
STUDY_ARTIFACT_ID = "study-artifact-001"


def build_projection_pair():
    log = new_session(
        "session-alpha",
        course_release_id="release-fixture",
        now=NOW,
    )
    log = reduce_command(
        log,
        {
            "kind": "commit_run",
            "command_nonce": "nonce-run",
            "run_id": RUN_ID,
            "artifact_id": ARTIFACT_ID,
            "stimulus_id": "stimulus-v1",
            "condition_code": "neutral",
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
            "prompt": "neutral prompt",
            "parameters": {"seed": 7},
            "prediction": "The streams may reinforce one another.",
            "initial_explanation": "The timing appears aligned.",
        },
        now=NOW,
    ).log
    log = reduce_command(
        log,
        {
            "kind": "attach_result",
            "command_nonce": "nonce-result",
            "run_id": RUN_ID,
            "result_digest": "fixture-result-digest",
            "metrics": {},
        },
        now=NOW,
    ).log
    log = reduce_command(
        log,
        {
            "kind": "commit_reflection",
            "command_nonce": "nonce-reflection",
            "run_id": RUN_ID,
            "reflection": "The result supports one reading but not a causal claim.",
        },
        now=NOW,
    ).log
    private = build_private_portfolio(
        log,
        artifact_versions=[
            {
                "artifact_id": ARTIFACT_ID,
                "content_sha256": CONTENT_SHA256,
                "version_label": "V1",
            }
        ],
    )
    config = {
        "mode": "Research",
        "approved": True,
        "approved_protocol_version": "protocol-fixture-1",
        "approved_wording_reference": "wording-fixture-1",
        "independent_permission_scopes": {
            "process_data_analysis": True,
            "classroom_or_audience_sharing": False,
            "quotation_or_reproduction": False,
            "future_reuse": False,
        },
        "permitted_fields": ["version_label"],
        "destination": "mock://approved-instructor-destination",
        "retention_period": "30 days",
        "withdrawal_path": "mailto:withdraw@example.invalid",
        "contact": "mailto:pi@example.invalid",
        "expiry": "2027-07-21T00:00:00Z",
        "compatible_schema_version": "append-only-process/1.0.0",
        "compatible_course_release_id": "release-fixture",
    }
    decision = resolve_operating_mode(
        config,
        expected_schema_version="append-only-process/1.0.0",
        expected_course_release_id="release-fixture",
        approved_field_allowlist=["version_label"],
        now=NOW,
    )
    private_before = serialize_private_portfolio(private)
    research = project_research(
        private,
        decision=decision,
        identity_map={
            ARTIFACT_ID: STUDY_ARTIFACT_ID,
            CONTENT_SHA256: STUDY_ARTIFACT_ID,
        },
    )
    return private, private_before, research


class ExportValidationTests(unittest.TestCase):
    def test_lane3_private_and_research_outputs_validate_without_mutation(self) -> None:
        private, private_before, research = build_projection_pair()
        private_report = validate_private_portfolio(private)
        research_report = validate_research_projection(research)
        self.assertTrue(private_report.is_valid, private_report.issues)
        self.assertTrue(research_report.is_valid, research_report.issues)
        self.assertEqual(serialize_private_portfolio(private), private_before)
        self.assertNotIn(CONTENT_SHA256, research.to_bytes().decode("utf-8"))
        self.assertIn(STUDY_ARTIFACT_ID, research.to_bytes().decode("utf-8"))

    def test_validator_accepts_mapping_or_projection_object(self) -> None:
        private, _, research = build_projection_pair()
        self.assertTrue(validate_projection(private, projection_kind="private").is_valid)
        self.assertTrue(validate_projection(research, projection_kind="research").is_valid)

    def test_private_export_detects_orphan_duplicate_and_missing_prerun_fields(self) -> None:
        private, _, _ = build_projection_pair()
        invalid = deepcopy(private)
        invalid["validation"]["valid"] = True
        invalid["records"][0]["initial_explanation"] = ""
        orphan = deepcopy(invalid["records"][-1])
        orphan["record_id"] = "orphan-reflection"
        orphan["reflection_id"] = "orphan-reflection"
        orphan["run_id"] = "missing-run"
        orphan["command_nonce"] = "nonce-orphan"
        orphan["event_index"] = len(invalid["records"])
        invalid["records"].append(orphan)
        duplicate = deepcopy(invalid["records"][0])
        duplicate["event_index"] = len(invalid["records"])
        duplicate["command_nonce"] = "nonce-duplicate"
        invalid["records"].append(duplicate)
        report = validate_private_portfolio(invalid)
        codes = {issue.code for issue in report.issues}
        self.assertFalse(report.is_valid)
        self.assertIn("missing_prerun_field", codes)
        self.assertIn("orphan_run_link", codes)
        self.assertIn("duplicate_record_id", codes)

    def test_research_export_rejects_forbidden_nested_fields(self) -> None:
        _, _, research = build_projection_pair()
        invalid = research.to_dict()
        invalid["records"][0]["device_metadata"] = {"gpu": "should-not-export"}
        invalid["artifacts"][0]["content_sha256"] = CONTENT_SHA256
        report = validate_research_projection(invalid)
        self.assertFalse(report.is_valid)
        forbidden_paths = {
            issue.path
            for issue in report.issues
            if issue.code == "forbidden_research_field"
        }
        self.assertIn("records[0].device_metadata", forbidden_paths)
        self.assertIn("artifacts[0].content_sha256", forbidden_paths)

    def test_audience_packet_uses_same_export_validation_surface(self) -> None:
        packet = AudiencePacket.from_mapping(load_fixture("packet.json"))
        report = validate_audience_packet_export(packet)
        self.assertTrue(report.is_valid, report.issues)
        self.assertEqual(report.projection_kind, "audience")

    def test_formula_and_markup_are_neutralized_without_content_loss(self) -> None:
        self.assertEqual(neutralize_spreadsheet_formula("=2+2"), "'=2+2")
        self.assertEqual(neutralize_spreadsheet_formula("  @SUM(A1:A2)"), "'  @SUM(A1:A2)")
        csv_text = safe_csv_text([["interpretation", "=HYPERLINK(\"bad\")"]])
        self.assertIn("'=HYPERLINK", csv_text)
        self.assertEqual(
            safe_report_text("<script>alert('x')</script>"),
            "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;",
        )

    def test_canonical_export_round_trip_preserves_supported_fields(self) -> None:
        private, _, _ = build_projection_pair()
        text = export_json_text(private)
        restored = import_json_text(text)
        self.assertEqual(restored, private)
        self.assertEqual(export_json_text(restored), text)
        with self.assertRaises(ValueError):
            import_json_text(json.dumps(["not", "an", "object"]))


if __name__ == "__main__":
    unittest.main()
