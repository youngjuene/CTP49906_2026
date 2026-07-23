from datetime import datetime, timedelta, timezone
import unittest

from curriculum_common.session_records import (
    NonceConflictError,
    OperatingMode,
    PermissionScope,
    ResultConflictError,
    load_jsonl,
    new_session,
    reduce_command,
    resolve_operating_mode,
    validate_process_log,
)


NOW = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)


def valid_research_config(**overrides):
    value = {
        "mode": "Research",
        "approved": True,
        "approved_protocol_version": "protocol/1",
        "approved_wording_reference": "wording/1",
        "independent_permission_scopes": {
            "process_data_analysis": True,
            "classroom_or_audience_sharing": False,
            "quotation_or_reproduction": False,
            "future_reuse": False,
        },
        "permitted_fields": [
            "condition_code",
            "metrics",
            "metric_versions",
            "result_digest",
            "stimulus_id",
        ],
        "destination": "https://approved.invalid/ingest",
        "retention_period": "P90D",
        "withdrawal_path": "mailto:withdraw@example.invalid",
        "contact": "research@example.invalid",
        "expiry": "2027-01-01T00:00:00Z",
        "compatible_schema_version": "append-only-process/1.0.0",
        "compatible_course_release_id": "course/1",
    }
    value.update(overrides)
    return value


def run_command(nonce="run-nonce"):
    return {
        "kind": "commit_run",
        "command_nonce": nonce,
        "artifact_id": "private-artifact-1",
        "stimulus_id": "stimulus-1",
        "condition_code": "baseline",
        "model_id": "model-1",
        "model_revision": "immutable-revision",
        "prompt": "Describe the audiovisual relationship.",
        "parameters": {"temperature": 0.0},
        "prediction": "The sound will dominate.",
        "initial_explanation": "The onset is abrupt.",
    }


class ModeTests(unittest.TestCase):
    def test_invalid_and_over_permissive_configs_fail_closed(self):
        allowlist = {"metrics", "condition_code"}
        missing = resolve_operating_mode(
            None,
            expected_schema_version="append-only-process/1.0.0",
            expected_course_release_id="course/1",
            approved_field_allowlist=allowlist,
            now=NOW,
        )
        self.assertEqual(missing.mode, OperatingMode.TEACHING)

        config = valid_research_config(permitted_fields=["metrics", "legal_name"])
        excess = resolve_operating_mode(
            config,
            expected_schema_version="append-only-process/1.0.0",
            expected_course_release_id="course/1",
            approved_field_allowlist=allowlist,
            now=NOW,
        )
        self.assertEqual(excess.mode, OperatingMode.TEACHING)
        self.assertIn("over-permissive", excess.reason)

    def test_valid_config_preserves_independent_permission_scopes(self):
        config = valid_research_config()
        decision = resolve_operating_mode(
            config,
            expected_schema_version="append-only-process/1.0.0",
            expected_course_release_id="course/1",
            approved_field_allowlist=config["permitted_fields"],
            now=NOW,
        )
        self.assertEqual(decision.mode, OperatingMode.RESEARCH)
        self.assertTrue(decision.permits(PermissionScope.PROCESS_DATA_ANALYSIS))
        self.assertFalse(decision.permits(PermissionScope.FUTURE_REUSE))
        session = new_session("student-code", decision=decision, now=NOW)
        self.assertEqual(session.protocol_version, "protocol/1")


class ReducerTests(unittest.TestCase):
    def setUp(self):
        self.ids = iter(
            [
                "00000000-0000-4000-8000-000000000001",
                "00000000-0000-4000-8000-000000000002",
                "00000000-0000-4000-8000-000000000003",
                "00000000-0000-4000-8000-000000000004",
            ]
        )
        self.log = new_session("student-code", course_release_id="course/1", now=NOW)

    def apply(self, command, seconds=1):
        result = reduce_command(
            self.log,
            command,
            now=NOW + timedelta(seconds=seconds),
            id_factory=lambda: next(self.ids),
        )
        self.log = result.log
        return result

    def test_exact_nonce_replay_is_idempotent_and_conflict_is_rejected(self):
        first = self.apply(run_command())
        original_jsonl = self.log.to_jsonl()
        replay = reduce_command(self.log, run_command(), now=NOW + timedelta(seconds=10))
        self.assertTrue(replay.replayed)
        self.assertIs(replay.log, self.log)
        self.assertEqual(replay.log.to_jsonl(), original_jsonl)

        conflict = run_command()
        conflict["prompt"] = "Different payload"
        with self.assertRaises(NonceConflictError):
            reduce_command(self.log, conflict, now=NOW + timedelta(seconds=11))
        self.assertEqual(first.record_ids[0], self.log.records[0].record_id)

    def test_result_attachment_is_exactly_once_even_with_a_new_nonce(self):
        run_id = self.apply(run_command()).record_ids[0]
        result_command = {
            "kind": "attach_result",
            "command_nonce": "result-1",
            "run_id": run_id,
            "result_digest": "sha256:result",
            "metrics": {"entropy": 1.25},
        }
        self.apply(result_command, seconds=2)
        count = len(self.log.records)
        duplicate = dict(result_command, command_nonce="result-retry")
        replay = self.apply(duplicate, seconds=3)
        self.assertTrue(replay.replayed)
        self.assertEqual(len(self.log.records), count)

        different = dict(result_command, command_nonce="result-conflict", result_digest="other")
        with self.assertRaises(ResultConflictError):
            self.apply(different, seconds=4)

    def test_reflection_revision_and_withdrawal_never_rewrite_prior_bytes(self):
        run_id = self.apply(run_command()).record_ids[0]
        reflection = self.apply(
            {
                "kind": "commit_reflection",
                "command_nonce": "reflect-1",
                "run_id": run_id,
                "reflection": "My prediction missed the visual transition.",
            },
            seconds=2,
        )
        reflection_id = reflection.record_ids[0]
        prior_bytes = tuple(record.serialized for record in self.log.records)
        self.apply(
            {
                "kind": "revise_reflection",
                "command_nonce": "revise-1",
                "parent_reflection_id": reflection_id,
                "reflection": "The audio onset mattered more than I predicted.",
            },
            seconds=3,
        )
        self.apply(
            {
                "kind": "withdraw",
                "command_nonce": "withdraw-1",
                "target_record_id": reflection_id,
                "reason": "student requested withdrawal",
            },
            seconds=4,
        )
        self.assertEqual(tuple(record.serialized for record in self.log.records[:2]), prior_bytes)
        self.assertEqual(self.log.records[-1].record_type, "withdrawal")
        self.assertEqual(validate_process_log(self.log), ())

    def test_jsonl_reload_restores_nonce_replay_guarantee(self):
        self.apply(run_command())
        restored = load_jsonl(self.log.to_jsonl())
        replay = reduce_command(restored, run_command(), now=NOW + timedelta(seconds=20))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.log.to_jsonl(), self.log.to_jsonl())


if __name__ == "__main__":
    unittest.main()
