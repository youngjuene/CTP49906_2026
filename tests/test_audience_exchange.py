from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audience.response_core import (
    AUDIENCE_CONTENT,
    CLASSROOM_MINIMUM_NOTICE,
    DISPLAY_PACKET_FIELDS,
    build_reading,
    packet_for_display,
    parse_json_object,
    prepare_reveal_bundle,
)
from curriculum_common.audience_packets import (
    AUDIENCE_SHARING_PERMISSION,
    PACKET_FORBIDDEN_FIELDS,
    AudiencePacket,
    AudienceReading,
    AudienceValidationError,
    RevealState,
    presentation_asset_checksum,
    validate_audience_exchange,
    validate_packet_mapping,
)


FIXTURES = Path(__file__).parent / "fixtures" / "audience_export"
CORE = Path(__file__).parents[1] / "audience" / "response_core.py"
SURFACE = Path(__file__).parents[1] / "audience" / "CTP49906_audience_response_molab.py"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class AudiencePacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet_mapping = load_fixture("packet.json")
        self.packet = AudiencePacket.from_mapping(self.packet_mapping)

    def test_valid_packet_is_allowlisted_and_checksum_bound(self) -> None:
        self.assertEqual(validate_packet_mapping(self.packet_mapping), ())
        self.assertEqual(
            presentation_asset_checksum(self.packet.presentation_asset),
            self.packet.presentation_checksum,
        )
        display = packet_for_display(self.packet)
        self.assertEqual(set(display), set(DISPLAY_PACKET_FIELDS))
        self.assertTrue(PACKET_FORBIDDEN_FIELDS.isdisjoint(display))
        self.assertNotIn("creator_intention", json.dumps(display))
        self.assertNotIn("model_output", json.dumps(display))

    def test_packet_rejects_nested_creator_or_model_data(self) -> None:
        unsafe = load_fixture("packet.json")
        unsafe["presentation_asset"]["creator_intention"] = "secret"
        unsafe["presentation_checksum"] = presentation_asset_checksum(
            unsafe["presentation_asset"]
        )
        issues = validate_packet_mapping(unsafe)
        self.assertIn("unsafe_asset_field", {issue.code for issue in issues})
        self.assertIn("nested_forbidden_field", {issue.code for issue in issues})
        with self.assertRaises(AudienceValidationError):
            AudiencePacket.from_mapping(unsafe)

    def test_packet_rejects_checksum_mismatch_and_hash_identity(self) -> None:
        invalid = load_fixture("packet.json")
        invalid["exchange_artifact_id"] = "a" * 64
        invalid["presentation_checksum"] = "b" * 64
        codes = {issue.code for issue in validate_packet_mapping(invalid)}
        self.assertIn("exchange_id", codes)
        self.assertIn("checksum_mismatch", codes)

    def test_reveal_is_monotonic(self) -> None:
        revealed = self.packet.reveal(protocol_deviation="scheduled classroom reveal")
        self.assertIs(revealed.reveal_state, RevealState.REVEALED)
        with self.assertRaisesRegex(ValueError, "cannot be restored"):
            revealed.transition_to(RevealState.BLINDED)


class AudienceReadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = AudiencePacket.from_mapping(load_fixture("packet.json"))
        self.reading_1 = AudienceReading.from_mapping(
            load_fixture("reading-1.json"), packet=self.packet
        )
        self.reading_2 = AudienceReading.from_mapping(
            load_fixture("reading-2.json"), packet=self.packet
        )

    def test_two_distinct_blinded_readings_complete_classroom_activity(self) -> None:
        report = validate_audience_exchange(
            self.packet, [self.reading_1, self.reading_2]
        )
        self.assertTrue(report.is_valid, report.issues)
        self.assertTrue(report.is_complete)
        self.assertEqual(
            report.valid_blinded_response_ids,
            (self.reading_1.response_id, self.reading_2.response_id),
        )

    def test_duplicate_respondent_session_is_rejected(self) -> None:
        duplicate = replace(
            self.reading_2,
            respondent_session_pseudonym=self.reading_1.respondent_session_pseudonym,
        )
        report = validate_audience_exchange(self.packet, [self.reading_1, duplicate])
        self.assertFalse(report.is_valid)
        self.assertFalse(report.is_complete)
        self.assertIn("duplicate_session", {issue.code for issue in report.issues})

    def test_response_after_reveal_requires_deviation_and_is_not_blinded(self) -> None:
        revealed_without_record = replace(
            self.packet,
            reveal_state=RevealState.REVEALED,
            protocol_deviation=None,
        )
        with self.assertRaises(AudienceValidationError):
            AudienceReading.from_mapping(
                self.reading_1.to_dict(), packet=revealed_without_record
            )

        revealed_with_record = self.packet.reveal(
            protocol_deviation="late response retained for audit"
        )
        retained = AudienceReading.from_mapping(
            self.reading_1.to_dict(), packet=revealed_with_record
        )
        report = validate_audience_exchange(revealed_with_record, [retained])
        self.assertFalse(report.is_complete)
        self.assertIn("response_after_reveal", {issue.code for issue in report.issues})

    def test_surface_builds_only_schema_fields(self) -> None:
        reading = build_reading(
            self.packet,
            load_fixture("response-form.json"),
            audience_pseudonym="audience-gamma",
            respondent_session_pseudonym="audience-session-gamma",
            event_index=2,
            elapsed_ms=62000,
            response_id="response-00000000-0000-4000-8000-000000000003",
            created_at_utc="2026-07-21T09:02:00Z",
        )
        self.assertEqual(reading.permission_scope, AUDIENCE_SHARING_PERMISSION)
        self.assertEqual(reading.exchange_artifact_id, self.packet.exchange_artifact_id)

    def test_json_import_rejects_duplicate_keys_and_non_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_json_object(b'{"response_id":"a","response_id":"b"}')
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_json_object(b"[]")

    def test_reveal_requires_complete_blinded_exchange(self) -> None:
        with self.assertRaisesRegex(ValueError, "two distinct valid blinded readings"):
            prepare_reveal_bundle(
                self.packet,
                [self.reading_1],
                creator_reading="The sound suggests anticipation.",
                creator_tags=["anticipation"],
                model_labels=["music", "motion"],
            )

        bundle = prepare_reveal_bundle(
            self.packet,
            [self.reading_1, self.reading_2],
            creator_reading="The sound suggests anticipation.",
            creator_tags=["anticipation"],
            model_labels=["music", "motion"],
        )
        rendered = bundle.to_dict()
        self.assertEqual(rendered["reveal_state"], "revealed")
        self.assertEqual(len(rendered["audience_readings"]), 2)
        self.assertIn("disagreement_matrix", rendered)
        self.assertIn("another reading", rendered["reflection_prompt"].lower())
        self.assertIn("classroom minimum", CLASSROOM_MINIMUM_NOTICE.lower())
        self.assertNotIn("research sample", CLASSROOM_MINIMUM_NOTICE.lower())

    def test_audience_content_has_english_korean_key_parity(self) -> None:
        self.assertEqual(set(AUDIENCE_CONTENT), {"en", "ko"})
        self.assertEqual(set(AUDIENCE_CONTENT["en"]), set(AUDIENCE_CONTENT["ko"]))
        self.assertTrue(all(AUDIENCE_CONTENT["en"].values()))
        self.assertTrue(all(AUDIENCE_CONTENT["ko"].values()))

    def test_surface_is_strict_marimo_shaped_and_gpu_free(self) -> None:
        source = SURFACE.read_text(encoding="utf-8")
        self.assertIn("import marimo", source)
        self.assertIn("app = marimo.App", source)
        self.assertIn("mo.ui.file", source)
        self.assertIn(".form(", source)
        self.assertIn("mo.download", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import transformers", source)
        self.assertNotIn("avllm_interpretability", source)

    def test_gpu_free_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "reading.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CORE),
                    "--packet",
                    str(FIXTURES / "packet.json"),
                    "--response",
                    str(FIXTURES / "response-form.json"),
                    "--output",
                    str(output),
                    "--audience-pseudonym",
                    "audience-cli",
                    "--session-pseudonym",
                    "audience-session-cli",
                    "--event-index",
                    "2",
                    "--elapsed-ms",
                    "62000",
                    "--response-id",
                    "response-00000000-0000-4000-8000-000000000004",
                    "--created-at-utc",
                    "2026-07-21T09:02:00Z",
                ],
                cwd=Path(__file__).parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reading = AudienceReading.from_mapping(
                json.loads(output.read_text(encoding="utf-8")), packet=self.packet
            )
            self.assertEqual(reading.audience_pseudonym, "audience-cli")


if __name__ == "__main__":
    unittest.main()
