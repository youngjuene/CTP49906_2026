"""Regression and hostile-fixture tests for the frozen WP-0 RQ3 audit."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rq3_estimability_audit as audit


MANIFEST_PATH = HERE / "rq3_design_manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def status_by_code(result: dict) -> dict[str, str]:
    return {check["code"]: check["status"] for check in result["checks"]}


class RQ3EstimabilityTests(unittest.TestCase):
    def test_frozen_manifest_passes_structure_but_blocks_release(self) -> None:
        result = audit.audit_manifest(load_manifest())
        self.assertEqual(result["structural_estimability_gate"], "PASS")
        self.assertEqual(result["research_release_gate"], "BLOCKED")
        self.assertEqual(result["mandatory_failure_codes"], [])

    def test_exact_schedule_size_rank_and_contrast_count(self) -> None:
        manifest = load_manifest()
        result = audit.audit_manifest(manifest)
        self.assertEqual(len(manifest["stimuli"]), 128)
        self.assertEqual(len(manifest["assignments"]), 256)
        self.assertEqual(len(manifest["declared_contrasts"]), 9)
        self.assertEqual(result["rank_diagnostic"]["columns"], 23)
        self.assertEqual(result["rank_diagnostic"]["exact_rank"], 23)
        self.assertEqual(result["rank_diagnostic"]["aliased_columns"], [])
        self.assertTrue(all(item["status"] == "PASS" for item in result["contrast_results"]))

    def test_corrected_schedule_is_exactly_balanced_by_order(self) -> None:
        rows = load_manifest()["assignments"]
        operation_order = Counter(
            (row["technical_operation"], row["presentation_order"]) for row in rows
        )
        congruence_order = Counter(
            (row["congruence_level"], row["presentation_order"]) for row in rows
        )
        self.assertEqual(set(operation_order.values()), {4})
        self.assertEqual(set(congruence_order.values()), {16})

    def test_all_nine_contrasts_overlap_every_declared_nuisance_level(self) -> None:
        result = audit.audit_manifest(load_manifest())
        self.assertEqual(len(result["contrast_results"]), 9)
        for contrast in result["contrast_results"]:
            self.assertEqual(contrast["status"], "PASS", contrast)
            for overlap in contrast["nuisance_overlap"].values():
                self.assertEqual(overlap["missing_from_lhs"], [])
                self.assertEqual(overlap["missing_from_rhs"], [])

    def test_negative_fixture_congruence_order_confound_fails(self) -> None:
        manifest = load_manifest()
        for row in manifest["assignments"]:
            row["presentation_order"] = 1 if row["congruence_level"] == "congruent" else 2
        result = audit.audit_manifest(manifest)
        statuses = status_by_code(result)
        self.assertEqual(result["structural_estimability_gate"], "FAIL")
        self.assertEqual(statuses["balance.order"], "FAIL")
        self.assertEqual(statuses["rank.main_effects"], "FAIL")

    def test_negative_fixture_operation_source_confound_fails(self) -> None:
        manifest = load_manifest()
        operations = [item["technical_operation"] for item in manifest["operation_taxonomy"]]
        source_for_operation = {
            operation: f"SRC-{index % 4 + 1:02d}"
            for index, operation in enumerate(operations)
        }
        for stimulus in manifest["stimuli"]:
            stimulus["source_artifact_id"] = source_for_operation[stimulus["technical_operation"]]
        for row in manifest["assignments"]:
            row["source_artifact_id"] = source_for_operation[row["technical_operation"]]
        result = audit.audit_manifest(manifest)
        statuses = status_by_code(result)
        self.assertEqual(result["structural_estimability_gate"], "FAIL")
        self.assertEqual(statuses["reuse.source"], "FAIL")
        self.assertEqual(statuses["rank.main_effects"], "FAIL")

    def test_negative_fixture_missing_operation_cell_fails(self) -> None:
        manifest = load_manifest()
        removed = "video_omitted_model_input"
        manifest["stimuli"] = [
            item for item in manifest["stimuli"] if item["technical_operation"] != removed
        ]
        manifest["assignments"] = [
            item for item in manifest["assignments"] if item["technical_operation"] != removed
        ]
        result = audit.audit_manifest(manifest)
        self.assertEqual(status_by_code(result)["coverage.operations"], "FAIL")
        self.assertEqual(result["structural_estimability_gate"], "FAIL")

    def test_negative_fixture_signal_control_as_omission_fails(self) -> None:
        manifest = load_manifest()
        silence = next(
            item
            for item in manifest["operation_taxonomy"]
            if item["technical_operation"] == "audio_silence_control"
        )
        silence["intervention_kind"] = "modality_omission"
        silence["audio_modality_supplied"] = False
        result = audit.audit_manifest(manifest)
        self.assertEqual(status_by_code(result)["semantics.operation_kinds"], "FAIL")
        self.assertEqual(result["structural_estimability_gate"], "FAIL")

    def test_duplicate_assignment_identity_and_slot_fail(self) -> None:
        manifest = load_manifest()
        duplicate = copy.deepcopy(manifest["assignments"][0])
        manifest["assignments"].append(duplicate)
        statuses = status_by_code(audit.audit_manifest(manifest))
        self.assertEqual(statuses["ids.assignment_id"], "FAIL")
        self.assertEqual(statuses["ids.assignment_slot"], "FAIL")

    def test_run_writes_blocked_release_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            result = audit.run(MANIFEST_PATH, report_path)
            text = report_path.read_text(encoding="utf-8")
        self.assertEqual(result["report"]["structural_estimability_gate"], "PASS")
        self.assertIn(
            "STRUCTURAL ESTIMABILITY PASS; RESEARCH RELEASE BLOCKED", text
        )
        self.assertIn("Exact rational rank: `23`", text)
        self.assertIn("licensed_course_stimuli", text)


if __name__ == "__main__":
    unittest.main()
