"""Standard-library structural estimability audit for the WP-0 RQ3 design.

This module intentionally audits only design structure.  It cannot establish
power, construct validity, manipulation success, licensing, or target-platform
modality feasibility.  Those are separate WP-0 gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "rq3_design_manifest.json"
DEFAULT_REPORT = HERE / "rq3_estimability_report.md"

REQUIRED_OPERATIONS = {
    "original_reference": ("baseline", True, True),
    "audio_swap_duration_matched": ("media_transform", True, True),
    "temporal_offset": ("media_transform", True, True),
    "audio_silence_control": ("stimulus_signal_control", True, True),
    "video_neutral_control": ("stimulus_signal_control", True, True),
    "audio_omitted_model_input": ("modality_omission", False, True),
    "video_omitted_model_input": ("modality_omission", True, False),
    "direct_attention_edge_knockout": ("model_intervention", True, True),
}

MAIN_EFFECT_FACTORS = (
    "congruence_level",
    "technical_operation",
    "source_artifact_id",
    "target_artifact_id",
    "block_id",
    "presentation_order",
)


def _failure(code: str, message: str, evidence: Any = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "status": "FAIL", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _pass(code: str, message: str, evidence: Any = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "status": "PASS", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _duplicates(values: Iterable[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _categorical_matrix(
    rows: list[dict[str, Any]], factors: Iterable[str]
) -> tuple[list[list[Fraction]], list[str]]:
    """Return treatment-coded exact matrix and human-readable column names."""

    columns: list[tuple[str, Any]] = []
    names = ["intercept"]
    for factor in factors:
        levels = sorted({row[factor] for row in rows}, key=lambda value: str(value))
        for level in levels[1:]:
            columns.append((factor, level))
            names.append(f"{factor}={level}")
    matrix = [
        [Fraction(1)]
        + [Fraction(int(row[factor] == level)) for factor, level in columns]
        for row in rows
    ]
    return matrix, names


def _exact_rank(matrix: list[list[Fraction]]) -> tuple[int, list[int]]:
    """Compute rank and pivot columns by exact Gaussian elimination."""

    if not matrix:
        return 0, []
    work = [row[:] for row in matrix]
    row_count = len(work)
    column_count = len(work[0])
    pivot_row = 0
    pivots: list[int] = []
    for column in range(column_count):
        candidate = next(
            (row for row in range(pivot_row, row_count) if work[row][column]),
            None,
        )
        if candidate is None:
            continue
        work[pivot_row], work[candidate] = work[candidate], work[pivot_row]
        divisor = work[pivot_row][column]
        work[pivot_row] = [value / divisor for value in work[pivot_row]]
        for row in range(pivot_row + 1, row_count):
            if not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [
                value - multiplier * pivot
                for value, pivot in zip(work[row], work[pivot_row])
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row, pivots


def _matches(row: dict[str, Any], criterion: dict[str, Any]) -> bool:
    return row.get(criterion["field"]) in criterion["levels"]


def _audit_contrast(
    rows: list[dict[str, Any]], contrast: dict[str, Any]
) -> dict[str, Any]:
    population = [
        row
        for row in rows
        if all(row.get(field) in levels for field, levels in contrast["population"].items())
    ]
    lhs = [row for row in population if _matches(row, contrast["lhs"])]
    rhs = [row for row in population if _matches(row, contrast["rhs"])]
    failures: list[str] = []
    overlap: dict[str, dict[str, Any]] = {}
    if not lhs or not rhs:
        failures.append("one or both contrast sides are empty")

    for factor in contrast["nuisance_factors"]:
        levels = sorted({row[factor] for row in population}, key=lambda value: str(value))
        missing_lhs = [
            level for level in levels if not any(row[factor] == level for row in lhs)
        ]
        missing_rhs = [
            level for level in levels if not any(row[factor] == level for row in rhs)
        ]
        overlap[factor] = {
            "levels": levels,
            "missing_from_lhs": missing_lhs,
            "missing_from_rhs": missing_rhs,
        }
        if missing_lhs or missing_rhs:
            failures.append(
                f"{factor} lacks side overlap (lhs missing {missing_lhs}; "
                f"rhs missing {missing_rhs})"
            )

    return {
        "contrast_id": contrast["contrast_id"],
        "status": "FAIL" if failures else "PASS",
        "lhs_n": len(lhs),
        "rhs_n": len(rhs),
        "defining_factors": contrast["defining_factors"],
        "nuisance_overlap": overlap,
        "failures": failures,
    }


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stimuli = manifest.get("stimuli", [])
    rows = manifest.get("assignments", [])
    sources = manifest.get("source_artifacts", [])
    targets = manifest.get("target_artifacts", [])

    required_top = {
        "schema_version",
        "design_status",
        "operation_taxonomy",
        "declared_contrasts",
        "source_artifacts",
        "target_artifacts",
        "stimuli",
        "assignments",
    }
    missing_top = sorted(required_top - set(manifest))
    checks.append(
        _failure("schema.top_level", "Required top-level fields are missing", missing_top)
        if missing_top
        else _pass("schema.top_level", "Required top-level fields are present")
    )

    id_specs = (
        ("source_artifact_id", sources),
        ("target_artifact_id", targets),
        ("stimulus_id", stimuli),
        ("assignment_id", rows),
    )
    for field, records in id_specs:
        missing = [index for index, record in enumerate(records) if not record.get(field)]
        duplicates = _duplicates(record.get(field) for record in records if record.get(field))
        if missing or duplicates:
            checks.append(
                _failure(
                    f"ids.{field}",
                    f"{field} values must be non-empty and unique",
                    {"missing_record_indexes": missing, "duplicates": duplicates},
                )
            )
        else:
            checks.append(
                _pass(f"ids.{field}", f"All {len(records)} {field} values are unique")
            )

    assignment_keys = [
        (row.get("block_id"), row.get("sequence_id"), row.get("presentation_order"))
        for row in rows
    ]
    duplicate_slots = _duplicates(assignment_keys)
    checks.append(
        _failure(
            "ids.assignment_slot",
            "A block/sequence/order slot may contain only one assignment",
            duplicate_slots,
        )
        if duplicate_slots
        else _pass("ids.assignment_slot", "All block/sequence/order slots are unique")
    )

    stimulus_by_id = {item.get("stimulus_id"): item for item in stimuli}
    required_columns = {
        "assignment_id",
        "stimulus_id",
        "pair_id",
        "block_id",
        "sequence_id",
        "order_id",
        "presentation_order",
        "congruence_level",
        "technical_operation",
        "source_artifact_id",
        "target_artifact_id",
        "donor_source_id",
        "offset_ms",
        "manipulation_check_status",
    }
    missing_columns: dict[str, list[str]] = {}
    mismatches: list[str] = []
    copied_fields = (
        "pair_id",
        "congruence_level",
        "technical_operation",
        "source_artifact_id",
        "target_artifact_id",
        "donor_source_id",
        "offset_ms",
        "manipulation_check_status",
    )
    for row in rows:
        absent = sorted(required_columns - set(row))
        if absent:
            missing_columns[str(row.get("assignment_id"))] = absent
        stimulus = stimulus_by_id.get(row.get("stimulus_id"))
        if stimulus is None:
            mismatches.append(f"{row.get('assignment_id')}: unknown stimulus_id")
            continue
        for field in copied_fields:
            if row.get(field) != stimulus.get(field):
                mismatches.append(f"{row.get('assignment_id')}: {field} differs from stimulus")
    if missing_columns or mismatches:
        checks.append(
            _failure(
                "schema.factor_columns",
                "Assignments must contain consistent flattened audit factor columns",
                {"missing": missing_columns, "mismatches": mismatches[:20]},
            )
        )
    else:
        checks.append(
            _pass(
                "schema.factor_columns",
                "Every assignment contains consistent factor, offset, and check columns",
            )
        )

    taxonomy = {
        item.get("technical_operation"): item for item in manifest.get("operation_taxonomy", [])
    }
    observed_operations = {item.get("technical_operation") for item in stimuli}
    if observed_operations != set(REQUIRED_OPERATIONS) or set(taxonomy) != set(REQUIRED_OPERATIONS):
        checks.append(
            _failure(
                "coverage.operations",
                "The exact eight required operation families must be represented",
                {
                    "missing_in_stimuli": sorted(set(REQUIRED_OPERATIONS) - observed_operations),
                    "extra_in_stimuli": sorted(observed_operations - set(REQUIRED_OPERATIONS)),
                    "missing_in_taxonomy": sorted(set(REQUIRED_OPERATIONS) - set(taxonomy)),
                },
            )
        )
    else:
        checks.append(_pass("coverage.operations", "All eight operation families are present"))

    semantic_errors: list[str] = []
    for operation, (kind, audio_present, video_present) in REQUIRED_OPERATIONS.items():
        entry = taxonomy.get(operation, {})
        if entry.get("intervention_kind") != kind:
            semantic_errors.append(f"{operation}: expected intervention_kind={kind}")
        if entry.get("audio_modality_supplied") is not audio_present:
            semantic_errors.append(f"{operation}: wrong audio modality-presence declaration")
        if entry.get("video_modality_supplied") is not video_present:
            semantic_errors.append(f"{operation}: wrong video modality-presence declaration")
    if semantic_errors:
        checks.append(
            _failure(
                "semantics.operation_kinds",
                "Signal controls, true omissions, and edge knockout must remain distinct",
                semantic_errors,
            )
        )
    else:
        checks.append(
            _pass(
                "semantics.operation_kinds",
                "Silence/neutral controls, true modality omission, and edge knockout are distinct",
            )
        )

    congruence_levels = {item.get("congruence_level") for item in stimuli}
    offset_levels = {
        item.get("offset_ms")
        for item in stimuli
        if item.get("technical_operation") == "temporal_offset"
    }
    factor_evidence = {
        "congruence_levels": sorted(congruence_levels),
        "source_levels": len({item.get("source_artifact_id") for item in stimuli}),
        "target_levels": len({item.get("target_artifact_id") for item in stimuli}),
        "block_levels": sorted({row.get("block_id") for row in rows}),
        "order_levels": sorted({row.get("presentation_order") for row in rows}),
        "temporal_offset_levels_ms": sorted(offset_levels),
    }
    factor_ok = (
        congruence_levels == {"congruent", "incongruent"}
        and factor_evidence["source_levels"] >= 2
        and factor_evidence["target_levels"] >= 2
        and len(factor_evidence["block_levels"]) >= 2
        and factor_evidence["order_levels"] == list(range(1, 9))
        and any(value < 0 for value in offset_levels)
        and any(value > 0 for value in offset_levels)
    )
    checks.append(
        _pass("coverage.factor_levels", "Required factor levels are covered", factor_evidence)
        if factor_ok
        else _failure(
            "coverage.factor_levels", "One or more required factor levels are absent", factor_evidence
        )
    )

    block_failures: list[str] = []
    for block in sorted({row.get("block_id") for row in rows}):
        block_rows = [row for row in rows if row.get("block_id") == block]
        if {row.get("technical_operation") for row in block_rows} != set(REQUIRED_OPERATIONS):
            block_failures.append(f"{block}: incomplete operation coverage")
        if {row.get("congruence_level") for row in block_rows} != {
            "congruent",
            "incongruent",
        }:
            block_failures.append(f"{block}: incomplete congruence coverage")
        if {row.get("presentation_order") for row in block_rows} != set(range(1, 9)):
            block_failures.append(f"{block}: incomplete order coverage")
    checks.append(
        _failure("variation.within_block", "Within-block variation is incomplete", block_failures)
        if block_failures
        else _pass("variation.within_block", "Each block varies operation, congruence, and order")
    )

    pair_failures: list[str] = []
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stimulus in stimuli:
        pair_groups[str(stimulus.get("pair_id"))].append(stimulus)
    for pair_id, items in sorted(pair_groups.items()):
        if {item.get("congruence_level") for item in items} != {"congruent", "incongruent"}:
            pair_failures.append(f"{pair_id}: congruence does not vary")
        for field in ("source_artifact_id", "target_artifact_id"):
            by_level: dict[Any, set[Any]] = defaultdict(set)
            for item in items:
                by_level[item.get(field)].add(item.get("congruence_level"))
            if any(levels != {"congruent", "incongruent"} for levels in by_level.values()):
                pair_failures.append(f"{pair_id}: {field} does not cross congruence")
    checks.append(
        _failure("variation.within_pair", "Matched pair-sets are not counterbalanced", pair_failures)
        if pair_failures
        else _pass(
            "variation.within_pair",
            "Every matched pair-set crosses congruence with source and target identity",
        )
    )

    reuse_failures: list[str] = []
    for source in sorted({item.get("source_artifact_id") for item in stimuli}):
        source_items = [item for item in stimuli if item.get("source_artifact_id") == source]
        if {item.get("technical_operation") for item in source_items} != set(REQUIRED_OPERATIONS):
            reuse_failures.append(f"{source}: not reused across all operation families")
        if {item.get("congruence_level") for item in source_items} != {
            "congruent",
            "incongruent",
        }:
            reuse_failures.append(f"{source}: not reused across congruence levels")
    checks.append(
        _failure("reuse.source", "Source/donor reuse is incomplete", reuse_failures)
        if reuse_failures
        else _pass(
            "reuse.source", "Every source is reused across all operations and congruence levels"
        )
    )

    operation_order_counts = Counter(
        (row.get("technical_operation"), row.get("presentation_order")) for row in rows
    )
    congruence_order_counts = Counter(
        (row.get("congruence_level"), row.get("presentation_order")) for row in rows
    )
    operation_values = set(operation_order_counts.values())
    congruence_values = set(congruence_order_counts.values())
    order_ok = (
        len(operation_order_counts) == len(REQUIRED_OPERATIONS) * 8
        and len(congruence_order_counts) == 2 * 8
        and operation_values == {4}
        and congruence_values == {16}
    )
    order_evidence = {
        "observed_operation_order_cells": len(operation_order_counts),
        "expected_operation_order_cells": len(REQUIRED_OPERATIONS) * 8,
        "operation_by_order_counts": sorted(operation_values),
        "observed_congruence_order_cells": len(congruence_order_counts),
        "expected_congruence_order_cells": 2 * 8,
        "congruence_by_order_counts": sorted(congruence_values),
    }
    checks.append(
        _pass("balance.order", "Operation and congruence are exactly balanced by order", order_evidence)
        if order_ok
        else _failure("balance.order", "Presentation order is not balanced", order_evidence)
    )

    rank_diagnostic: dict[str, Any]
    if rows and all(all(factor in row for factor in MAIN_EFFECT_FACTORS) for row in rows):
        matrix, column_names = _categorical_matrix(rows, MAIN_EFFECT_FACTORS)
        rank, pivot_indexes = _exact_rank(matrix)
        aliased = [
            name for index, name in enumerate(column_names) if index not in set(pivot_indexes)
        ]
        rank_diagnostic = {
            "rows": len(matrix),
            "columns": len(column_names),
            "exact_rank": rank,
            "aliased_columns": aliased,
            "factors": list(MAIN_EFFECT_FACTORS),
        }
        checks.append(
            _pass(
                "rank.main_effects",
                "Exact main-effect design matrix has full column rank",
                rank_diagnostic,
            )
            if rank == len(column_names)
            else _failure(
                "rank.main_effects",
                "Exact main-effect design matrix is rank deficient",
                rank_diagnostic,
            )
        )
    else:
        rank_diagnostic = {"error": "rows or required factors absent"}
        checks.append(
            _failure("rank.main_effects", "Cannot construct the main-effect design matrix")
        )

    contrast_results = [
        _audit_contrast(rows, contrast) for contrast in manifest.get("declared_contrasts", [])
    ]
    failed_contrasts = [
        result["contrast_id"] for result in contrast_results if result["status"] != "PASS"
    ]
    checks.append(
        _failure(
            "contrasts.single_nuisance_overlap",
            "At least one declared contrast lacks overlap on a nuisance factor",
            failed_contrasts,
        )
        if failed_contrasts
        else _pass(
            "contrasts.single_nuisance_overlap",
            "No declared contrast is perfectly determined by any declared single nuisance factor",
            [result["contrast_id"] for result in contrast_results],
        )
    )

    license_statuses = {
        record.get("license_status") for record in sources + targets + stimuli
    }
    honest_synthetic = (
        manifest.get("design_status", {}).get("synthetic_non_student") is True
        and manifest.get("design_status", {}).get("course_release_eligible") is False
        and license_statuses == {"synthetic_feasibility_only_not_course_licensed"}
    )
    checks.append(
        _pass(
            "honesty.synthetic_license",
            "Synthetic fixtures are explicitly non-student and not course-release licensed",
        )
        if honest_synthetic
        else _failure(
            "honesty.synthetic_license",
            "Synthetic/license status must not imply final course-release eligibility",
            sorted(str(status) for status in license_statuses),
        )
    )

    mandatory_failures = [check for check in checks if check["status"] == "FAIL"]
    structural_status = "PASS" if not mandatory_failures else "FAIL"
    unauditable = [
        {
            "gate": "statistical_power",
            "status": "NOT_AUDITED",
            "reason": "No approved sample size, variance/effect assumptions, or pilot outcomes exist.",
        },
        {
            "gate": "construct_validity",
            "status": "NOT_AUDITED",
            "reason": "Congruence labels are synthetic scheduling codes pending independent coding/pretest.",
        },
        {
            "gate": "manipulation_success",
            "status": "NOT_AUDITED",
            "reason": "All empirical manipulation checks are planned_not_run.",
        },
        {
            "gate": "target_platform_modality_feasibility",
            "status": "NOT_AUDITED",
            "reason": "This audit does not execute Molab processor/model audio- or video-omission paths.",
        },
        {
            "gate": "licensed_course_stimuli",
            "status": "BLOCKED",
            "reason": "Fixtures are synthetic feasibility records, not a licensed final course set.",
        },
    ]
    return {
        "schema_version": "wp0.rq3-estimability-audit.v1",
        "structural_estimability_gate": structural_status,
        "research_release_gate": "BLOCKED",
        "checks": checks,
        "rank_diagnostic": rank_diagnostic,
        "contrast_results": contrast_results,
        "unauditable_or_separate_gates": unauditable,
        "mandatory_failure_codes": [check["code"] for check in mandatory_failures],
    }


def render_markdown(
    manifest_path: Path, manifest_digest: str, manifest: dict[str, Any], audit: dict[str, Any]
) -> str:
    structural = audit["structural_estimability_gate"]
    gate_label = (
        "STRUCTURAL ESTIMABILITY PASS; RESEARCH RELEASE BLOCKED"
        if structural == "PASS"
        else "STRUCTURAL ESTIMABILITY FAIL; REDESIGN REQUIRED"
    )
    try:
        display_manifest_path = manifest_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_manifest_path = manifest_path
    lines = [
        "# WP-0 RQ3 estimability audit",
        "",
        f"**Gate verdict:** `{gate_label}`",
        f"**Manifest:** `{display_manifest_path.as_posix()}`",
        f"**Manifest SHA-256:** `{manifest_digest}`",
        f"**Design schema:** `{manifest.get('schema_version', 'missing')}`",
        "",
        "## Scope boundary",
        "",
        "This exact, standard-library audit evaluates structural main-effect rank,",
        "factor overlap, counterbalancing, and single-nuisance perfect confounding.",
        "A structural pass is **not** evidence of adequate power, construct validity,",
        "successful manipulation, licensing, or Molab/model modality-omission",
        "feasibility. Those remain separate mandatory WP-0 gates.",
        "",
        "Operation contrasts necessarily use `technical_operation` as their defining",
        "factor. The audit does not nonsensically treat that defining factor as a",
        "nuisance; instead it proves exact full rank for operation alongside the other",
        "main effects and requires every declared non-defining nuisance level on both",
        "sides of each contrast.",
        "",
        "## Structural checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in audit["checks"]:
        evidence = check.get("evidence", "")
        if isinstance(evidence, (dict, list)):
            evidence = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        lines.append(
            f"| `{check['code']}` | **{check['status']}** | "
            f"{check['message']} {str(evidence).replace('|', '&#124;')} |"
        )

    rank = audit["rank_diagnostic"]
    lines.extend(
        [
            "",
            "## Exact rank/alias diagnostic",
            "",
            f"- Rows: `{rank.get('rows', 'n/a')}`",
            f"- Treatment-coded columns: `{rank.get('columns', 'n/a')}`",
            f"- Exact rational rank: `{rank.get('exact_rank', 'n/a')}`",
            f"- Aliased columns: `{rank.get('aliased_columns', 'n/a')}`",
            f"- Factors: `{rank.get('factors', 'n/a')}`",
            "",
            "Full column rank supports estimability of the declared additive main-effect",
            "design columns. It does not prove estimability of arbitrary interactions or",
            "a future analysis model not represented here.",
            "",
            "## Declared-contrast overlap",
            "",
            "| Contrast | Status | LHS n | RHS n | Defining factor(s) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for result in audit["contrast_results"]:
        lines.append(
            f"| `{result['contrast_id']}` | **{result['status']}** | "
            f"{result['lhs_n']} | {result['rhs_n']} | "
            f"{', '.join(result['defining_factors'])} |"
        )

    lines.extend(
        [
            "",
            "## Separate or unauditable gates",
            "",
            "| Gate | Status | Why this audit cannot resolve it |",
            "|---|---:|---|",
        ]
    )
    for gate in audit["unauditable_or_separate_gates"]:
        lines.append(f"| `{gate['gate']}` | **{gate['status']}** | {gate['reason']} |")

    if structural == "PASS":
        redesign = (
            "No structural redesign is required for the synthetic counterbalancing "
            "schedule. Replace synthetic sources/targets with licensed, independently "
            "coded assets while preserving the frozen factor crossing and order schedule; "
            "then rerun this audit. Do not change a factor level in isolation."
        )
    else:
        redesign = (
            "Restore both congruence levels within every matched pair-set and block; "
            "reuse every source/target across contrast sides; assign every operation to "
            "every order in every block; and rerun until the exact main-effect matrix is "
            "full rank and every declared nuisance level occurs on both contrast sides."
        )
    lines.extend(
        [
            "",
            "## Minimal redesign / next evidence",
            "",
            redesign,
            "",
            "WP-0 must remain non-complete while any mandatory licensing, empirical",
            "manipulation, target-platform omission, resource, governance, or comparison-",
            "arm gate remains unresolved. This RQ3 lane does not waive those gates.",
            "",
        ]
    )
    return "\n".join(lines)


def run(manifest_path: Path, report_path: Path | None) -> dict[str, Any]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    audit = audit_manifest(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    if report_path is not None:
        report_path.write_text(
            render_markdown(manifest_path, digest, manifest, audit), encoding="utf-8"
        )
    return {"manifest_sha256": digest, "report": audit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.manifest, None if args.no_write_report else args.report)
    summary = {
        "manifest_sha256": result["manifest_sha256"],
        "structural_estimability_gate": result["report"]["structural_estimability_gate"],
        "research_release_gate": result["report"]["research_release_gate"],
        "failed_checks": result["report"]["mandatory_failure_codes"],
        "report_path": None if args.no_write_report else str(args.report),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["report"]["structural_estimability_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
