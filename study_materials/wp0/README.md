# WP-0 evidence package

**Package status:** `BLOCKED`
**Evidence cut:** 2026-07-21
**Scope:** WP-0 protocol, governance, comparison-design, and technical-feasibility
gates only.

This directory freezes the interfaces that must exist before the treatment UI is
implemented and records what is and is not yet proven. It does **not** implement
the shared modules, comparison notebook, audience application, instruments, or
treatment notebook.

## Canonical inputs

| Input | SHA-256 at evidence cut |
|---|---|
| `.omx/plans/prd-avllm-arts-integrated-classroom.md` | `d2e65fb1b21178ba6f6b85880aa642241f8bcdf8668fabca7e9ad6dd53735a39` |
| `.omx/plans/test-spec-avllm-arts-integrated-classroom.md` | `e058ed7fa2294c83dbf0e2aec4aacafb5ab5b171ddbbba4b901a95b65ecc3f43` |
| `.omx/plans/ralplan-handoff-avllm-arts-integrated-classroom.json` | `786dcfa01a7b62252bc13c111ebe6de421393a4fb9b06bd82897e7ebdbfdd350` |

The re-attestation handoff records sequential Architect then Critic approval and
no plan revision. These planning approvals do not turn unresolved human or
target-platform gates into technical passes.

## Artifact index

| Artifact | Frozen decision or evidence |
|---|---|
| `classroom_information_architecture.md` | Exact four-section/eight-stage route, disclosure levels, and immutable UI IDs |
| `path_contract.json` | Exact shared, comparison, audience, and study-material paths plus atomic rename policy |
| `shared_contracts.json` | Machine-readable schema, identity, mode, projection, and audience boundaries |
| `shared_contracts.md` | Human-readable interpretation of the shared contract |
| `matched_comparison_contract.md` | Mirrored production matrix and treatment-contamination boundary |
| `human_governance_register.md` | Decisions that remain with named accountable humans |
| `molab_feasibility_report.md` | Local technical probes, target-Molab limits, and gate conclusions |
| `molab_feasibility_evidence.json` | Structured commands, observations, and PASS/FAIL/UNVERIFIED results |
| `stimulus_release_requirements.md` | License, provenance, manipulation, and derived-variant release requirements |
| `course_release_profile.json` | Immutable-profile schema populated only with grounded candidate facts |
| `rq3_design_manifest.json` | Synthetic 128-stimulus, 256-assignment design with corrected balanced schedules and nine declared contrasts |
| `rq3_estimability_audit.py` | Dependency-free exact-rank, balance, overlap, identity, and semantic audit |
| `rq3_estimability_report.md` | Generated estimability report: structural PASS separated from release BLOCKED |
| `test_rq3_estimability.py` | Positive and confounded negative regression fixtures for the audit |
| `WP0_GATE_REPORT.md` | Requirement traceability, mandatory gate ledger, outcome, and stop decision |

## Gate semantics

- `PASS`: the exact gate was exercised with adequate evidence in the named
  environment and no mandatory residual condition remains.
- `FAIL`: evidence demonstrates that the required behavior does not hold.
- `UNVERIFIED`: evidence is absent, is only static, is from the wrong environment,
  or depends on an unresolved human decision.
- `BLOCKED`: the aggregate work package cannot complete because at least one
  mandatory gate is `FAIL` or `UNVERIFIED`.

Local `tx01` results are deliberately not promoted to target-Molab results.
Processor-only evidence is not promoted to a full classroom rehearsal, and an
automated artifact cannot stand in for institutional, methods, accessibility,
licensing, or audience approval.

## Stop boundary

WP-0 completes only when **every** mandatory gate in `WP0_GATE_REPORT.md` is
`PASS`. Until then:

1. G001/WP-0 must remain non-complete with blockers;
2. WP-1 must not start;
3. `avllm_interpretability/CTP49906_avllm_molab.py` must not be changed for this
   work package; and
4. Research mode and research-ready claims must remain disabled.
