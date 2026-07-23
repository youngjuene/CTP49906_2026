# G011 Completion Checklist

> Current state: **G011_WAITING_FOR_HUMAN_DECISIONS**. This checklist does not record or imply approval.

## Separate readiness gates

- **Classroom release:** Group A may support classroom release only after every Group A human decision is complete with a release-permitting outcome, every condition is satisfied, and the external classroom technical gate passes.
- **Research readiness:** Requires confirmed classroom readiness, every Group B human decision complete with a research-permitting outcome, every condition satisfied, and the external research gate passes. Research mode remains disabled until then.

## Group A — classroom decisions

- [ ] A-T01 — teaching-only/ethics determination
- [ ] A-T02 — privacy-security-storage-deletion-retention
- [ ] A-T03 — media rights and institutional model-license fit
- [ ] A-T04 — accessibility and assistive-technology review
- [ ] A-T05 — English/Korean localization review
- [ ] A-T06 — arts-pedagogy and instructor usability
- [ ] A-T07 — matched-arm production fidelity
- [ ] A-T08 — classroom release owner and incident response

## Group B — research decisions

- [ ] B-R01 — IRB/equivalent
- [ ] B-R02 — consent-assent-withdrawal
- [ ] B-R03 — instruments and translation/adaptation permissions
- [ ] B-R04 — allocation-estimands-sample-size-power
- [ ] B-R05 — rater training-reliability-adjudication
- [ ] B-R06 — audience recruitment/data handling
- [ ] B-R07 — missing-data/analysis
- [ ] B-R08 — publication-quotation-sharing-reuse permissions

## Exact cross-file completion rules

For each of the 16 decision IDs above, all of the following must be true:

1. The ID occurs exactly once in `governance_decision_register.json` and exactly once in `governance_evidence_index.json`.
2. The decision register status is no longer `UNRESOLVED`; the evidence-index status is no longer `MISSING`.
3. `accountable_approver`, `role_or_authority`, `scope`, `decision_date`, `outcome`, `conditions`, `evidence_path`, `evidence_sha256`, `expiry_date`, and `revisit_trigger` are populated identically in both JSON files.
4. `signature_or_record_reference` is populated in the evidence index and corresponds to the completed applicable template or authoritative institutional record.
5. `outcome` is exactly one of `APPROVED`, `REJECTED`, or `CONDITIONAL`; no template option may be treated as preselected.
6. The evidence file exists at `evidence_path`, is authoritative for the named approver and scope, and its calculated SHA-256 exactly equals `evidence_sha256`.
7. Any `CONDITIONAL` outcome has explicit conditions, evidence that every condition is satisfied, and a release-permitting determination by the accountable human authority.
8. `expiry_date` and `revisit_trigger` are explicitly recorded, including an explicit institutional value such as “not applicable” only where the accountable authority permits it; guidance fields alone never satisfy a decision fact.
9. Decision-specific required authority, required scope, and required evidence in the register are satisfied; `default_revisit_triggers` and `legacy_governance_ids` remain guidance only.

G011 may be marked complete only when:

- All 16 per-ID rules above pass.
- Both JSON roots are changed consistently from `G011_WAITING_FOR_HUMAN_DECISIONS` only by an authorized completion process.
- `g011_complete=true` is identical in both JSON files.
- `classroom_ready_claim=true` only after all eight Group A decisions permit release, all their conditions are satisfied, and the external classroom technical gate has passed.
- `research_ready_claim=true` and `research_mode=enabled` only after `classroom_ready_claim=true`, all eight Group B decisions permit research, all their conditions are satisfied, and the external research gate has passed.
- A `REJECTED` decision, an unsatisfied condition, missing/mismatched evidence, failed external gate, expired decision, or triggered revisit keeps the affected readiness claim false.
- Until every completion rule is met, `g011_complete=false`, `research_mode=disabled`, `classroom_ready_claim=false`, and `research_ready_claim=false`.
