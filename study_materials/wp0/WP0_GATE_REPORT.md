# WP-0 gate report

```text
WP-0 outcome: BLOCKED
G001 status: NON-COMPLETE
WP-1 authorization: PROHIBITED
Research mode: DISABLED
```

**Evidence date:** 2026-07-21
**Protected notebook baseline/current SHA-256:**
`4302b5933fed8d7e54b39c2e8e2989d7e4a786630391c636a082bbd8a873c064`

## Frozen-requirement trace

| # | Requirement | WP-0 result | Evidence and residual condition |
|---:|---|---:|---|
| 1 | Exact four-section/eight-stage IA and progressive disclosure | **PASS** | `classroom_information_architecture.md` freezes hierarchy, IDs, obligations, required/choice/advanced routes, Checkpoints, and student-surface exclusions. |
| 2 | Exact shared/comparison/audience/study paths and dependency direction | **PASS** | `path_contract.json` freezes all four paths, the protected treatment path, atomic rename rule, and one-way neutral dependencies. |
| 3 | Shared schemas, stable IDs, projections, append-only process, Teaching/Research modes | **PASS — technical contract only** | `shared_contracts.json`/`.md` reproduce the required field families and invariants. Approved protocol and immutable release remain `null`; Research stays disabled. |
| 4 | Molab RAM/VRAM/cache/replay/PyAV/true-omission feasibility | **UNVERIFIED** | Local tx01 cache/replay and true-omission probes pass, but target runtime, full route, credible 24 GiB behavior, restart/offline replay, codec/browser/reingest, and egress evidence are absent. |
| 5 | RQ3 design not perfectly confounded | **PASS — structural only; release BLOCKED** | 128 synthetic stimuli, 256 assignments, exact main-effect rank 23/23, complete operation/congruence order balance, and nuisance overlap for all nine contrasts. Licensing, coding, manipulation, power, and target omission remain separate blockers. |
| 6 | Licensed stimuli and immutable course-release profile | **FAIL / INCOMPLETE** | Rights manifest is absent, Qwen Research License fit is unapproved, `REPO_REF=main` is mutable, dependencies/runtime/assets/protocol/thresholds are incomplete, and `immutable=false`. |
| 7 | Matched comparison frozen before treatment-interface freeze | **PASS — structural; concrete freeze BLOCKED** | Structural matrix and contamination boundary are frozen. Time/help/source/editor/export/instrument/fidelity values, tolerances, allocation/estimands, and approvals are unresolved. |
| 8 | Honest register of human governance decisions | **PASS — register only** | Twelve accountable decisions are recorded without fabricated approval. All are `UNRESOLVED` except the current rights/release gate, which is `FAIL`. |

## Mandatory gate ledger

| Mandatory gate | Status |
|---|---:|
| IA freeze | **PASS** |
| path freeze | **PASS** |
| shared technical contracts | **PASS** |
| target Molab host RAM | **UNVERIFIED** |
| target/full-route VRAM under credible 24 GiB constraint | **UNVERIFIED** |
| target cache/restart persistence | **UNVERIFIED** |
| target no-network replay after restart | **UNVERIFIED** |
| target PyAV encode/mux/decode/browser/reingest | **UNVERIFIED** |
| local true audio/video omission | **PASS — local only** |
| target true omission and full model route | **UNVERIFIED** |
| RQ3 rank/alias/overlap audit | **PASS — structural 23/23** |
| final licensed stimulus manifest | **FAIL** |
| immutable course release profile | **FAIL** |
| concrete matched-arm fidelity/interface freeze | **UNVERIFIED / BLOCKED** |
| IRB or formal teaching-only determination | **UNVERIFIED** |
| instruments/methods/accessibility/audience/arts sign-offs | **UNVERIFIED** |
| protected notebook content integrity during WP-0 | **PASS** |

Any `FAIL` or `UNVERIFIED` mandatory gate keeps the aggregate work package
`BLOCKED`; a local or static pass cannot be promoted to a target/human pass.

## RQ3 audit evidence

The generated `rq3_estimability_report.md` records:

- assignments: `256` across two blocks and 32 sequences;
- synthetic stimuli: `128` across eight distinct operation families;
- exact additive main-effect columns/rank: `23/23`, no aliases;
- operation × order: all 64 cells, exactly 4 assignments per cell;
- congruence × order: all 16 cells, exactly 16 assignments per cell;
- declared contrasts: `9/9 PASS`, with every declared nuisance level on both sides;
- semantic separation: signal controls, true modality omissions, and direct edge
  knockout remain distinct; and
- verdict: `STRUCTURAL ESTIMABILITY PASS; RESEARCH RELEASE BLOCKED`.

Negative fixtures demonstrate failure for congruence/order confounding,
operation/source confounding, a missing operation cell, duplicate IDs/slots, and
mislabeling a signal control as true omission.

## Fresh verification

```text
PYTHONDONTWRITEBYTECODE=1 python study_materials/wp0/rq3_estimability_audit.py
  structural_estimability_gate: PASS
  research_release_gate: BLOCKED
  manifest SHA-256: 530eb6ebbf9ce2ad2633b96ce84c35d2c086b1e6b771c8c9e44cba99afc5149c

PYTHONDONTWRITEBYTECODE=1 python -m unittest -v study_materials/wp0/test_rq3_estimability.py
  10 tests: PASS

PYTHONDONTWRITEBYTECODE=1 avllm_interpretability/.venv/bin/pytest -q -p no:cacheprovider \
  study_materials/wp0/test_rq3_estimability.py
  10 passed

PYTHONDONTWRITEBYTECODE=1 avllm_interpretability/.venv/bin/pytest -q -p no:cacheprovider \
  avllm_interpretability/tests/test_precompute.py \
  avllm_interpretability/tests/test_playground_clips.py
  20 passed

PYTHONDONTWRITEBYTECODE=1 avllm_interpretability/.venv/bin/pytest -q -p no:cacheprovider \
  avllm_interpretability/tests/test_precompute.py::test_committed_classroom_replay_pack_is_complete
  1 passed

JSON parse: 5/5 WP-0 JSON files PASS
AST parse without bytecode: 2/2 WP-0 Python files PASS
Protected notebook SHA-256: exact frozen baseline PASS
```

## Blockers that must clear before WP-1 authorization

1. Target Molab rehearsal with resource, cache, restart/offline replay, codec,
   omission, temporary-file/deletion, and egress evidence.
2. Real/enforced 24 GiB evidence or an explicitly narrower approved claim.
3. Complete media rights/provenance manifest and institutional Qwen-license decision.
4. Fully pinned immutable source, runtime, dependency, prompt/template, asset,
   protocol, route, cache/replay, and threshold profile.
5. Licensed final RQ3 manifest plus independent congruence coding/pretest,
   manipulation checks, power, and approved analysis plan.
6. Concrete matched-arm time/help/source/editor/export/instrument/fidelity values,
   tolerances, and allocation/estimand decisions.
7. Required ethics, privacy/security, methods, accessibility/localization,
   audience, arts, claim-audit, and release-owner decisions.

## Stop decision

Mandatory gates remain unresolved. Therefore G001/WP-0 is intentionally left
non-complete, WP-1 is not authorized, and Research mode/research-ready claims remain
disabled. No target-platform, licensing, matched-arm, or human-approval pass is
inferred from the structural/local evidence.
