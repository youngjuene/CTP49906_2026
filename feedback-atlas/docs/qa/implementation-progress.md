# Implementation ledger — Feedback Atlas classroom plan

User approved implementation on 2026-09-09. Work is preserved on `codex/feedback-atlas-classroom-20260909` in `/mnt/hdd/research/2026/.codex-worktrees/ctp49906-feedback-atlas-20260909/feedback-atlas` on `ssh june`. The local source mirror is `/Users/youngjuene/Code/ctp49906/feedback-atlas`.

Baseline snapshot: `c02de3e277f34150efd7ffb7639184f1897cec7f`. All 74 captured source hashes in the original checkout remain unchanged. Production deployment was not performed.

The [final release report](feedback-atlas-release.md) owns current evidence and limitations; the plan and design remain historical approved inputs.

| Plan task | Implementation status |
| --- | --- |
| 0 — Baseline | Isolated snapshot, source manifest and binary patch preserved; 83 focused baseline checks passed |
| 1 — Semantic experiment | Shared-model candidate and frozen calibration/held-out evaluation complete; **quality gate failed**, human labels absent |
| 2 — Records/migration | Immutable parents/revisions/spans, owner capabilities, legacy backfill, private consistent backup and rollback implemented |
| 3 — Protocol v2 | Durable long-paste acceptance, nonce replay, input bounds, context validation and private owner/admin routes implemented |
| 4 — Worker/publication | Shared executor, persistent queue, retry, complete CAS publication, withdrawal guards and metadata-only geometry reuse implemented |
| 5 — Composition/recovery | IndexedDB draft/outbox/receipts/leases, explicit errors/download, version guards and recovery implemented; final protocol refresh controls verified in the combined browser run |
| 6 — Classroom context | Join/change-ID, frozen dirty context, explicit confirmation, intake controls and live roster refresh implemented |
| 7 — Inspection/corrections | Original reader, per-unit source, grapheme-aware split, merge, target/week/history and withdrawal implemented |
| 8 — Map/counts | Responsive classroom map, keyboard/touch reader, stable filters, units/originals/server distinct-submitter counts implemented |
| 9 — Scope/export | Canonical membership/selection, separate emphasis and revision-bound visible/selected/all CSV exports implemented |
| 10 — Advanced analysis | Worker asset paths repaired and served correctly; viewer kept disabled under the approved parity fallback; full GPU test skipped for missing adapter |
| 11 — Operations | Exact/idempotent imports, all-state archives, private backups, launcher gate and recovery documentation implemented |
| 12 — Integrated QA/release | 335 default tests; final combined browser run 32 passed/1 GPU skip; real CPU load and Mac synthetic walkthrough recorded. **No classroom release**: semantic and actual-device gates remain open |

Independent review found two backend and three UI defects. Each was reproduced before its fix; final re-review closed all five. The UI report records the focused protocol increment and the final combined 32-pass/1-skip collection. Existing dependency pins remain; the rejected stock Chonkie integration adds no runtime dependency. GPU evaluation was unavailable due to the installed driver/runtime mismatch, so real-model measurements use CPU.

Measured 30-participant load with 1,000 starting units: representative 3,176-code-point originals had receipt p95 48.11 ms, ready p95 17.29 s; maximum 20,000-code-point originals had receipt p95 54.47 ms, ready p95 24.93 s. All results finished within 18.82/27.14 s respectively. These performance results do not clear the semantic failure: held-out single-topic preservation was only 3/8 (37.5%).
