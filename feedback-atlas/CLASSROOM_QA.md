# Classroom readiness QA — 2026-09-09

Status: **NO-GO for automatic classroom segmentation**. The frozen held-out synthetic evaluation preserves only 3/8 single-topic cases (37.5%), and two-reader human judgments are absent. Structural and operations checks do not override that quality failure. Target workload: 30 simultaneous participants, up to 1,000 existing units.

The current implementation and measured results are recorded in the [release report](docs/qa/feedback-atlas-release.md). Tests used synthetic rosters, temporary databases and owned server processes. No live class was opened, stopped or migrated.

| Area | Evidence | Release status |
| --- | --- | --- |
| Backend contracts, migration, ownership and publication | 335 default tests passed | Automated checks complete |
| Join, source/context, submit, recovery, correction and export | 31 interaction cases passed in the final combined run; with worker checks: 32 passed/1 GPU skip; see [browser report](docs/qa/classroom-ui-report.md) | Synthetic browser checks complete |
| Semantic boundaries | Required recall 91.43%; forbidden cuts 5% pooled; single-topic preservation **37.5%** | **Failed; human judgments absent** |
| 30 participants, 1,000 existing units, real CPU model | Representative all-ready upper bound 18.82 s; maximum-size 27.14 s; exact originals and idempotent replay | Tested workload meets latency thresholds |
| PCA-to-UMAP threshold and short-input load | Two v2 fake-model tests passed | Functional evidence only |
| Advanced viewer assets/rendering | Three emitted worker paths served JavaScript; clustering worker starts; one real-renderer test skipped for missing compatible GPU adapter | Viewer disabled pending parity and rendering checks |
| Source privacy and safe display/export | Private identity/capability controls, public field allowlists, SQL relation separation and CSV escaping exercised | Does not add identity authentication; existing SQL sandbox limits remain in README |
| Imports, archives, backup, launch/close | 43 focused operations/store tests passed; later migration/diagnostic coverage included in default suite | Live operator walkthrough still pending |
| Mac participant/instructor interaction | Temporary localhost synthetic preview: submit, inspect, reload, source correction, search and scope preview | Manual software walkthrough complete; fake model |
| Two people, physical phone, actual classroom QR URL | Not performed | Pending before release |
| Build/source preservation | Frontend build succeeded; all 74 captured original source hashes unchanged | Isolated implementation retained; no deployment |

## V2 operations evidence

See [operations report](docs/qa/operations-report.md) for exact test progression and commands, and [semantic evaluation](docs/qa/segmentation-evaluation.md) for the frozen calibration/held-out results. No actual class was opened or shut down by these tests.

- Exact raw offline imports, default already-unitized mode, opt-in semantic mode, idempotent file/row replay, capacity limits and atomic initial staging: tested against temporary databases.
- Pseudonymous archive copies preserve every original/revision/span, including pending/failed work, while removing ownership/replay handles and consistently remapping identifiers. Raw prose can still identify people.
- Full private recovery backups include committed WAL content and ownership hashes, refuse overwrite and use mode 0600. They are distinct from sanitized archives.
- Launcher checks model/semantic readiness before a tunnel, labels the explicit development override, checks intake/queue before shutdown and creates a backup before stopping. Test process signals are stubbed.

Before a real release, independently review boundaries, rerun changed semantic behavior on an appropriately held-out corpus, and complete the actual-device/operator scenario. Do not set `classroom_enabled=true` or the development override to disguise a failed quality gate. `ATLAS_ALLOW_UNREVIEWED_SEMANTICS` defaults to 0 in `.env.example`.

Operator close sequence: close intake → check/drain pending work and resolve or preserve failures → export all accepted originals through instructor recovery export → create a private backup → stop. Preserve unsent browser drafts separately. A new quick-tunnel origin cannot read storage from the previous origin. Before rollback, preserve the current database and account for all originals accepted since the older snapshot.
