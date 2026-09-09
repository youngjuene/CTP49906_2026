# Desktop QA repairs — completed 2026-09-09

**All four goals are complete. All 12 desktop-applicable QA findings are resolved.** F04 (touch-only interaction) and F05 (phone tooltip clipping) are excluded by the user's instruction. The original F05 scenario passed when replayed at 1440×900. Mobile behavior was not a repair acceptance requirement.

| Goal | Findings | Result |
| --- | --- | --- |
| G001: safe login and receipt recovery | F01, F02 | Complete: inert startup, safe POST fallback, bounded metadata loading, visible boot retry, and identity-bound receipt recovery |
| G002: consistent desktop controls and current admin data | F06, F07, F08, F09, F11, F12, F13 | Complete: audience-safe live roster updates, correct selection emphasis/mode, fresh nonce-correlated neighbors, preserved compose choice, current status and zoom scale |
| G003: compatible and recoverable viewer | F03, F10, F14 | Complete: read-only category expressions, visible fallback/retry, stale-query503 gating until recovery, real table/search/filter/category checks |
| G004: integrated verification and review | All accepted findings | Complete: tests/build/static checks, narrow cleanup, independent code review and recorded evidence |

**What changed**

| Files | Repair |
| --- | --- |
| [web/index.html](web/index.html), [web/bootstrap.js](web/bootstrap.js), [web/app-context.js](web/app-context.js) | Login is unavailable until initialized; native submission uses POST; delayed/failed startup cannot put a code in the URL; module failure has an actual retry path |
| [src/access.py](src/access.py), [web/compose.js](web/compose.js) | Opaque owner context survives unchanged-code restarts and differs by identity/environment/code rotation; another login preserves pending text without automatically republishing it |
| [src/server.py](src/server.py), [src/protocol.py](src/protocol.py), [web/app.js](web/app.js) | Add owner/roster metadata in the appropriate audience, process it before resending, echo bounded neighbor nonces, and keep UI/fallback state accurate |
| [web/admin.js](web/admin.js), [web/atlas.js](web/atlas.js) | Preserve mode and selected-list state, dim selected nonmatches, refresh neighbors, reject superseded responses, and notify status rendering on camera changes |
| [src/atlas_state.py](src/atlas_state.py), [web/viewer.js](web/viewer.js) | Track viewer errors separately; recover both relations before serving queries; catch in-flight failures and stale asynchronous completions; retain the viewer instance across ordinary hide/reopen |
| [frontend/readonly-category-transform.js](frontend/readonly-category-transform.js), [frontend/vite.config.js](frontend/vite.config.js), [frontend/package.json](frontend/package.json), [web/vendor/embedding-atlas.js](web/vendor/embedding-atlas.js) | Reproducibly adapt pinned Embedding Atlas 0.24.0 category/bin queries to inline read-only expressions and refresh their metadata without resetting the dashboard; regenerate the vendor bundle; add guard-test command |
| [src/mosaic_db.py](src/mosaic_db.py), [README.md](README.md) | Correct obsolete writable-SQL documentation and explain the guarded build adapter |
| [tests/browser_adversarial.py](tests/browser_adversarial.py), [tests/browser_admin_repair.py](tests/browser_admin_repair.py), [tests/browser_boot_repair.py](tests/browser_boot_repair.py), [tests/browser_viewer_contract.py](tests/browser_viewer_contract.py) | Turn accepted faults into ordinary desktop regressions; exclude only the two mobile cases; verify the real vendor contract and recovery |
| [tests/test_access.py](tests/test_access.py), [tests/test_roster_updates.py](tests/test_roster_updates.py), [tests/test_viewer_pipeline.py](tests/test_viewer_pipeline.py), [tests/qa_backend_scenarios.py](tests/qa_backend_scenarios.py), [frontend/tests/readonly-category-transform.test.js](frontend/tests/readonly-category-transform.test.js) | Cover owner privacy/scoping, roster updates without a viewer, relation failures/recovery, in-flight queries, and fail-closed adapter drift guards |

Real-viewer verification uncovered two additional issues within G003: worker assets used origin-root URLs and returned 404, leaving full-text search indexing forever; and an already-built search index missed later feedback under the same predicate. Relative asset URLs fix classroom/demo packaging. Corpus-version invalidation now rebuilds the native search index while retaining its serialized rebuild queue. Neither repair adds dependencies or permits SQL mutations.

The compatibility decision uses the installed upstream code and the [Embedding Atlas API](https://apple.github.io/embedding-atlas/embedding-atlas.html) and [EmbeddingViewMosaic API](https://apple.github.io/embedding-atlas/embedding-view-mosaic.html). Its full dashboard category setup lacked a supported read-only integration hook. The adapter is guarded by the exact dependency version, source hashes, helper boundaries, and category read sites. An upstream change deliberately fails the build instead of silently restoring the mutation path.

**Final verification**

| Check | Result | Evidence |
| --- | --- | --- |
| Full backend suite | **338 passed**, no skips | [JUnit](qa/2026-09-09-repair/unit.xml), [log](qa/2026-09-09-repair/unit.log) |
| Combined browser suites | **64 passed, 2 intentionally skipped** | [JUnit](qa/2026-09-09-repair/browser.xml), [log](qa/2026-09-09-repair/browser.log) |
| Viewer failure/recovery, including in-flight query race | **4 passed** | [log](qa/2026-09-09-repair/recovery.log) |
| 30 participants, 1,000 seeded opinions and 79→80 projection transition | **2 passed** | [JUnit](qa/2026-09-09-repair/load.xml), [timings](qa/2026-09-09-repair/load.log) |
| Real-bundle desktop contract, included in combined browser total | **10 passed** | [focused run](qa/2026-09-09-repair/real-viewer.log) |
| Adapter guard tests and frontend production build | Passed | `npm run test:readonly-category`; `npm run build -- --outDir /tmp/feedback-atlas-repair-build` |
| Generated asset comparison | All served vendor files match the clean build | Byte-for-byte comparison against the temporary build |
| JavaScript syntax, Python compilation, diff whitespace, installed Python dependencies | Passed | `node --check`; `python -m compileall`; `git diff --check`; `python -m pip check` |

The 10 real-viewer checks cover source setup/table loading; category changes to week, target name, timestamp and numeric x; live category/table updates and reopening; an empty dataset's first opinion; actual full-text worker/search including newly added data; cross-filter persistence/clearing; and more than 10 target categories with quoted labels. These use the real bundle, connector and database. GPU device acquisition is held pending, so these checks do not claim GPU drawing coverage.

Load measurements with fake embeddings: acknowledgement p95 **10.5/6.3 ms**, maximum broadcast latency **1468.5/517.2 ms**, heartbeat p95 **30.4 ms** across two bursts. No lost or duplicate accepted opinions. Startup was 30.923 s. These are synthetic-model measurements, not production-model guarantees.

**Invariants and review**

- Credentials stay out of URLs and persistent browser storage; existing session tokens remain in session storage.
- Participant SQL and roster frames exclude reviewer authorship. Classroom/demo data and owner contexts remain separate.
- Same-owner receipt recovery remains durable and idempotent; a different owner cannot trigger automatic replay.
- Client SQL remains single-statement, bounded and read-only, with file/network access disabled. Old mutation-shaped requests still fail.
- Viewer refresh failure does not prevent feedback acceptance or map updates. Viewer queries return 503 until recovery, including queries spanning a failure/recovery race.
- Desktop filters, modes, camera/status, drafts and viewer choices remain coherent through reconnect and data changes.

Two native code-reviewer passes, including follow-up review of the worker/search fixes, found **zero remaining changed-code defects**. [Review evidence](.omx/plans/review-desktop-qa-repair.md) records the disposition. There is no configured TypeScript project/lint command, and code-intel transport became unavailable; reviewers recorded that limitation. The configured architect role could not run, so no formal two-lane OMX merge approval is claimed. This was an implementation task, not a merge/deployment task.

The [implementation plan](.omx/plans/prd-desktop-qa-repair.md) was independently reviewed by an available critic. Execution used one native aggregate goal after the optional Ralplan/Ultragoal architect gate proved unavailable. No approval was fabricated and no extra user permission was needed for the authorized fixes.

**Cleanup and remaining limits**

Cleanup plan: restrict work to changed files after regression coverage; remove unused adapter exports/constants and obsolete comments; replace implementation-shaped assertions with actual ES-module compilation and negative build-drift checks. Retain the version-specific adapter, visible map fallback and async cancellation guards as tested compatibility/recovery boundaries. Completed that pass and reran relevant build/browser checks. No broad refactor or dependency additions were made.

The pre-existing uncommitted workspace changes were preserved. Production credentials and databases were not modified, and the running service was not restarted. Backend changes require the normal deployment/restart process to take effect in an already-running service.

Remaining limits: physical WebGPU rendering, Safari/Firefox and real-model performance were not verified in this environment. Mobile-only F04/F05 are consciously outside scope. The pinned adapter requires review when upgrading Embedding Atlas. This report does not certify all possible browser/hardware scenarios.

**Reproduce**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/browser_classroom.py tests/browser_demo.py tests/browser_schedule.py tests/browser_security.py tests/browser_viewer.py tests/browser_adversarial.py tests/browser_admin_repair.py tests/browser_boot_repair.py tests/browser_viewer_contract.py -q
.venv/bin/python -m pytest tests/qa_backend_scenarios.py tests/classroom_load.py -q
cd frontend
npm run test:readonly-category
npm run build
```

Local socket/process access is required for browser/server tests. All fixtures use synthetic credentials and temporary data. Historical fault logs remain under [qa/2026-09-09](qa/2026-09-09); current results are under [qa/2026-09-09-repair](qa/2026-09-09-repair). No in-scope test remains marked xfail.
