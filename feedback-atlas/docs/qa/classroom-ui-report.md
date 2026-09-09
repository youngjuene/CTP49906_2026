# Classroom browser implementation evidence

Execution source: local mirror `/Users/youngjuene/Code/ctp49906/feedback-atlas`, synchronized only to the authorized isolated June checkout. No original live checkout, real classroom SQLite data, or deployment was changed by this browser work. No commits created.

## Implemented behavior

- V2 one-paste submit sends the exact textarea value, including whitespace and Unicode decomposition. Counts use Unicode code points; there is no `maxlength` truncation. Source remains the intentional AI default with explicit `피드백 출처 / AI 생성 / 사람 작성` labels and source/submitter guidance.
- IndexedDB stores drafts, immutable outbox envelopes, receipts/capabilities and expiring tab leases. A 32-byte cryptographic capability and nonce are persisted before wire transmission. Acknowledgements commit receipts and remove only the acknowledged envelope; draft clearing is guarded by its immutable version. Storage failures preserve visible text and offer a text download. Explicit local clearing warns that pending text and owner access are removed.
- Optimistic draft saves reject an unseen newer version from another tab and preserve the visible conflicting draft for download. Reconnect does not silently adopt another tab's newer draft as a writable base.
- Private submitter identity is shown locally; changing IDs flushes the current draft, closes the connection, clears visible/private state, and restores only the new ID/dataset scope.
- Empty forms follow classroom context; dirty metadata stays fixed until explicit current-context or retained-context confirmation. Blocked old envelopes are resolved through the private receipt endpoint before replacement. Closed intake keeps unsent text locally.
- Owner status cards separate local save, transmission waiting, original saved, processing, ready, failed and withdrawn. Owners can inspect original text, source and revision history, change parent target/week, place a split using a readonly text cursor and boundary preview, merge adjacent units, correct each unit's source and withdraw the entire parent.
- Published originals open through a public reader that never requests identity or capability data. Keyboard users have a visible-results list; pointer taps open the same reader. Text and history are rendered using textContent.
- The classroom map is the default at all widths. One query state defines week, search, target/source membership, selected IDs and separate emphasis. Hidden members retain coordinates. Counts distinguish units and unique original submissions. Admin export previews visible/selected/all scope, unit/original format, row count and data revision; empty selection stays empty and a 409 requires another click.
- Instructor context/open-close controls and live processing counts/queue diagnostics are connected. Admin totals consume server-computed unit, original and distinct submitter aggregates; no participant count is inferred from units. Optional `projection_method` metadata drives the actual PCA-to-UMAP transition announcement.
- Advanced analysis is deliberately disabled with a visible reason until its pinned API can satisfy canonical filter and selection parity. Old tests that asserted automatic wide-screen enabling were replaced with this approved release behavior. The root agent owns build-time worker-path repair and real-bundle verification.

## Test-first evidence

1. Before production changes, `tests/browser_submission_flow.py --maxfail=1` failed entering the classroom because browser protocol 1 was incompatible with the new protocol 2 server: **1 failed in 31.95s**. The new recovery, correction and export modules existed before implementation.
2. First integrated browser run reported by root: **6 passed, 1 failed**. The failure was a fixture assumption: it expected two units after manually splitting, although the automatic splitter already produced two. The fixture now asserts the original unit count plus one and does not constrain production semantic behavior.
3. Root confirmed red for `test_second_tab_cannot_overwrite_a_newer_persisted_draft`: the stale tab overwrote the newer text. Optimistic version checks and serialized UI saves were added after that failure.
4. Root confirmed red for touch focus: Escape did not focus the map; a focusable map and replacement-trigger fallback were then added.
5. Root confirmed red for simultaneous enqueue: two concurrent calls created two outbox items; the enqueue transaction now permits only one pending envelope per scope. Final outcomes are recorded below when returned by the coordinating run.

Local syntax checks: `node --check` passed for `web/app.js`, `web/admin.js`, `web/compose.js`, `web/draft-store.js`, and `web/feedback-reader.js`.

## Review regressions and integration

- Reviewer reproduced rejected-envelope A being lost while draft B existed. The corrected flow atomically replaces only the rejected immutable envelope after checking its receipt, retaining A and B independently. A real-server test withholds A until class context changes, types B, receives the actual stale-context rejection, confirms A and verifies B survives reload.
- Reviewer reproduced a timed-out correction being replayed when the user asked to withdraw. Unknown mutation outcomes now lock other actions and expose an explicit retry of the original nonce. The real-server test drops one correction response and verifies two identical correction nonces followed by a separate requested withdrawal.
- Reviewer reproduced a late A correction response replacing open original B. Reader generations and captured submission/action identities guard results, errors and cleanup. The real-server test delays A's response, opens and edits B, then releases A and verifies B remains intact.
- The previous integrated run reported **27 passed, 1 failed in 130.80s**. The sole failure was SIGHUP target-name refresh: the backend only sent a cache-invalidation frame. The root added public targets/admin-only roster to that frame; the browser now updates live controls while preserving drafts.
- Chromium's readonly textarea does not move a collapsed caret for plain ArrowRight. A focused red showed selectionStart remained 0. Explicit navigation made the full split/source/history/withdraw scenario pass. A subsequent emoji regression showed code-point stepping split `👩🏽‍💻`; navigation now uses `Intl.Segmenter` grapheme boundaries and converts the resulting UTF-16 position back to original code-point offsets.
- Native dialog close/focus events are asynchronous. The touch regression waits for actual focus restoration rather than asserting in the same event turn.
- Socket loss immediately rejects pending RPC waits, allowing unchanged outbox envelopes to recover after the new handshake instead of waiting for an 8-second request timeout.
- Correction history shows revision, target, week, unit count and source counts in Korean, without exposing its stored JSON representation.

## Final browser verification

The following command ran in the isolated June checkout against temporary databases and rosters:

```bash
.venv/bin/python -m pytest tests/browser_classroom.py tests/browser_submission_flow.py tests/browser_submission_corrections.py tests/browser_export_scopes.py tests/browser_viewer.py -q --maxfail=3
```

**29 passed in 100.25 seconds; exit status 0.** This includes all later review regressions, the emoji grapheme case, admin aggregate display, readable history, live roster refresh, and all existing classroom scenarios updated for the approved v2 behavior. No skips occurred in this run.

## Final protocol-mismatch increment

The last approved Task 5 gap is closed. Protocol mismatch now exposes explicit current-text download and refresh controls on the gate. Refresh remains disabled until local draft preservation succeeds or the user starts a text download after storage failure. There is no automatic reload, existing drafts/outbox/receipts are not cleared, and the entered ID remains remembered for same-dataset recovery.

A real-server regression first failed because `#protocol-refresh` did not exist (**1 failed in 2.68s**). The test holds original A before wire delivery, types newer B, sends an actual protocol-1 hello to the protocol-2 server, and checks both records survive. It then downloads B and explicitly refreshes, recovering B while the unchanged A envelope is delivered. A second case denies IndexedDB and verifies refresh stays disabled until the visible text is downloaded.

```bash
.venv/bin/python -m pytest tests/browser_submission_flow.py::test_protocol_mismatch_preserves_outbox_and_new_draft_until_explicit_refresh tests/browser_submission_flow.py::test_protocol_mismatch_requires_download_when_storage_is_unavailable -q
```

**2 passed in 17.54 seconds; exit status 0**, with the existing pytest plugin assertion-rewrite warning. The earlier full run remains **29 passed in 100.25 seconds**; it preceded these last controls, so this report does not relabel it as a 31-case full run.

Final increment paths: `web/app.js`, `web/compose.js`, `web/index.html`, `tests/browser_submission_flow.py`, and this report. The reported trailing whitespace on `web/compose.js` line 6 was removed. No README edits or commits were made by this worker.

## Remaining limits

- Physical Mac/phone classroom QR checks were not performed. Touch and responsive checks are Chromium emulation, not physical-device evidence.
- Model quality, real-model latency, holding out human judgments, and release gates are owned by the root/semantic tasks. Browser tests use a real temporary server and database with the fake model; they make no semantic-quality claim.
- Raw preservation is exact for the textarea value submitted to the server. Native textarea line-ending normalization occurs before that value is read; no claim is made that browser-normalized clipboard CRLF bytes survive as different byte sequences.
- Advanced viewer parity remains unavailable and the viewer stays disabled. There is no product deployment in this increment.


## Coordinating final verification

After the last protocol controls were synchronized, the coordinating agent ran all six browser modules on that final tree:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest tests/browser_classroom.py tests/browser_submission_flow.py tests/browser_submission_corrections.py tests/browser_export_scopes.py tests/browser_viewer.py tests/browser_viewer_assets.py -q --maxfail=2
```

**32 passed, 1 skipped in 124.01 seconds; exit status 0.** This includes all 31 interaction scenarios and the emitted-worker asset check. The sole skip is the real viewer renderer because the environment lacks a compatible WebGPU adapter. The earlier 29-case run and two focused protocol tests above remain the development progression, not additional unique cases.

The root also completed a Mac embedded-browser walkthrough through the temporary localhost SSH tunnel using synthetic data and the fake model; see the release report. This does not supply the still-missing physical-phone/actual-classroom-URL evidence. Owned preview tabs, server and tunnel were closed afterward.
