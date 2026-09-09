# Interface QA — 2026-09-09

**Desktop repair follow-up:** the 12 desktop-applicable findings are now resolved; F04/F05 are excluded by the user’s mobile-scope instruction. See [completed goals, changes and verification](DESKTOP_QA_REPAIR.md). The report below preserves the original QA snapshot; its failure statuses and test names describe that historical state.

**Verdict: 14 confirmed defects: 3 high, 8 medium, 3 low.** The existing tests pass, but the additional scenarios expose credential handling, authorship, viewer integration, interaction, and recovery failures. Prioritize F01–F03 before classroom use.

This reviews the implemented working tree, including the pre-existing uncommitted changes, based on commit `bcd3bb9`. Production code, configuration, credentials, databases, and the running classroom service were not changed. New files contain QA tests and evidence only.

**Scope and evidence**

Tests ran on Linux with Python 3.12.13, Node 24.4.1, installed dependencies, Chromium, synthetic rosters, temporary databases, and loopback servers. Model-dependent execution used the fake embedder. The additional browser scenarios retained the real Content Security Policy. Where a viewer-capability probe or network response was substituted, that is stated below.

| Verification | Result | Evidence |
| --- | --- | --- |
| Existing server/unit/integration tests | 335 passed; no skips; two dependency deprecation warnings | [JUnit](qa/2026-09-09/unit.xml) |
| Existing browser suites: classroom, demo, schedule, security, viewer | 30 passed | [JUnit](qa/2026-09-09/browser.xml) |
| New browser scenarios | 19 checks: 6 passed, 13 confirmed failures before marking | [Raw assertions](qa/2026-09-09/adversarial-raw.log), [JUnit](qa/2026-09-09/adversarial-raw.xml) |
| New backend failure injection | 1 confirmed failure before marking | [Raw assertion](qa/2026-09-09/backend-fault-raw.log) |
| Final rerun of the new scenarios | 6 passed, 14 strict expected failures; no unexpected failures | [Browser rerun](qa/2026-09-09/adversarial-marked.log), [Backend rerun](qa/2026-09-09/backend-fault-marked.log) |
| 30 participants, 1,000 seeded opinions; 79→80 projection transition | 2 passed | [Timing log](qa/2026-09-09/load.log), [JUnit](qa/2026-09-09/load.xml) |
| Production frontend build | Passed, 323 modules; output directed to `/tmp/feedback-atlas-qa-build` | `npm run build -- --outDir /tmp/feedback-atlas-qa-build` |
| JavaScript syntax, Python compilation, diff whitespace | Passed | `node --check`, `python -m compileall`, `git diff --check` |
| Installed Python dependency consistency | Passed | `python -m pip check` |

The initial sandbox disallowed local sockets. Those setup errors were environmental; the successful runs above used authorized loopback access. There is no configured frontend lint/typecheck script or TypeScript project. Syntax/build checks are not a substitute for type checking. The installed environment has no Ruff or ShellCheck executable.

Measured load results used fake embeddings: first/second burst acknowledgement p95 **7.8/8.7 ms**, maximum broadcast latency **1329.6/689.2 ms**, heartbeat p95 **36.4 ms**, startup **32.568 s**. Accepted IDs, broadcasts, reconnect snapshots, and CSV rows remained consistent. These numbers do not establish real-model performance.

**Ranked findings**

| ID | Priority | Faulty scenario | Result |
| --- | --- | --- | --- |
| F01 | High | Submit login while startup metadata is delayed | Private access code enters the URL |
| F02 | High | Lost acknowledgement → restart → another person's login | Pending feedback is duplicated under the new author |
| F03 | High | Real viewer initializes categorical colors | Shipped setup SQL receives HTTP 500 |
| F04 | Medium | Touch-only participant taps an opinion | Opinion text never opens |
| F05 | Medium | Hover a long opinion in the phone layout | Tooltip content is clipped and cannot be scrolled |
| F06 | Medium | Roster changes while participants remain connected | Target dropdown retains obsolete choices |
| F07 | Medium | Search after selecting points | Nonmatching selected points stay prominently highlighted |
| F08 | Medium | Admin reconnects after choosing marquee/lasso | Mode control says selection; dragging pans |
| F09 | Medium | New opinion arrives while nearest opinions are open | Neighbor list and distances remain stale |
| F10 | Medium | Viewer backend is unavailable | Failed viewer leaves the working map hidden |
| F14 | Medium | Viewer relation replacement fails | Query endpoint returns successful stale results |
| F11 | Low | Reconnect after closing the compose panel | Panel reopens while its toggle remains off |
| F12 | Low | Restore a week after filtering out all points | Empty-filter status text remains |
| F13 | Low | Zoom the map | Distance scale remains unchanged |

All findings have executable reproductions. F14 requires an injected backend failure; its real-world frequency was not measured. F03's SQL failure was isolated from the limitations of the GPU mock.

**F01 — Access code leaks into the URL during incomplete startup**

Reproduce by delaying `GET /api/environment`, entering a synthetic code in the already visible landing form, and pressing Enter or the submit button. The page navigates to `/?code=writer-one-code`. Expected: the code is submitted only through the authenticated handshake, or the form remains unavailable until initialization completes.

The form defaults to native GET submission (`web/index.html:17–25`), while `web/app.js:18` awaits metadata before registering the handler at `web/app.js:589`. A password input conceals display but still participates in form submission. The resulting URL can persist in browser history and request logs; the observed evidence is the actual navigation URL. Slow metadata, slow modules, or script startup failure can expose this path.

Repair direction: make the initial HTML safe before JavaScript starts; prevent a native credential-bearing GET and enable login only after its handler is ready. Add an explicit startup failure state. Reproduction: `test_slow_environment_boot_does_not_put_access_code_in_url`.

**F02 — Pending feedback crosses the authenticated identity boundary**

Reproduce by submitting as writer1, dropping the acknowledgement after server persistence, restarting the test server, then using target2's code at the reauthentication gate. Without another submit action, the browser sends the old pending frame. CSV contains two copies, one attributed to writer1 and the other to target2. Expected: receipt recovery must remain bound to the original author; a different login must not automatically publish another person's draft.

`web/compose.js:104–110` resends pending frames on every online event. Fatal authentication recovery retains that pending state, while `web/app.js:431` announces the new connection and `src/server.py:749–753` looks up receipts using the current identity. A nonce protects retries only within one identity.

Repair direction: bind recovery to the original authenticated owner, or preserve the text while requiring a new explicit submission after an identity change. Reproduction: `test_pending_receipt_cannot_be_replayed_as_a_different_person`.

**F03 — The shipped viewer's categorical setup conflicts with backend SQL policy**

The real vendored viewer emits `ALTER TABLE dataset ADD COLUMN ... "__ev_source_id"; UPDATE dataset ...` during categorical setup (`web/vendor/embedding-atlas.js:67723–67730`). `web/viewer.js:152–155` configures category `source`. The backend applies a single-read-only-SELECT guard to every query kind, including `exec` (`src/mosaic_db.py:166–201`). The captured setup request returns **500: query must contain exactly one statement**, while a control SELECT returns 200.

This is a confirmed frontend/backend compatibility failure for categorical embedding setup. [The real browser trace](qa/2026-09-09/viewer-query-trace.json) captured the SQL; [independent HTTP and direct database replay](qa/2026-09-09/viewer-sql-replay.json) reproduced the rejection without GPU involvement. The browser trace also contains a mock-GPU `maxBufferSize` error, so it does not prove the entire viewer's blank appearance was caused solely by SQL. Full table/search/chart operation is not certified.

Repair direction: support the viewer's derived data through trusted server preprocessing, a compatible viewer configuration, or tightly constrained operations. Preserve participant privacy and SQL hardening; simply allowing arbitrary writes is not an acceptable resolution. Reproduction: `test_vendored_category_setup_query_is_supported`. If the repair removes that SQL requirement, replace this captured-query probe with a corresponding real-bundle integration assertion.

**F04 — Participants cannot read opinions by touch**

On a 390×844 touch context, log in, hide the legend, and tap an existing opinion. No `.tip .text` appears. Expected: tapping should expose the full opinion and a way to dismiss it. The participant has neither the admin list nor the GPU viewer fallback on this device.

Hover text is opened by pointer movement (`web/atlas.js:346`, `web/atlas.js:412`), whereas taps dispatch `onPick` (`web/atlas.js:370–376`), which is wired only for admins (`web/admin.js:261–263`). Pointer leave also removes the tooltip. Add a touch/keyboard reading interaction. Reproduction: `test_mobile_tap_opens_opinion_text`.

**F05 — Long opinion tooltips are unreadable**

Submit the test's long Korean opinion within the 1,000-character limit and hover it in the phone layout. Tooltip bounds were top **342.1 px**, bottom **1080.3 px**, while the map ended at **407.1 px**. Expected: the whole opinion should be readable through a bounded, scrollable surface or an expanded view.

`web/app.js:106–113` flips tall tooltips below the point without a bottom bound. The map clips overflow (`web/app.css:101–103`), and the tooltip has `pointer-events:none` with no scrollable reading path (`web/app.css:136–140`). The [screenshot](qa/2026-09-09/long-tooltip.png) illustrates the affected layout; the decisive evidence is the measured DOM bounds in the raw assertion. Reproduction: `test_long_opinion_tooltip_is_readable_within_map`.

**F06 — Roster reload does not refresh connected target controls**

Select target2, compose text, remove target2 from the synthetic roster, and send SIGHUP. Wait for the server's `viewer_refresh` notification. The dropdown still contains and selects target2. Expected: a removed target requires reselection; added/renamed targets should be reflected without losing the draft.

`src/server.py:204–224` reloads the roster and viewer relation. `src/server.py:523–530` broadcasts only `viewer_refresh`; `web/app.js:483–485` refreshes only an open viewer. Targets are rebuilt on `hello_ok`, and compose rebuilds them only on `atlas:hello`. The server will reject an obsolete target even though the UI continues offering it.

Repair direction: broadcast public target metadata and refresh controls while preserving valid selections/drafts. Reproduction: `test_roster_target_removal_requires_explicit_reselection`.

**F07 — Selection overrides search emphasis**

In admin mode, marquee-select all three seeded opinions, then search for text present in only one. The UI reports one match, but a nonmatching selected point retains opacity **0.95**. Expected: selected styling should preserve the active search/target/reviewer emphasis.

`web/atlas.js:166–168` computes dimming, then the selected branch at `web/atlas.js:194–197` overwrites it whenever the week is visible. The separate week filter does correctly dim selected points; that scenario passed. CSV also retains the original selection (`web/admin.js:175–178`), so any intended export/filter relationship should be made explicit rather than inferred from highlight state. Reproduction: `test_selected_points_still_obey_search_focus`.

**F08 — Admin mode control becomes misleading after reconnect**

Choose marquee, restart the temporary server, wait for automatic admin reconnection, then drag the map. The marquee button still has `aria-pressed=true`, but the drag pans and no points are selected. Expected: the selected control matches actual pointer behavior.

`web/admin.js:254–257` resets the Atlas mode to pan on every hello without updating the buttons maintained by `web/admin.js:109–116`. Preserve the mode or reset both model and controls together. Reproduction: `test_admin_selection_mode_remains_consistent_after_reconnect`.

**F09 — Nearest-opinion results do not refresh with the corpus**

With three opinions, open one point's two nearest neighbors. Submit a fourth opinion from another page. The map reaches four points, but the neighbor panel remains at two rows, including after a three-second wait. Since the request allows eight neighbors, the current corpus should yield three. Expected: refresh the list/distances or visibly mark the old result stale.

`web/admin.js:134–161` fetches neighbors only on pick; `web/admin.js:259` handles data changes by refreshing reviewers and search alone. Reproduction: `test_nearest_panel_refreshes_after_new_opinion`.

**F10 — Failed viewer initialization leaves the fallback map hidden**

Pass the viewer capability probe, then make its query endpoint return the documented disabled-viewer 503. The real mount/connector displays the error note, but the viewer toggle stays active and the map remains hidden. Expected: return to the working map or present a direct recovery action. A user can currently recover by manually toggling the viewer off.

`web/app.js:240–244` switches panels before mounting; the catch at `web/app.js:257–258` only sets text. The regression substitutes the capability probe and endpoint response, not the mount/connector. Reproduction: `test_viewer_backend_failure_returns_to_working_map`.

**F14 — A failed viewer refresh serves stale data as success**

After a clean test-app startup, inject a failure in `MosaicService.replace`, then submit feedback. The submission is acknowledged and the websocket broadcasts one point, but an authenticated `SELECT COUNT(*)` returns **HTTP 200 with n=0**. `/healthz` records the exception; ordinary viewer queries provide no stale/unavailable signal.

`src/atlas_state.py:221–230` catches relation-refresh failures, while the recompute loop still commits and broadcasts (`src/atlas_state.py:422–424`). Keeping submission/map operation available is useful, but serving stale results as current contradicts the frontend's consistency assumption. Mark the viewer unavailable/stale until a successful rebuild and expose a recoverable state. Reproduction: `test_failed_viewer_refresh_does_not_silently_serve_stale_rows` in [the backend scenario file](tests/qa_backend_scenarios.py).

**F11–F13 — State and status inconsistencies**

- **F11:** Close compose, restart/rejoin, and observe the panel reopening while its toggle remains false. `web/compose.js:99–101` sets visibility only from the role on every hello. Preserve the panel choice or synchronize the toggle. Reproduction: `test_closed_compose_choice_survives_reconnect`.
- **F12:** Disable the only populated week, then enable it. The count returns to three, but “주차가 모두 꺼져 있습니다” remains. The same message is inaccurate when other empty weeks are enabled. `web/app.js:123` sets the text without clearing it when points return. Reproduction: `test_filtered_empty_status_clears_when_points_return`.
- **F13:** Zoom with the wheel. Point coordinates change, but the scale SVG and its numeric label are identical. `web/atlas.js:397–409` redraws the map without invoking the status/scale calculation at `web/app.js:124–127`. Reproduction: `test_map_zoom_updates_distance_scale`.

**Passing behavior exercised**

The existing suites cover invalid credentials, code/session restoration, participant authorship privacy, classroom/demo separation, Unicode and markup-shaped opinions, same-author receipt recovery, schedule boundaries and receipt recovery after a period closes, search, marquee selection, neighbors, CSV, responsive widths 360/390/768/1440, theme changes, viewer-unavailable fallback, and delayed capability-probe/reconnect choices.

The six additional passing browser scenarios verify whitespace validation plus actual rate-limit refusal without draft loss; preserving newly edited text when an older acknowledgement arrives; concurrent schedule edits with a 409 and explicit reload; overlapping-date refusal and cancellation; failed schedule save followed by retry; and week filtering of selected points. These passes do not negate the more specific failures above.

**Reproduction and expected failures**

The new files are opt-in, consistent with the repository's existing browser-test convention. The 14 known defects are marked `xfail(strict=True, raises=AssertionError)` with finding IDs. Expected failures are recorded defects, not passing features. Unexpected exceptions remain failures; an unexpected pass requires reviewing the finding and removing its marker.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/browser_classroom.py tests/browser_demo.py tests/browser_schedule.py tests/browser_security.py tests/browser_viewer.py -q
.venv/bin/python -m pytest tests/classroom_load.py -q -s
.venv/bin/python -m pytest tests/browser_adversarial.py tests/qa_backend_scenarios.py -q -rx
# Reproduce the unmarked failures:
.venv/bin/python -m pytest tests/browser_adversarial.py tests/qa_backend_scenarios.py -q --runxfail --tb=short
cd frontend
npm run build -- --outDir /tmp/feedback-atlas-qa-build
```

Run local browser/server tests where loopback sockets and Chromium process creation are permitted. No test needs production credentials. The raw logs predate the addition of markers, so their test-file line numbers can differ from the final test files; test names and finding IDs are stable.

**Limits and remaining risks**

- Chromium desktop and touch-emulated phone coverage is confirmed. Installed system Firefox could not launch under Playwright; the Playwright WebKit executable is absent. Native iOS/Android keyboards, Safari, Firefox, accessibility assistive technology, and real phone gestures remain unverified.
- This container has no working WebGPU adapter. GPU rendering, visual fidelity, and complete real viewer table/search/cross-filter workflows remain unverified. SQL compatibility is independently confirmed faulty. [Browser capability evidence](qa/2026-09-09/browser-capabilities.json) records the launch limitations.
- The load tests use fake embeddings. Real embedding-model latency, cold downloads, GPU memory pressure, production TLS/certificate renewal, and classroom network behavior were not exercised.
- F14 proves behavior under an injected relation-replacement exception; it does not establish how often that exception occurs in deployment.
- This was functional QA, not a dependency-advisory or penetration audit. Read-only review also noticed different invalid-admin-code delay behavior on schedule updates; rate limiting still exists, and this was not counted as a confirmed functional defect. Older browser dialog support has no established acceptance target here.

**Changes made:** added this report, [browser reproductions](tests/browser_adversarial.py), [backend failure injection](tests/qa_backend_scenarios.py), and evidence under [qa/2026-09-09](qa/2026-09-09). No production simplifications or fixes were made. The 14 findings remain open.
