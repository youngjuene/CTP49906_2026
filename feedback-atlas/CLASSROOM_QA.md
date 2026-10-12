# Classroom readiness QA — 2026-09-08, results rerun 2026-10-12

Status: **Automated checks pass. Not yet exercised on the production host, with the
default model, on a GPU, or with a real class.** Target: 30 simultaneous
participants, up to 1,000 opinions.

Acceptance: roster users can join and submit within one minute; accepted opinions
persist and reach other screens within a few seconds; reconnects do not lose drafts
or strand submission controls; participant traffic excludes reviewer identity;
instructor tools and launch/recovery instructions work.

Tests use synthetic rosters/opinions, temporary databases, and owned local server
processes. No production opinions, credentials, or running class service are changed.

**How the results below were produced (2026-10-12, macOS, Python 3.12, Node 22,
Chromium via Playwright 1.63):** `pytest` (338 passed, 0 skipped); the nine browser
suites (68 passed, 0 skipped); `tests/classroom_load.py` and
`tests/qa_backend_scenarios.py` (6 passed); `npm run test:readonly-category` (passed);
`node --check web/*.js`, `compileall`, and `npm run build` into a scratch directory.
The built output is byte-identical to the committed `web/vendor`.

"Pass" means the automated test named in the evidence column passes. Browser rows are
Chromium only. Rows marked **Partial** have a stated gap.

| ID | Scenario / user model | Evidence | Result |
| --- | --- | --- | --- |
| B01 | Existing automated regressions | 338 passed, no skips | **Pass** |
| U01 | New participant, wrong ID, normalized ID, phone viewport | `browser_classroom::test_join_validation_and_responsive_form`, `test_roster.py`, `test_textnorm.py` | **Pass** |
| U02 | Two participants submit Korean, Unicode, markup text | `browser_classroom::test_live_submission_unicode_privacy_filters_and_theme`, `test_wire_safety.py` | **Pass** |
| U03 | Week filters and theme / viewport changes | same browser test; coordinates-unchanged check; widths 360/390/768/1440 per the 2026-09-09 report | **Pass** |
| U04 | Network restart while composing | `browser_classroom::test_reconnect_preserves_the_entire_draft` | **Pass** |
| U05 | Connection drops after persistence but before acknowledgement | `browser_classroom::test_lost_ack_recovers_without_duplicate_submission`, `test_submission_retry.py`, `browser_adversarial` receipt-owner cases | **Pass** |
| U06 | Instructor login, search, selection, neighbors, CSV | `browser_classroom::test_admin_search_neighbors_selection_and_export` | **Pass** |
| V01 | WebGPU fallback and viewer reconnect/probe lifecycle | seven tests in `browser_viewer.py` | **Partial:** the viewer is stubbed; GPU rendering was never observed |
| V02 | Viewer bundle / disabled backend capability | `browser_viewer_contract.py` (real bundle, real DuckDB), failure/recovery tests | **Partial:** same GPU gap; tables, search and cross-filters work against the real bundle |
| L01 | 30 simultaneous participants, 1,000-opinion corpus | `classroom_load.py`; no lost or duplicate opinions. 2026-09-09 timings: ack p95 10.5/6.3 ms, max broadcast 1468/517 ms, heartbeat p95 30.4 ms | **Partial:** fake embeddings; see O02 for real-model latency |
| L02 | PCA-to-UMAP threshold transition | `classroom_load.py::test_threshold_transition_from_79_to_80_keeps_all_points`; real embeddings project cleanly at n=79/80 (see O02) | **Pass** |
| P01 | Hostile participant HTTP/SQL/websocket requests | `test_mosaic_safety.py`, `test_viewer_pipeline.py`, `test_security_routes.py`, `test_payload_privacy.py`, `browser_security.py` | **Pass** (functional QA only, not a penetration test or dependency audit) |
| P02 | Spreadsheet / HTML-shaped opinions | `test_classroom_validation.py::test_spreadsheet_export_treats_opinions_as_text`, `browser_viewer::test_viewer_failure_displays_error_text_without_executing_markup` | **Pass** |
| O01 | Startup, health timeout, roster reload, process recovery | `test_class_launcher.py`, `test_roster_updates.py`, `browser_viewer::test_sighup_refreshes_open_viewer_and_queries_the_renamed_target` | **Partial:** launcher logic is tested; the systemd units, IP certificate and renewal timer were not run |
| O02 | Intended real embedder and installed dependencies | Real pass with the app's own `SentenceTransformerEmbedder` and `Layout` using `e5-small-ko` (below) | **Partial:** the default `embeddinggemma` is gated and was **not** loaded |
| F01 | Build, syntax, lint, artifact/process cleanup | `node --check`, `compileall`, `npm run build` all pass; `web/vendor` matches a clean build; working tree contains only the intended changes. No lint or type-check is configured | **Pass** (no linter exists to run) |

**O02 detail (`dragonkue/multilingual-e5-small-ko-v2`, 384-d, CPU, offline after
download, on the 500-opinion demo corpus):** model load 10.9 s; single-opinion encode
median 7.7 ms, max 9.1 ms; 500 opinions in 1.1 s; all vectors finite and unit-norm.
Same-project mean cosine 0.518 versus cross-project 0.476; the 10 nearest neighbours
share a project 62% of the time in the full space (chance is about 20%), and 48% in
the 2-D layout at n=80 and 64% at n=500. A paraphrase scored 0.805 cosine against 0.365
for an unrelated sentence. **Not measured:** EmbeddingGemma latency or Korean quality,
GPU or MPS behaviour, and any host other than this laptop.

## Not covered by any row above

- **EmbeddingGemma.** Accept the licence, log in as that same account, and run the
  O02 pass again with `ATLAS_EMBEDDING_MODEL=embeddinggemma` before relying on the
  default.
- **WebGPU rendering of the Embedding Atlas viewer**, and Safari, Firefox and real
  phones. Touch reading was verified with Chromium touch emulation only.
- **The production deployment:** certificate on `143.248.249.7`, systemd units, renewal
  timer, classroom Wi-Fi, and 30 real devices. A dry run on the real host is the
  remaining gate.
- **Dependency advisory scan and penetration testing.**
- **Keyboard-only reading of an opinion.** Map points are not focusable.

## Defect history

14 defects were found on 2026-09-09 (F01–F14). F01–F03 and F06–F14 were fixed that
day; F04 (touch-only users could not read an opinion) and F05 (long tooltips
overflowed the phone map) were excluded from that pass and were **fixed on
2026-10-12**: a tapped opinion is pinned with a close button, bounded to the map and
scrollable, and a hovered one is bounded to the map. Regression tests are
`browser_adversarial::test_mobile_tap_opens_opinion_text`,
`test_tapped_opinion_can_be_dismissed`, `test_long_opinion_tooltip_is_readable_within_map`
and `test_tapped_long_opinion_is_scrollable_inside_map`. See
[INTERFACE_QA.md](INTERFACE_QA.md) and [DESKTOP_QA_REPAIR.md](DESKTOP_QA_REPAIR.md).
