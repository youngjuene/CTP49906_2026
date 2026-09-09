# Feedback Atlas Classroom Workflow Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work sequentially unless the user requests delegation. Read the design before execution.

**Goal:** Accept complete feedback pastes, automatically produce meaningful map units, and make the classroom interaction recoverable and unambiguous.

**Architecture:** Preserve the current FastAPI/WebSocket/SQLite application and its embedding/projection pipeline. Add durable parent submissions, a semantic-boundary adapter, a single shared model executor, and versioned publication of derived opinions. Connect source labels, drafts, corrections, filters and export to those records.

**Tech Stack:** Existing Python 3.12+, FastAPI, SQLite, sentence-transformers/EmbeddingGemma, NumPy/UMAP, vanilla JavaScript, IndexedDB, Vite and optional Embedding Atlas. Evaluate and pin Chonkie only after compatibility and quality checks.

**Spec:** [2026-09-09-feedback-atlas-classroom-design.md](../specs/2026-09-09-feedback-atlas-classroom-design.md).

## Global constraints

- `AI/사람` describes the origin of the feedback, not the identity of the person submitting it.
- AI remains the intentional default. Processing human-authored text with AI tools does not change its source.
- The participant does not have to pre-split text, approve a preview before every send, or repeatedly submit separate topics.
- Raw text is immutable. All offsets are half-open Unicode code-point positions in that exact string.
- One submission concerns one target and one week. Source can be corrected per unit.
- Public map payloads must not contain submitter IDs/names, capabilities, pending raw text, private diagnostics, or revision history.
- Filter changes never recompute coordinates.
- One executor owns blocking model work; segmentation and projection must not call the model concurrently.
- Initial bounds: 20,000 raw code points, 256 KiB incoming frames, 64 automatic units, 300 queued submissions, 32-window inference batches, up to four parents per wave, three transient attempts total with 1 and 5 second retry delays.
- Model windows: at most 1,792 content tokens and at most 2,048 total prepared tokens including prompts and special tokens.
- Keep existing dependency pins and Python >= 3.12. Any added dependency must be tested against the installed environment and exactly pinned when accepted.
- No product deployment, changes to real class data, automatic historical resegmentation, or external messaging are authorized by this planning request.

## Execution location and baseline

All code paths below are relative to `/mnt/hdd/research/2026/CTP49906_2026/feedback-atlas` on `ssh june`. This plan itself is local to the Mac because planning does not require mutating the active remote working tree.

Reviewed HEAD: `bcd3bb9`, with existing tracked and untracked modifications. In particular, `frontend/`, `web/viewer.js`, `web/vendor/`, and classroom tests are not all present in HEAD. Recheck before execution; do not assume the tree stayed unchanged or discard its work. There were no applicable AGENTS.md files in the inspected ancestor path; recheck at execution time.

Create an isolated execution checkout from a snapshot of the **current working tree**, including reviewed untracked source and excluding secrets, live SQLite data, virtual environments and dependency caches. Use the worktree skill at execution time. Do not stash/reset the original tree. Record `git diff --binary`, the untracked-source manifest and checksums in the execution notes. Commit only scoped implementation paths after review; never use `git add .` against the original dirty checkout.

Run server/browser tests against temporary rosters and databases. The existing opt-in load test is explicitly fake-model based and is not a semantic-quality test.

## Release sequence

| Increment | Tasks | Independently reviewable result |
| --- | --- | --- |
| A. Establish boundaries and contracts | 0–2 | Measured segmentation candidate and safe schema migration; current classroom still works |
| B. Complete the paste-to-map path | 3–5 | One durable paste produces linked units, with status and reload recovery |
| C. Finish classroom interaction | 6–9 | Correct source/target/week handling, corrections, touch reading and predictable export |
| D. Integrate optional analysis and operations | 10–11 | Analysis behaves consistently; import/archive/launcher understand new records |
| E. Validate and release | 12 | Evidence against semantic, recovery, load and actual-device gates |

Task 1's quality result is required before semantic splitting is enabled for real classes. Tasks 2, 6, 8 and the existing viewer defects can still be investigated if that evaluation fails. Do not quietly substitute sentence splitting and call the requirement complete.

## File boundaries and contracts

Keep existing modules where they already own a responsibility. Avoid making `src/server.py` or `web/app.js` contain the segmentation algorithm or the durable browser queue.

| New file | Responsibility |
| --- | --- |
| `src/submissions.py` | Typed intake, status, unit-span and revision contracts |
| `src/segmentation.py` | Structure candidates, semantic adapter, exact span validation |
| `src/model_runtime.py` | Serialized executor, task-specific embeddings and long-unit pooling |
| `src/submission_worker.py` | Durable queue, staged revisions, retries, publication coordination |
| `src/class_context.py` | Current week/target, context revision, submission-open state |
| `src/exporting.py` | Published-view scope validation and CSV rows |
| `web/draft-store.js` | IndexedDB draft/outbox/receipt persistence and tab leases |
| `web/submission-status.js` | Saved/processing/result cards and private status recovery |
| `web/feedback-reader.js` | Touch/keyboard unit reader, original context and correction controls |
| `web/query-state.js` | Canonical filters, emphasis, selected IDs and export scope |
| `tests/fixtures/segmentation_cases.jsonl` | Labeled calibration and held-out feedback cases |
| `scripts/evaluate_segmentation.py` | Reproducible real-model quality/latency evaluation |

Proposed shared types in `src/submissions.py`:

```python
from dataclasses import dataclass
from typing import Literal

Source = Literal["ai", "human"]
Status = Literal["queued", "splitting", "embedding", "ready", "failed", "withdrawn"]

@dataclass(frozen=True)
class FeedbackSpan:
    start: int
    end: int

@dataclass(frozen=True)
class SubmissionRequest:
    reviewer_id: str  # assigned from connection, never copied from client body
    target_id: str
    week: int
    source: Source
    raw_text: str
    nonce: str
    owner_capability: str
    context_revision: int
    metadata_confirmed: bool = False

@dataclass(frozen=True)
class SubmissionReceipt:
    submission_id: str
    nonce: str
    revision: int
    state: Status

@dataclass(frozen=True)
class SplitResult:
    spans: tuple[FeedbackSpan, ...]
    algorithm_key: str
```

Key contracts, defined by the tasks below:

```python
# src/segmentation.py
validate_spans(raw: str, spans: tuple[FeedbackSpan, ...]) -> None
split_feedback(raw: str, runtime, policy) -> SplitResult

# src/model_runtime.py
ModelRuntime.run(fn, *args)  # async; executes synchronous callable on one executor
ModelRuntime.encode_similarity(texts: list[str])  # worker-thread-only vectors
ModelRuntime.encode_units(texts: list[str])       # worker-thread-only map vectors

# src/store.py; methods on Store
accept_submission(request: SubmissionRequest) -> SubmissionReceipt
read_receipt(submission_id: str, reviewer_id: str, capability: str) -> SubmissionReceipt
stage_revision(submission_id: str, expected_revision: int,
               result: SplitResult, sources: tuple[Source, ...]) -> tuple[str, ...]
commit_publication(expected_data_rev: int, parent_revisions: dict[str, int],
                   coords: dict[str, tuple[float, float]], layout_rev: int) -> int
withdraw_submission(submission_id: str, expected_revision: int,
                    action_nonce: str, reviewer_id: str, capability: str) -> int

# src/class_context.py
ClassContext(week: int, target_id: str | None, revision: int, accepting: bool)
```

Here `runtime` is the shared `ModelRuntime`; `policy` is the exact immutable configuration selected and recorded in Task 1. `algorithm_key` fingerprints that configuration, the library and the model/prompt. `stage_revision` returns deterministic unit IDs without publishing them. `commit_publication` returns the new durable data revision, checking both the expected dataset revision and each requested parent revision. Unauthorized, conflicting and invalid operations produce typed exceptions mapped to private protocol errors in Task 3; they do not return partial success.

### Task 0: Capture the baseline and run existing relevant checks

**Files:** Read `requirements.txt`, `pytest.ini`, `CLASSROOM_QA.md`, `README.md`, `tests/conftest.py`; create execution notes in `docs/qa/2026-09-09-baseline.md` in the isolated checkout.

- [ ] Capture current source state as described above; verify the isolated checkout contains the untracked viewer and classroom tests.
- [ ] Check Python and installed versions, then run existing focused processing and browser regressions:

```bash
.venv/bin/python --version
.venv/bin/python -m pytest tests/test_submission_validation.py tests/test_submission_retry.py tests/test_store_sqlite.py tests/test_recompute_pipeline.py tests/test_payload_privacy.py tests/test_protocol_messages.py -q
.venv/bin/python -m pytest tests/browser_classroom.py tests/browser_viewer.py -q
```

- [ ] Record actual passes, failures and skips; identify pre-existing failures instead of hiding them or treating them as introduced regressions. Do not re-run the previous full real-model load experiment for this baseline.
- [ ] Review the source snapshot and baseline notes. This task creates no behavior change.

### Task 1: Evaluate semantic boundaries and shared-model integration

**Files:** Create `src/submissions.py`, `src/segmentation.py`, `src/model_runtime.py`, `config/segmentation.json`, `scripts/evaluate_segmentation.py`, `tests/test_segmentation.py`, `tests/test_model_runtime.py`, `tests/fixtures/segmentation_cases.jsonl`; modify `src/embedder.py`, `src/config.py`, `requirements.txt` only as required by the accepted adapter.

**Consumes:** Existing `SentenceTransformerEmbedder` and clustering cache convention. **Produces:** The span/model contracts above and an evaluated, versioned segmentation policy.

- [ ] Add pure structural tests before adding a chunker dependency. A real example is:

```python
def test_units_partition_the_exact_original():
    from src.submissions import FeedbackSpan
    from src.segmentation import validate_spans
    raw = "조명이 따뜻했습니다. 색 변화도 자연스러웠습니다.\n\n음악이 큽니다. 볼륨을 낮추면 좋겠습니다."
    boundary = raw.index("음악")
    spans = (FeedbackSpan(0, boundary), FeedbackSpan(boundary, len(raw)))
    validate_spans(raw, spans)
    assert "".join(raw[s.start:s.end] for s in spans) == raw

def test_overlap_is_rejected():
    import pytest
    from src.submissions import FeedbackSpan
    from src.segmentation import validate_spans
    with pytest.raises(ValueError):
        validate_spans("abcdef", (FeedbackSpan(0, 4), FeedbackSpan(3, 6)))
```

- [ ] Add gap, empty/whitespace-only unit, Unicode NFD, emoji, repeated substring, heading and unpunctuated-input cases. Run `python -m pytest tests/test_segmentation.py -q` and confirm the new contract fails before implementation.
- [ ] Implement original-offset validation independently of Chonkie. Use monotonically validated offsets, not `raw.find(chunk.text)` from index zero; repeated passages must not be assigned the first occurrence repeatedly.
- [ ] Add one executor around the loaded model. Keep existing `encode(texts)` compatible. Expose separate similarity and clustering encode paths without changing shared model prompts while a call is active. Add tokenizer checks before model encoding; keep model imports lazy.
- [ ] Install the candidate only inside the isolated environment. Verify its actual adapter interface and tokenizer behavior from the installed package. Confirm exactly one model object is loaded; reject any adapter that silently instantiates another model or changes the clustering prompt.
- [ ] Build the 80-case corpus and fixed 40/40 calibration/held-out split described in the design. Include feedback from both declared sources. Store case ID, raw text, language, split, required/optional/forbidden code-point boundaries and single-topic label. Human boundary judgments remain distinguishable from synthetic expectations.
- [ ] Compare structure-only baseline against semantic grouping on calibration data. Start with adjacent context windows of one and two candidates and similarity thresholds 0.60, 0.70, 0.80 and 0.90; these are search candidates, not claimed suitable defaults. Avoid forced two-sentence minimums, overlap and non-consecutive merges. Evaluate the supported similarity prompt against the current clustering prompt without sharing their cached vectors.
- [ ] Handle a too-long coherent unit with internal token windows and normalized token-weighted pooling. Assert each fully prepared input is <= 2,048 tokens. Test a short unit uses the existing short-text path, and a long single-topic case remains one visible unit. Record the long-vector policy in the cache key; do not reuse averaged similarity vectors as clustering vectors.
- [ ] Pin the accepted library version and policy, freeze them, and evaluate held-out cases once. Implement evaluator arguments `--cases`, `--split`, `--output` and `--policy`; write JSON metrics and per-case spans/latency to the requested output path.

```bash
.venv/bin/python -m pytest tests/test_segmentation.py tests/test_model_runtime.py tests/test_embedder_contract.py -q
.venv/bin/python scripts/evaluate_segmentation.py --cases tests/fixtures/segmentation_cases.jsonl --split heldout --policy config/segmentation.json --output /tmp/atlas-segmentation-heldout.json
```

- [ ] Save the chosen immutable policy as `config/segmentation.json` and the result analysis in `docs/qa/segmentation-evaluation.md`. If the quality gate fails, retain the result, document concrete failure categories, and do not enable automatic splitting in classroom configuration. Review and commit the scoped evaluator/adapter changes.

### Task 2: Add parent records, revisions and a safe v1 migration

**Files:** Modify `src/store.py`, `src/models.py`, `src/textnorm.py`, `tests/test_store_sqlite.py`; create `tests/test_submission_store.py`, `tests/test_store_migration_v2.py`.

**Consumes:** `SubmissionRequest`, `FeedbackSpan`, `SplitResult`. **Produces:** Durable parents, staged/active membership and monotonic data revisions; preserves current opinion IDs.

- [ ] Build a v1 fixture with two opinions, cached vectors, coordinates and nonce receipts. Add tests that migration preserves those values, can be run twice, recovers from an injected rollback and refuses a newer schema without writing it.
- [ ] Before migration, use SQLite's backup API to a sibling backup path; do not copy only the main database file while WAL writes may exist. Enable foreign keys and perform versioned DDL/backfill in one transaction. Check schema version before modifying tables; do not rely on the current `executescript` followed by a late version mismatch.
- [ ] Add `submissions`, `submission_revisions`, `submission_units` and `submission_actions` tables. Keep legacy `submission_receipts` readable for migrated replay links. Essential constraints:

```sql
CREATE UNIQUE INDEX submissions_replay
    ON submissions(reviewer_id, nonce);
CREATE UNIQUE INDEX unit_revision_order
    ON submission_units(submission_id, revision, ordinal);
CREATE UNIQUE INDEX unit_revision_span
    ON submission_units(submission_id, revision, start_cp, end_cp);
CREATE UNIQUE INDEX correction_replay
    ON submission_actions(submission_id, action_nonce);
```

The tables include the fields in the design; enforce source and state enums, positive revisions, valid offsets and parent foreign keys. Null legacy nonces are permitted; new v2 nonces and capabilities are mandatory. Store capability digests, never plaintext capabilities.

- [ ] Backfill one `legacy_normalized` parent and published revision for every existing opinion. Retain its opinion ID and map vector cache. Recover existing nonce-to-opinion receipts as nonce-to-parent mappings; mark legacy corrections admin-only.
- [ ] Change `all_opinions()` to return only published, non-withdrawn units. Keep compatibility insertion methods used by existing scripts/tests working by creating a one-unit legacy parent transactionally. Add an explicit staged-opinion query for recomputation; staged units must not leak through ordinary snapshots.
- [ ] Implement `accept_submission`, `stage_revision` and `commit_publication`; use full UUIDs or deterministic UUID5 IDs derived from parent/revision/ordinal, plus database uniqueness. A committed stage can be retried without duplicating children. Commit publication with compare-and-swap checks and coordinate cleanup; count-changing and metadata-only publications both increment stored `data_rev`.
- [ ] Verify duplicate content from different people remains distinct submissions, but a repeated nonce from one person returns the same parent. Reject a changed body or capability under an existing nonce.

```bash
.venv/bin/python -m pytest tests/test_store_sqlite.py tests/test_submission_store.py tests/test_store_migration_v2.py tests/test_submission_retry.py -q
```

- [ ] Review the migration diff and before/after fixture counts, then commit this storage increment. Do not run migration against the real class database during development.

### Task 3: Accept long submissions durably through protocol v2

**Files:** Modify `src/protocol.py`, `src/server.py`, `src/hub.py`, `src/config.py`, `src/payloads.py`, `.env.example`, `tests/test_submission_retry.py`, `tests/test_protocol_messages.py`, `tests/test_wire_safety.py`, `tests/test_payload_privacy.py`; create `tests/test_submission_intake_v2.py`.

**Consumes:** Parent storage and existing participant/admin channels. **Produces:** V2 acceptance, private receipt/status access, advertised limits and explicit errors.

- [ ] Add server tests for 1,001-character and 20,000-code-point Korean/emoji input, 20,001 code points, frame boundary overflow, invalid target/source/week, missing nonce/capability, closed intake, and queue full. Measure raw code points before normalization. Never coerce a non-string body into an accepted text string.
- [ ] Advertise `dataset_id`, limits, context revision, and v2 capabilities in `hello_ok`. Retain explicit participant serialization; display submitter identity from the client's entered ID rather than adding full roster identity to broadcasts.
- [ ] Replace the submit handler's direct Opinion creation with validation and `Store.accept_submission`. The accepted event is private and has this contract:

```json
{"t":"submission_accepted","nonce":"client-random-id","submission_id":"s_uuid","revision":1,"state":"queued"}
```

The incoming envelope is `{t:"submit", text, target_id, week, source, nonce, owner_capability, context_revision, metadata_confirmed}`. Canonical `reviewer_id` comes only from the connection. Do not return the capability in broadcast payloads.

- [ ] Add capability-checked `submission_status` and `submission_retry` requests. Return an allowlisted original/result only to its owner/admin while processing. Accepted-nonce lookup happens before rate/queue/session-close refusal, but only after verifying the capability and payload fingerprint. `NONCE_CONFLICT` must not mutate the existing submission.
- [ ] Map errors to concrete Korean recovery messages: invalid input stays editable; stale context prompts metadata confirmation; queue full retains the outbox; failed processing offers retry; protocol mismatch offers refresh after draft preservation. Never include raw exception traces in participant replies.
- [ ] Lift protocol/config/HTTP intake bounds coherently to the design values. A long paste consumes one existing intake rate-limit token, not one token per derived unit. Apply correction/retry limits to expensive new work while accepted receipt replay remains cheap. Use connection-private status routing or request/subscription tokens; do not broadcast a person's processing status to every participant.
- [ ] Verify byte-for-byte raw storage, durable-before-ack behavior and no sensitive fields in snapshots, deltas, public original reads or Mosaic participant queries.

```bash
.venv/bin/python -m pytest tests/test_submission_intake_v2.py tests/test_submission_retry.py tests/test_protocol_messages.py tests/test_wire_safety.py tests/test_payload_privacy.py tests/test_mosaic_safety.py -q
```

- [ ] Review the wire examples and backward-client refresh path, then commit the intake increment.

### Task 4: Process queued parents and publish complete revisions

**Files:** Create `src/submission_worker.py`, `tests/test_submission_worker.py`; modify `src/atlas_state.py`, `src/server.py`, `src/model_runtime.py`, `src/store.py`, `src/deltas.py`, `src/mosaic_db.py`, `tests/test_recompute_pipeline.py`, `tests/test_viewer_pipeline.py`.

**Consumes:** Durable parents, semantic policy and shared executor. **Produces:** Recoverable processing, staged embeddings and coherent public revisions.

- [ ] Add failure-injection tests at acceptance, post-segmentation, post-child insertion, pre-publication and post-publication/pre-notification. Assert one parent, deterministic children, original coverage, and either the old published set or the complete new set after restart.
- [ ] Implement FIFO persisted jobs and per-stage state transitions. Recover `splitting`/`embedding` jobs left by restart. Persist successful split manifests before later work. Back off only transient failures with the specified attempts; bad manifests/context overflow are explicit failures, not silently converted to sentence chunks.
- [ ] Route both `RecomputeLoop` and segmentation through `ModelRuntime.run`; remove independent `asyncio.to_thread` paths that could overlap model calls. Do not cancel a timed-out thread and start another model call while the first is still running. Expose queue age and stage failure to instructor health state.
- [ ] Batch windows and process up to four parents per wave. Embed only missing texts belonging to the immutable recompute input; the current global `missing_embeddings()` scan must not accidentally include unrelated staged work that arrived mid-pass. Reuse the current short-text cache and separate task/pooling keys.
- [ ] Build layouts from current published units plus the next staged revisions, excluding replaced/withdrawn units. Only `commit_publication` makes children public. Recompute failure leaves public points and published revision unchanged.
- [ ] Update SQLite publication, in-memory opinions/coords/vectors, neighbors and Mosaic before notifying clients. Broadcast full snapshots on replacement/removal and add/move deltas on ordinary additions. Persist `data_rev` independently of row count; empty and metadata-only changes must not hit the current coordinate-only early return.
- [ ] Guard worker results against withdrawn parents, changed requested revisions and stale corpus generation. Discard/requeue stale publication work; caches may remain reusable. Apply new publication waves between recomputes to avoid an intake stream repeatedly invalidating its own work.
- [ ] Verify the model never runs concurrently using a test encoder with an active-call counter; assert its maximum is one while segmentation and projection are both requested. Verify heartbeat responsiveness during queued work.

```bash
.venv/bin/python -m pytest tests/test_submission_worker.py tests/test_recompute_pipeline.py tests/test_delta_math.py tests/test_viewer_pipeline.py tests/test_model_runtime.py -q
```

- [ ] Review a captured parent lifecycle and crash recovery trace, then commit the processing increment.

### Task 5: Deliver one-send composition, durable drafts and result status

**Files:** Create `web/draft-store.js`, `web/submission-status.js`, `tests/browser_submission_flow.py`; modify `web/compose.js`, `web/app.js`, `web/index.html`, `web/app.css`, `tests/browser_classroom.py`.

**Consumes:** V2 advertised limits, dataset ID and private receipt/status requests. **Produces:** One-send long input with reload-safe recovery and optional result review.

- [ ] Add browser cases for exact raw/metadata restoration, reload after Send before ack, ack arriving after new typing, duplicate tabs replaying one outbox item, storage failure, and switching IDs. Use existing isolated server fixtures; do not replace network failures with successful mock responses.
- [ ] Implement IndexedDB stores `drafts`, `outbox`, `receipts` and `leases`. Key drafts by dataset ID plus normalized entered ID; use random per-tab lease owner and expiry for outbox delivery. Persist immutable envelope/nonce/capability before first transmission. Capabilities use `crypto.getRandomValues` with 32 random bytes, encoded base64url.
- [ ] Save input after 250 ms and complete an explicit local transaction before Send. Do not use textarea maxlength to truncate a paste. Count with `Array.from(text).length`; disable Send with a visible explanation above the advertised bound. Recover the exact input, including CR/LF, Korean decomposition and emoji.
- [ ] On ack, store the receipt and remove only that outbox item. Clear the compose text only if it is still the submitted draft version. Never clear a newer draft or reset target/week/source merely because a prior submission completed.
- [ ] Render source terms and helper copy from the design, a target/week/source summary, connection/local-save/server-save states, processing stages and `확인·수정` result action. Keep a user's deliberate source choice; source inference is not part of this UI.
- [ ] Add explicit local-data clearing and draft text download. If storage is unavailable, keep text in the form and explain the limitation rather than saying it was saved. Show a refresh action for old protocol only after persisting or downloading the current draft.

Representative browser assertion, using the existing `live_server`/`context` fixtures and the existing `#c-*` form IDs:

```python
def test_reload_restores_raw_draft_and_metadata(live_server, context):
    from browser_classroom import enter
    page = context.new_page()
    enter(page, live_server.url)
    raw = "조명 설명입니다.\n\n음악에 대한 긴 의견입니다. 🎵"
    page.fill("#c-text", raw)
    page.select_option("#c-target", "target2")
    page.select_option("#c-week", "4")
    page.click('#c-source button[data-value="human"]')
    page.get_by_text("이 기기에 저장됨", exact=True).wait_for()
    page.reload()
    page.wait_for_function("document.querySelector('#c-text').value.length > 0")
    assert page.input_value("#c-text") == raw
    assert page.input_value("#c-target") == "target2"
    assert page.input_value("#c-week") == "4"
    assert page.get_attribute('#c-source button[data-value="human"]', "aria-pressed") == "true"
```

The reviewed source uses select elements for target/week and pressed buttons for source; preserve those selectors while changing labels.

```bash
.venv/bin/python -m pytest tests/browser_submission_flow.py tests/browser_classroom.py -q
```

- [ ] Manually paste a complete multi-topic response once, observe saved/processing/result states, reload and inspect its original. Review and commit the browser increment.

### Task 6: Make joining, presenter changes and session close explicit

**Files:** Create `src/class_context.py`, `tests/test_class_context.py`; modify `src/server.py`, `src/protocol.py`, `src/store.py`, `web/app.js`, `web/admin.js`, `web/compose.js`, `web/index.html`, `tests/browser_classroom.py`.

**Consumes:** Dataset metadata, private/admin authorization, durable drafts. **Produces:** Persisted classroom context and predictable stale-draft handling.

- [ ] Add tests for empty-form following, dirty-form freezing, explicit target override, two admin tabs racing a context update, closed intake and replay of an already accepted nonce after close.
- [ ] Store `ClassContext` in dataset metadata and broadcast a public context frame on admin updates. Validate target against student targets, week against existing weeks, and update context with expected revision. Never use context updates to mutate stored feedback metadata.
- [ ] Add an instructor current-week/target selector and open/close intake control. Show pending/failed counts. When no active target is set, compose requires a deliberate target instead of silently selecting the first row.
- [ ] Dirty drafts retain target/week. A banner lets the participant update to the current context or retain their original metadata explicitly. Send carries the observed context revision and confirmation bit. Stale-context errors retain the same draft; the participant updates the unsent envelope and assigns a new nonce only after confirming changed content, while checking whether any previous envelope was accepted.
- [ ] Show connection/retry errors on the landing page itself, allow another attempt, display the entered submitter ID after join, and implement change-ID without another person's visible draft surviving the switch.
- [ ] Test close/reopen and next-week transition with an offline pending envelope. The app must not silently submit it under new defaults.

```bash
.venv/bin/python -m pytest tests/test_class_context.py tests/test_submission_intake_v2.py tests/browser_classroom.py -q
```

- [ ] Review the empty/dirty/override scenario matrix, then commit the context increment.

### Task 7: Add optional inspection, source correction, merge/split and withdrawal

**Files:** Create `web/feedback-reader.js`, `tests/test_submission_corrections.py`, `tests/browser_submission_corrections.py`; modify `src/submissions.py`, `src/store.py`, `src/server.py`, `src/protocol.py`, `src/payloads.py`, `src/atlas_state.py`, `web/submission-status.js`.

**Consumes:** Owner capabilities, published/staged revisions and exact code-point spans. **Produces:** Idempotent corrections with history and safe map replacement.

- [ ] Add tests for wrong capability, another roster ID, guessed public parent ID, stale revision, replayed action nonce, source-only changes, parent target/week changes, adjacent merge, split inside emoji/NFD text, withdrawal during model work and restart after correction.
- [ ] Define a correction message with `submission_id`, `owner_capability`, `expected_revision`, `action_nonce` and a full ordered manifest `{start,end,source}` plus parent target/week. Derive unit text from the immutable original server-side. Validate the complete partition and metadata; never accept client-generated replacement quotations.
- [ ] Metadata-only corrections reuse vectors/coordinates. Boundary edits stage a new revision and run only missing clustering vectors. Store prior manifests; activate replacement atomically. Replayed action nonce returns its previous result; conflicting revision returns the latest state for review without overwriting it.
- [ ] Permit adjacent merges and a split at a user-selected valid code-point boundary. Keep headings/context visible during editing. A mixed AI/human paste is corrected through unit source choices; the parent retains its initial source declaration in history.
- [ ] Add private owner/admin original-and-history reads and a separate public read for currently published original/spans. Public reads use an allowlist and become unavailable on withdrawal. Raw content is rendered as text, never interpreted HTML.
- [ ] Implement whole-parent withdrawal with immediate published removal, a new data revision and full snapshot. Remove obsolete selected IDs, neighbors and Mosaic rows. Ensure a delayed worker cannot republish withdrawn units.

```bash
.venv/bin/python -m pytest tests/test_submission_corrections.py tests/test_payload_privacy.py tests/test_mosaic_safety.py tests/browser_submission_corrections.py -q
```

- [ ] Review correction history and both public/private payloads, then commit this increment.

### Task 8: Make the classroom map readable and its counts interpretable

**Files:** Modify `web/atlas.js`, `web/app.js`, `web/feedback-reader.js`, `web/index.html`, `web/app.css`, `src/payloads.py`, `src/protocol.py`, `tests/browser_classroom.py`; create `tests/test_feedback_counts.py`.

**Consumes:** Published units, opaque parent IDs, source metadata and the reader. **Produces:** Touch/keyboard reading and separate unit/submission counts.

- [ ] Add browser tests that a participant can open a point using tap and keyboard, read its source and original context, close the reader with Escape and return focus. Keep admin selection separate from ordinary participant reading.
- [ ] Wire pointer-up without drag to open the reader for all participants; hover remains optional. Expose a keyboard-accessible results list tied to the current visible set rather than trying to focus every canvas pixel.
- [ ] Keep target-color and source-shape legends with the corrected source terms. Make the classroom map the default irrespective of screen width. An analysis view requires an explicit toggle.
- [ ] Display `feedback units` and `original submissions` independently. Count unique parent IDs in the active visible set; use server-provided admin aggregates for distinct submitters. Do not infer participant count from units or expose reviewer IDs to calculate it in the participant browser.
- [ ] Add original-to-unit linkage in the reader. Add concise copy that proximity represents text similarity and that many units from one paste are not independent people. Announce the existing PCA-to-UMAP layout change without adding a new projection mode.

```python
def test_one_paste_is_not_three_submissions():
    from src.submissions import visible_counts
    points = [
        {"id": "u1", "submission_id": "s1"},
        {"id": "u2", "submission_id": "s1"},
        {"id": "u3", "submission_id": "s1"},
    ]
    assert visible_counts(points) == {"units": 3, "submissions": 1}
```

Define `visible_counts(points: list[dict]) -> dict[str,int]` as the corresponding pure server aggregate helper and implement the browser's identical count semantics against explicit fixture rows.

```bash
.venv/bin/python -m pytest tests/test_feedback_counts.py tests/browser_classroom.py -q
```

- [ ] Review on Mac and a touch device, then commit the map increment. Record physical-device checks separately from viewport emulation.

### Task 9: Unify filter membership, selection and export scope

**Files:** Create `web/query-state.js`, `src/exporting.py`, `tests/test_export_scopes.py`, `tests/browser_export_scopes.py`; modify `web/app.js`, `web/admin.js`, `src/server.py`, `src/mosaic_db.py`.

**Consumes:** Published snapshot/data revision and stable unit IDs. **Produces:** One definition of visible/selected results and predictable CSV exports.

- [ ] Add fixtures with two weeks, two targets, both sources, mixed-source parents and a replaced unit ID. Test empty selected scope, visible scope after week/search filtering, all scope, and concurrent publication during export preparation.
- [ ] Implement immutable query state `{weeks, targetIds, sources, search, selectedIds, emphasis}`. Define actual membership separately from emphasis; current search behavior that merely dims points must be explicitly converted to a filter for `visible results` or labeled as emphasis. Compute a stable visible ID set from the current published snapshot.
- [ ] Implement `resolve_export_ids(scope, visible_ids, selected_ids, all_ids)` in `src/exporting.py`. Visible/selected requests carry frozen IDs and `expected_data_rev`; all-scope requests also carry the revision shown beside the export action. Validate IDs against the published view and return conflict on a changed revision. Empty selection never expands to all.

```python
def test_empty_selection_does_not_export_everything():
    from src.exporting import resolve_export_ids
    assert resolve_export_ids("selected", ["u1"], [], ["u1", "u2"]) == []
```

- [ ] Show scope and row count immediately beside Export. Default to visible results. A stale revision updates the preview and requires a new click, not a silent export with a different count.
- [ ] Extend unit CSV with submission ID, published revision, ordinal, span offsets and split provenance. Add one-original-per-row export. Preserve source versus submitter separation, authorization and `_csv_cell` behavior. Export only active published units; private historical revisions are not mixed into ordinary results.

```bash
.venv/bin/python -m pytest tests/test_export_scopes.py tests/test_server_routes.py tests/browser_export_scopes.py -q
```

- [ ] Compare exported IDs/counts to the visible and selected map sets, then commit the export increment.

### Task 10: Repair and integrate the optional analysis viewer

**Files:** Modify `frontend/vite.config.js`, `frontend/src/bundle.js`, `web/viewer.js`, `web/app.js`, `web/query-state.js`, `src/mosaic_db.py`, `tests/browser_viewer.py`; rebuild `web/vendor/` from the pinned lockfile.

**Consumes:** Canonical query/selection state, published data revision and existing vendored viewer. **Produces:** Opt-in analysis whose controls act on its actual data.

- [ ] Add browser coverage using the real bundle for worker requests and live additions; a mocked viewer is insufficient to reproduce the observed worker-path and stale-count defects.
- [ ] Fix asset resolution at build time. Evaluate `base: "./"` for the existing Vite library build and assert that the emitted worker is requested under `/vendor/` with JavaScript MIME type. Do not hand-edit generated hashes or add a hard-coded route for one worker filename.

```javascript
// frontend/vite.config.js: apply within the existing defineConfig object
base: "./",
```

- [ ] Rebuild and inspect network requests. Verify label-generation work completes or gives a visible recoverable error; include import, worker and WebGPU failure paths. Keep the classroom map usable if analysis fails.
- [ ] Inspect the installed pinned viewer/Mosaic public interfaces before implementing the bridge. Connect global filters, selection IDs and corpus invalidation through supported APIs; do not depend on fabricated `setFilters` or private fields. Counts, query caches and selection must refresh after addition, correction and withdrawal.
- [ ] Assert week/search/target/source membership and selected IDs agree with canonical state. Do not change projection coordinates when filtering. Source must be readable even if the advanced renderer cannot reproduce both shape and color channels; expose explicit source fields and filters.
- [ ] If the pinned API cannot support the contract without substantial upstream replacement, leave advanced analysis disabled in classroom configuration and document that scoped limitation. The main workflow can release with the verified classroom map; an inconsistent analysis toggle cannot be enabled by default.

```bash
npm ci --prefix frontend
npm run build --prefix frontend
.venv/bin/python -m pytest tests/browser_viewer.py tests/test_viewer_pipeline.py tests/test_mosaic_safety.py -q
```

- [ ] Review the real-browser network trace and visible totals after live mutations, then commit source and regenerated assets together.

### Task 11: Align imports, archives and classroom preparation

**Files:** Modify `scripts/seed_ai_opinions.py`, `scripts/deidentify.py`, `scripts/class.sh`, `.env.example`, `README.md`, `README_kr.md`, `CLASSROOM_QA.md`, `tests/test_seed_ai_opinions.py`, `tests/test_deidentify.py`, `tests/test_class_launcher.py`.

**Consumes:** New schema, shared intake/processing service and class context. **Produces:** Consistent operational paths and recoverable class setup/close.

- [ ] Extend import tests for a repeated file, two identical texts in different rows, source preservation and a long AI response. Compute import nonce from input-file digest plus row index, not text alone; do not merge independently submitted duplicate text.
- [ ] Keep existing already-unitized import behavior explicit. Add `--segment` to enqueue each input row as one raw parent through the same processing service. Do not instantiate competing model workers inside a running server. Online imports use an existing/admin-authorized intake route with credentials in bodies; offline imports enqueue durable parents for the next startup. Update SIGUSR1 reload to wake that queue and read persisted data revision, not reset `rev` to row count.
- [ ] Extend de-identification to parent and revision actor metadata, removing capabilities and identifying submission associations while preserving opaque parent/unit links and AI/human source. Verify archived copies contain no recoverable owner credentials. Include pending/failed originals in instructor-controlled recovery exports; do not silently omit accepted inputs.
- [ ] Add launcher readiness output for roster count, current context, input bounds, model warmup, queue state and optional viewer state. Document one class per database and a stable URL as preferable for browser draft continuity. Keep full roster editing and multi-class management outside this increment.
- [ ] Document migration backup/restore and a class-close sequence: close intake, drain/check accepted work, export, then stop. Clarify that a new quick-tunnel origin cannot recover the previous origin's local draft automatically.

```bash
.venv/bin/python -m pytest tests/test_seed_ai_opinions.py tests/test_deidentify.py tests/test_class_launcher.py -q
```

- [ ] Review a seeded, exported and de-identified temporary dataset, then commit operations/docs together.

### Task 12: Prove the full scenario and prepare a controlled release

**Files:** Create `tests/classroom_semantic_load.py`, `docs/qa/feedback-atlas-release.md`; modify `CLASSROOM_QA.md` with the scenario results.

**Consumes:** All enabled increments and the frozen segmentation policy. **Produces:** Concrete go/no-go evidence and a reversible release procedure.

- [ ] Run the full automated suite once after integration, plus the explicit browser modules omitted by default discovery. Investigate skips in required components rather than treating them as passes.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/browser_classroom.py tests/browser_submission_flow.py tests/browser_submission_corrections.py tests/browser_export_scopes.py tests/browser_viewer.py -q
```

- [ ] Add an opt-in real-model load module using an isolated server, roster and database. Make it refuse fake-embedder configuration for quality/performance claims. Seed 1,000 units, submit 30 representative long originals concurrently, then a maximum-length burst. Track parents, units, source/target/week, original-span coverage, queue stages, acknowledgements, publication latency, heartbeat, snapshot bytes and finite normalized vectors/coordinates. Fetch full originals on demand; never repeat a parent's raw text in each child's snapshot payload.

```bash
.venv/bin/python -m pytest tests/classroom_semantic_load.py -s
```

- [ ] Inject disconnect/ack loss/reload/server restart, correction while queued and withdrawal during a recompute. Reopen SQLite and verify accepted-parent count, active-child membership and nonce/action receipts. Check no public unit at a placeholder origin and no obsolete unit in neighbors/Mosaic/export.
- [ ] Re-run the frozen held-out segmentation evaluation only if its implementation/config changed since Task 1; otherwise cite the existing result. Review human boundary disagreements and the actual failure examples rather than only an aggregate score.
- [ ] Walk through two real people/roles on the Mac and a physical phone via the actual classroom QR URL: human writes feedback -> selects 사람; person pastes ChatGPT response -> selects AI; each sends once -> reviews units -> changes presenter with a draft -> reconnects -> corrects source/boundary -> instructor filters and exports -> closes intake. Do not present these as completed checks until performed.
- [ ] Evaluate all design release gates. If latency fails, tune batching/window policy and retry the affected workload; re-run semantic evaluation if boundaries/vector policy changed. If advanced-viewer checks fail, keep it disabled and state the limitation. Do not lower quality metrics after seeing held-out results to manufacture a pass.
- [ ] Prepare release notes, exact commit/build identifiers, migration backup path, required configuration and restore commands for the actual environment. Deploy between sessions: stop new intake, resolve v1 in-flight submissions, let participants copy any unsent v1 draft before refresh, take a consistent backup and validate a copy. Old loaded JavaScript cannot retroactively acquire the new outbox behavior. Legacy receipt recovery remains instructor-only until a valid owner capability exists. After migration, verify legacy counts and representative new submission flow before opening intake.
- [ ] Rollback procedure: close intake; preserve the upgraded database and export all post-upgrade accepted originals with source/target/week; restore matching old code/database only after accounting for those submissions. Never restore a pre-upgrade backup over accepted new feedback without preserving it for recovery.
- [ ] Review the concrete release evidence with the user. Implementation and deployment are subsequent actions; this planning turn does not execute them.

## Plan self-review and requirement coverage

| Review finding or clarified requirement | Implementation task(s) |
| --- | --- |
| Whole paste -> coherent multi-sentence units; EmbeddingGemma capability | 1, 4 |
| Exact original and source survive processing | 1–4, 7 |
| AI source versus human submitter terminology | 5, 7–9 |
| 1,000-character and 8 KB limits | 3, 5 |
| Durable saved/processing/failed status and parent retries | 2–5 |
| Reload loses draft/metadata; old ack can affect new typing | 5 |
| Unexplained join failure and change-ID flow | 6 |
| Accidental old presenter/week/source; separate map filters | 5, 6, 9 |
| No correction/withdrawal; revision currently depends on point count | 2, 4, 7 |
| Touch reading and original-text inspection | 7, 8 |
| Units confused with submissions/participants | 8, 9 |
| Viewer workers, live counts and disconnected controls | 9, 10 |
| Export ignores visible scope | 9 |
| Imports/archive bypass new metadata; next-class preparation | 11 |
| Actual long-input and classroom-device validation | 12 |

Dependency decisions deliberately left to measured Task 1 outcomes are recorded as decision gates, not assumed capability. All other first-release behavior is specified above. The design remains a proposal for the user's review; none of the new acceptance results is claimed as achieved.
