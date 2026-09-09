# Feedback Atlas classroom workflow design

Date: 2026-09-09. Status: proposed design accompanying the requested implementation plan. No product implementation or deployment has been performed.

Implementation repository: `ssh june:/mnt/hdd/research/2026/CTP49906_2026/feedback-atlas`.

This document is stored on the local Mac for review. The remote working tree contains substantial uncommitted work, including the viewer and classroom QA. Execution must preserve that working tree as its baseline; starting from remote HEAD alone would omit reviewed behavior.

## Goal and confirmed requirements

A participant pastes a complete feedback response and sends once. The system preserves the submission, separates coherent feedback units by meaning, and publishes those units on a live map. A unit may contain several sentences.

- `AI/사람` describes the origin of the feedback, not the identity of the person submitting it.
- AI remains the intentional default. Processing human-authored text with AI tools does not change its source.
- The submitter, target, week, original submission, and derived units remain distinct concepts.
- The participant does not have to pre-split text, approve a preview before every send, or repeatedly submit separate topics.
- The implementation must cover joining, composing, processing, viewing, presenter transitions, recovery, correction, instructor export, and session close.

Basis: [interaction review](/Users/youngjuene/.codex/visualizations/2026/09/09/01a0838b-3779-7530-a6e3-a5e7e2a61f7c/feedback-atlas-interaction-review.md).

## Approach selection

| Approach | Benefit | Cost or limitation | Decision |
| --- | --- | --- | --- |
| Structure rules alone: paragraphs, bullets, punctuation | Cheap and deterministic | Formatting is not meaning; it fragments explanations and misses unformatted topic changes | Evaluation baseline and explicit degraded processing only |
| Semantic grouping using existing EmbeddingGemma and a small chunking adapter | Reuses installed weights and server; can preserve multi-sentence ideas | Needs Korean evaluation, token accounting, and exact original-span validation | Recommended first implementation |
| Generative boundary selection | Can reason about implicit topic changes and context | Another model/service, inference cost, validation, and possible rewriting | Reconsider only if the recommended approach misses the quality gate |

EmbeddingGemma outputs vectors with a 2K input context. It can provide the similarity signal; a separate algorithm must choose boundaries. Retain the current clustering convention for map vectors and evaluate a separate similarity convention for segmentation. [Google model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)

Evaluate Chonkie's SemanticChunker first. Its documented interface offers similarity grouping and text offsets. Integrate through an adapter around the already loaded model, rather than asking the library to load another copy. Disable overlap and non-consecutive merging for the first release: a unit must remain a contiguous, traceable part of the original. Pin the version actually evaluated rather than choosing an untested version in this plan. [Chonkie documentation](https://docs.chonkie.ai/oss/chunkers/semantic-chunker)

If the library cannot consume the shared model or preserve reliable offsets, keep the same application interface and evaluate a small adjacent-window implementation using the existing embeddings. This is a bounded contingency, not a second processing system to build in parallel.

## Architecture

```text
Browser draft/outbox
    -> durable original submission + receipt in SQLite
    -> private saved/processing status
    -> semantic boundary selection
    -> span validation + staged feedback units
    -> clustering embeddings + existing projection
    -> publish a complete submission revision
    -> shared map + original-text reader + optional correction
```

Keep FastAPI, WebSockets, SQLite, sentence-transformers, the existing model, and the existing projection. Add focused submission and segmentation modules. Do not introduce Redis, Celery, an ORM, a vector database, a hosted generation service, or a frontend framework migration.

One explicit executor owns blocking model work. Both segmentation and map recomputation use it; model calls never overlap. SQLite is the durable work queue. An in-memory wake signal accelerates it but is not its source of truth. Keep connection handling and draft acknowledgement independent of model latency.

### Original and derived records

Keep `Opinion` as the map's feedback-unit record to minimize downstream disruption. Add a parent submission and revision membership alongside existing opinions.

| Record | Required information |
| --- | --- |
| Submission | Opaque ID, canonical submitter, exact raw text, initial source, current target/week, nonce, creation time, requested revision, published revision, processing state, retry metadata, owner-capability hash |
| Unit membership | Submission ID, revision, ordinal, opinion ID, start/end offsets, source, origin of split (`semantic`, `manual`, `legacy`) |
| Submission revision | Immutable manifest of spans and effective sources; actor kind; creation time; algorithm/model/config fingerprint |
| Dataset metadata | Stable dataset ID, schema version, monotonic data revision, class context and its revision |

One submission concerns one target and one week. Do not guess these from its text. Target/week corrections apply to the whole submission. Source defaults from the submission and can be corrected per unit when a paste contains both AI and human writing. The initial declared source remains in the historical record; effective unit source determines map labels and source filters.

Raw text is immutable. Store the received Unicode string without stripping newlines or applying NFC. All offsets are half-open Unicode code-point positions in that exact string. Python slicing is authoritative. Browser code uses `Array.from(raw)` for offset conversion; UTF-16 indices must not be sent as code-point indices. Normalize a separate copy only for embeddings and cache keys.

For automatic segmentation, units partition `[0, len(raw_text))`: ordered, non-overlapping, no omitted characters. Boundary whitespace belongs to an adjacent unit. Store/display the exact span; apply visual whitespace handling separately. Headings, bullets, pronouns, explanations, and recommendations must remain attached where needed for context. Do not silently remove greetings, duplicate-looking opinions, or contradictory statements. Similarity does not establish agreement or quality.

### Boundaries versus model windows

Sentence and paragraph boundaries are candidates, not mandatory output points. Do not set a global minimum of two sentences: a short, independent observation can legitimately be one unit.

Use zero overlapping output units. A topic recurring later in a paste stays in its original position; the map may place related units near each other. Do not reorder or concatenate separated passages into a synthetic quotation.

Model context windows are an internal detail. If one coherent unit is too long for one embedding input, retain one visible unit and embed internal, non-overlapping windows. Use normalized token-weighted mean pooling for that long-unit vector, versioned separately from existing short-text cache keys, and include it in quality evaluation. Do not force extra map points solely to satisfy a model token limit.

Default internal window budget: at most 1,792 content tokens, with an additional assertion that the fully prepared input, including the actual prompt and special tokens, is at most 2,048 tokens. Count with the loaded tokenizer. Never rely on silent model truncation.

### Proposed operating limits

These are first-release defaults to validate on `june`, not measured capacities or inferred user requirements.

| Setting | Initial value and behavior |
| --- | --- |
| Raw input | 20,000 Unicode code points; show the bound before sending |
| Incoming WebSocket frame | 256 KiB, checked before JSON parsing; align any HTTP intake limit |
| Automatic units per submission | 64; if exceeded, retain raw text and show a recoverable processing problem, without silent truncation or forced topic merging |
| Durable pending queue | 300 submissions; refuse new intake before acknowledgement when full; preserve browser outbox |
| Inference batch size | Start with existing 32-window batch; measure memory and latency |
| Submission scheduling | FIFO, up to four submissions per processing wave; publish completed waves before draining the entire queue |
| Automatic transient retries | Three attempts total; retry after 1 and 5 seconds; then show retry action |

Short single-topic comments take a short path. Reuse valid embedding cache entries; do not re-encode the full historical corpus for each submission. Persist completed boundary results so a projection retry does not repeat segmentation. Keep similarity and clustering caches separate, including actual model revision, prompt, dimension, pooling policy, and segmentation-config version where applicable. Retain legacy cached rows on disk, but reuse them only when their recorded model/prompt identity can be verified against the loaded model; otherwise backfill before reopening intake.

## Persistence, lifecycle, and publication

Public states: `queued -> splitting -> embedding -> ready`, plus `failed` and `withdrawn`. The UI may describe `embedding` as `지도에 반영하는 중` because it includes projection and publication.

- Accept raw text and its deduplication receipt in one transaction, then acknowledge `submission_id` and `queued`.
- Deduplicate by canonical submitter and client nonce. A replay returns the same parent. Changed content with a reused nonce returns `NONCE_CONFLICT`, rather than creating or replacing content.
- A random capability generated in the browser before first send authorizes receipt recovery and later changes. Persist it with the outbox before network transmission; store only its hash server-side. Do not use roster ID or public submission ID as correction authority.
- Persist attempts, next-attempt time, revision and staged manifests. On restart recover interrupted jobs. A client refresh requests current status instead of assuming processing is complete.
- Insert a revision's children transactionally. Publish all children of that revision together after embeddings and coordinates exist. Pending children must not appear at `(0,0)` on reconnect.
- A processing failure leaves the original and retry identity intact. Do not label rule-only output as a successful semantic split. If a degraded option is later offered, it must be explicit.
- During a correction, retain the old published revision until the replacement is ready. Switch the active revision atomically. A failed correction keeps the previous map visible with a private failure message.
- Withdrawal removes all active units immediately from visible state; delayed work must never resurrect them.

Replace `rev = len(opinions)` with a persisted, monotonically increasing data revision. Record updates and removals independently of point count. Use full snapshots for correction/withdrawal in the first release; preserve efficient add/move deltas for ordinary additions. Empty-corpus and metadata-only changes still broadcast. Reject stale worker results against the requested submission revision and current corpus generation; coalesce and reschedule without concurrent model calls.

Update active SQLite rows, coordinate membership, in-memory points, neighbor indices, Mosaic relations, and outgoing payloads as one coherent publication sequence. In-flight exports or data queries must observe one published revision. Filter changes never recompute coordinates.

## Interaction design

### Join and compose

- Landing page: ID example, unknown-ID help, visible connection state and retry action.
- Main page: show the ID the participant entered as their submitter identity and an explicit change-ID action. Do not add roster names to participant broadcasts.
- Use `피드백 출처`, `AI 생성`, `사람 작성`, and `제출자` consistently.
- Source helper: `ChatGPT 등 AI가 만든 의견은 ‘AI 생성’, 사람이 직접 작성한 의견은 ‘사람 작성’을 선택하세요.`
- Use `기록할 주차` for submission metadata and `표시할 주차` for map visibility.
- Before Send, show the current target/week/source summary in the compose panel. Replace the hint encouraging users to ignore all defaults.
- Accept typing and pasting. Do not silently truncate paste with a textarea `maxlength`; retain over-limit input locally and explain the limit with Send disabled.

### Draft and outbox

Use IndexedDB for the raw draft and immutable pending envelopes. Key them by stable dataset ID and normalized local ID. Save on input with a 250 ms debounce and synchronously initiate a transaction on Send; do not rely on unload handlers for durability.

Generate nonce and owner capability once, persist them with the exact request, then send. Clear only the acknowledged envelope. A draft typed while the previous request is pending must survive its acknowledgement. Serialize retries across tabs using a short IndexedDB lease; server nonce deduplication remains the final guarantee.

Show `이 기기에 저장됨`, `전송 대기`, `원문 저장됨`, and later processing status distinctly. If browser storage fails, keep the visible text, explain that reload recovery is unavailable, and do not claim local durability. A disconnect never wipes form fields.

Keep receipts and owner capabilities on this browser, scoped to its dataset and ID, until the user clears that local record. Raw acknowledged outbox text can be removed after the receipt is safely stored. Change-ID closes the old connection and clears its visible state before exposing the next identity's draft; provide an explicit option to delete local drafts/receipts on shared devices. A new tunnel origin cannot automatically access the old origin's IndexedDB; explain recovery through the previous browser origin or instructor and provide draft text download.

### Presenter transitions and session close

Add a small instructor control for current week, active target, and whether submissions are open, persisted in existing dataset metadata. One class per database remains the operating model.

An empty, unmodified form follows current context. A dirty draft or explicit target override stays attached to its original context. Show a presenter-change banner; allow either applying the new context or explicitly retaining the old target. Carry `context_revision` in the submit envelope. If context changed since composing and the participant has not confirmed their metadata, return a recoverable stale-context response instead of silently retargeting.

When context has no active target, require an explicit target choice; do not choose the first roster row. Keep AI as the source default and preserve deliberate source choices.

Closing submissions stops new intake but drains accepted work. Replays of already accepted nonces still return their receipts. Offline unsent work stays local and does not silently move to a later week. The admin panel shows pending/failed counts before the instructor stops the service.

### Result reader and correction

After Send, show an unobtrusive card: `의견 N개로 나누어 반영했습니다` with `확인·수정`. Review is optional on the normal path.

Reader shows each unit, its source, and original-text context. Correction supports parent target/week, per-unit source, adjacent-unit merge, an inserted boundary at a valid code-point position, and whole-submission withdrawal. Preserve wording; text rewriting/resubmission editing is outside this release. Source changes at arbitrary positions can use split-then-source-correction.

Use optimistic revision checking and an action nonce for corrections. Return a conflict with the latest revision when another tab has changed it. Merge/split becomes a new revision; metadata-only changes reuse vectors. Keep the last successful map revision on a failed correction.

### Maps and export

The existing classroom map is the default on all screen sizes. Its target-color and AI/human shape encoding stays intact. Add touch and keyboard activation to a persistent feedback reader; hover is an enhancement, not the only reading path.

Use one application query state: selected weeks, search, target/source filters, selected unit IDs, and separate visual emphasis. `Visible results` means actual filter membership; highlight/dimming alone never changes export membership. Selection refers to stable unit IDs, not visual row positions. Never refit the map when a filter changes.

Advanced analysis remains opt-in. Correct emitted worker asset paths and live-count invalidation. Bridge filtering and selection through the pinned viewer's supported interface, verified from its installed source. If full integration is unavailable, keep that view disabled in classroom release rather than expose controls that appear to act on a different map. Do not rewrite the upstream viewer.

Show distinct counts for feedback units and original submissions. Instructor reports may also count distinct submitters. Source-specific counts count units and submissions explicitly; one mixed-source submission can occur in each source category, so those category totals need not add to the unique overall submission count.

CSV scope choices: `현재 필터 결과` (default), `선택한 의견`, `전체`. Show scope and row count directly beside the export action. Freeze IDs and data revision for visible/selected exports; an empty selection never falls back to all. If the data revision changes between preparation and export, refresh the scope/count and ask the user to click Export again.

Add parent ID, unit ordinal, published revision, span offsets, source, target/week, and segmentation provenance to unit exports. Offer a separate originals export with one row per submission so raw text is not repeated in every child row. Retain existing CSV formula escaping and admin authorization.

## Access and compatibility

Keep explicit participant/admin payload allowlists. Public map payloads contain active unit text and provenance identifiers, not submitter IDs/names, capability tokens, pending raw text, processing diagnostics, or revision history. A reader may request the original of a currently published submission using a separate allowlisted public response; it exposes raw text and unit spans, not submitter metadata. This visibility must be stated on the compose page. Withdrawal disables that public lookup.

Private receipt, processing status, and correction responses require the browser capability or existing admin authority. Capabilities travel in message/request bodies, never URLs or public logs. Legacy opinions without an owner capability are correctable by the instructor; knowing their submitter ID does not mint a capability.

Migrate v1 to v2 transactionally after a consistent SQLite backup. Backfill one legacy parent per existing opinion, preserving opinion IDs, text as currently stored, source, timestamps, embedding cache and coordinates. Record `legacy_normalized` because original pre-normalization whitespace cannot be recovered. Preserve old nonce receipts as links to the corresponding parent. Do not resegment historical data automatically.

Bump the protocol to v2 and deploy server/static assets together between sessions. A stale v1 client gets the existing explicit refresh error. Old loaded JavaScript cannot retroactively acquire the new draft store: resolve v1 in-flight submissions and have participants copy unsent text before refreshing. Do not promise recovery of an unsaved v1 draft or support a second permanent single-opinion submit protocol. Legacy receipts support instructor recovery; they do not grant a new browser ownership by roster ID alone. Keep the old code and backup for rollback. Disable new intake and retain the current database plus an export of post-upgrade submissions before any restore, so rollback cannot erase accepted work silently.

Bulk AI import must use the same parent intake and processing path, with deterministic import nonces. Existing rows that are already units remain one legacy unit unless the import explicitly opts into segmentation. De-identification must cover new parent/revision tables and remove owner capabilities, while preserving source and unit linkage.

## Release gates

These are proposed acceptance criteria, not results of the earlier QA.

1. Structural correctness: 100% original-span coverage, zero overlap, no rewritten text, correct source/target/week inheritance, all prepared model inputs within context, no duplicate parents/children after replay or restart.
2. Segmentation quality: build 80 representative cases (50 Korean, 20 English, 10 mixed-language), covering AI/human origins, single-topic prose, bullets, headings, no punctuation, contrast, pronouns, recurring topics and long single-topic text. Use 40 for calibration and 40 held out. Two human readers mark required, optional and forbidden boundaries on held-out cases. Require at least 90% recall on agreed required boundaries and at most 5% cuts at agreed forbidden boundaries; preserve the full disagreement record. At least 95% of agreed single-topic cases must stay one unit. A synthetic test set cannot alone establish classroom readiness.
3. Recovery: crash after acceptance, after splitting, during publication, and after correction; retry returns the same parent and a coherent active revision. Reload restores the exact draft/envelope. Withdrawal is never resurrected.
4. Interaction: join failure is understandable, target/week transitions cannot silently reassign drafts, source terminology passes AI-paste and human-writing scenarios, mobile tap/keyboard reading works, export rows match displayed scope.
5. Load: 30 simultaneous long submissions with 1,000 existing units, including Korean/emoji-heavy inputs, then a burst at configured maximum length. Target warm-server acknowledgement p95 <= 250 ms on loopback and <= 1 s over the actual class connection. For the representative 2,000–5,000-character workload, target all 30 published within 30 seconds. For maximum-length workload, target completion within 120 seconds. Event-loop heartbeat p95 <= 100 ms. These budgets must be measured; adjust scheduling or the explicitly displayed input bound if they fail, without hiding failures.
6. Classroom smoke test: real Mac browser and physical phone through the actual QR URL, model warmed, optional viewer using real workers, then close/reopen the session and verify export/restore. Report skipped device or library checks as unverified.

The existing short-opinion load results do not satisfy the new segmentation or long-input gates.

## Scope boundaries

This release does not add ChatGPT generation, automatic AI authorship detection, sentiment grading, non-contiguous topic rewriting, a full roster editor, multiple classes in one database, or a new account system. It delivers the clarified submission workflow and repairs the interaction stages necessary to use and interpret it reliably.
