# Feedback Atlas implementation and release evidence

Implementation date: 2026-09-09. **Classroom rollout: NO-GO.** Durable submission, recovery, correction and classroom UI work is implemented in an isolated branch. Automatic topic splitting fails the semantic-quality requirement; the real-model server and classroom launcher refuse unreviewed operation by default. No deployment was performed.

## Location and provenance

- SSH host: `june`.
- Branch: `codex/feedback-atlas-classroom-20260909`.
- Isolated repository: `/mnt/hdd/research/2026/.codex-worktrees/ctp49906-feedback-atlas-20260909`.
- Application: that repository's `feedback-atlas/` directory.
- Local review mirror: `/Users/youngjuene/Code/ctp49906/feedback-atlas`.
- Baseline snapshot commit: `c02de3e277f34150efd7ffb7639184f1897cec7f`. This preserves the existing dirty source before implementation; it is not the original repository HEAD.
- All 74 source files in the captured original manifest still match their original-checkout hashes at final review. Work and synthetic QA used the isolated checkout, temporary databases and a separate virtual environment. Existing model/package caches were read; no driver changes were made.
- Built `web/vendor/embedding-atlas.js` SHA-256: `31f953762ffdd5bb2e218ea0139e701213f1bd712fd925d029adc5347f1ee2b2`.
- Unchanged frontend lockfile SHA-256: `ef97156ad94561c4b9cde6cd22740428141484980ba468aed8dd4964d9c16b04`.

## Delivered behavior

1. **Feedback source is explicit.** `피드백 출처`, `AI 생성`, and `사람 작성` refer to where the feedback originated. AI remains the intentional default. Submitter identity is a separate private field; AI processing never changes a human source label.
2. **One whole paste is one durable original.** Up to 20,000 Unicode code points are saved before acknowledgement. Exact original spans form linked topic units; internal model windows never become map points merely because of their length. Raw originals remain immutable through corrections.
3. **Recovery follows the same submission.** IndexedDB saves drafts, immutable transmission envelopes, receipts and ownership capabilities. Lost acknowledgements/restarts replay the same nonce. Competing tabs cannot silently overwrite a newer stored draft or enqueue two originals in one scope. Storage failure preserves visible text and offers download.
4. **Context changes are deliberate.** Empty forms follow the current presenter/week; dirty forms keep their context until confirmation. Rejected original A remains recoverable even if the participant has already typed draft B. Closed intake keeps unsent text locally.
5. **Processing and publication are separate.** A persistent queue uses one shared model executor, bounded batches, retries and complete revision publication. No placeholder points are exposed. Stale computation cannot republish withdrawn originals. Metadata-only corrections reuse vectors and geometry.
6. **Owners can inspect and correct.** The reader shows exact originals, derived units, source, target/week and revision history. It supports unit source correction, adjacent merge, a grapheme-aware split cursor, target/week correction and withdrawal. Unknown mutation outcomes require retrying the original action; late responses cannot overwrite another open editor.
7. **Map and export scopes agree.** Filters preserve coordinates. Counts distinguish units, originals and the instructor's server-computed distinct submitter aggregate. Visible/selected/all export previews include row counts and data revision; empty selection exports zero rows and stale revision requires a fresh action.
8. **Operations preserve accepted feedback.** Imports use durable parents and idempotent row identities. Recovery backups include unpublished originals and credentials; pseudonymous archives remove ownership handles while preserving lineage. The launcher checks readiness and requires backup before stopping. Instructor diagnostics include queue age and failure stage.

The optional advanced analysis viewer stays disabled until its API can match the classroom map's filter and selection semantics. Its generated worker paths are repaired; this does not establish full viewer readiness.

## Verification results

All automated scenarios used synthetic data. Commands run from the isolated application directory.

| Verification | Observed result |
| --- | --- |
| Default backend suite | **335 passed**, 2 warnings, 60.62 s |
| Final six-module browser collection | **32 passed, 1 skipped**, 124.01 s, exit 0; includes all final protocol controls and independent-review regressions |
| Browser collection breakdown | 31 classroom interaction cases passed; asset test passed (three worker URLs and live clustering worker); real renderer skipped for unavailable compatible WebGPU adapter |
| v2 fake-model classroom load and PCA/UMAP threshold | **2 passed**, 37.19 s; functional throughput evidence only |
| Frontend build | `npm --prefix frontend run build` succeeded using Node 24.4.1 / Vite 8.2.2 and the existing lockfile |
| Independent review | Two backend findings and three browser findings reproduced and fixed; final bounded re-review closed all five, with no additional confirmed blocker |

The protocol-refresh increment first passed two focused real-server tests (17.54 s), then passed again in the final combined collection. It covers preserved outbox A/new draft B and storage-denied download before refresh; see [the browser report](classroom-ui-report.md). The default suite and browser command above are different collections; opt-in browser, model evaluation and load modules are not silently counted as default tests.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/browser_classroom.py tests/browser_submission_flow.py tests/browser_submission_corrections.py tests/browser_export_scopes.py tests/browser_viewer.py tests/browser_viewer_assets.py -q --maxfail=2
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest tests/classroom_load.py -q -s
```

Real-model CPU runs used the cached EmbeddingGemma revision `57c266a740f537b4dc058e1b0cda161fd15afa75`. Both started with 1,000 units and 30 simultaneous participants, exercised immutable duplicate replay and checked exact originals and final membership. Startup was measured separately (63–65 s), not included in warm publication latency.

| Per-participant input | Receipt p95 | Ready p95 | Run completion upper bound for all ready | Heartbeat p95 | Final units |
| --- | --- | --- | --- | --- | --- |
| 3,176 code points | 48.11 ms | 17.29 s | 18.82 s | 16.29 ms | 1,150 |
| 20,000 code points | 54.47 ms | 24.93 s | 27.14 s | 63.71 ms | 1,990 |

These meet the planned workload thresholds: receipt p95 ≤250 ms, heartbeat p95 ≤100 ms, all ready within 30 s for the representative workload and 120 s for the maximum-size workload. The artifacts predate an explicit `ready_max_s` field; the recorded total run time provides an upper bound instead. Maximum-workload vectors were finite and normalized; its final snapshot was 2,725,165 bytes.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATLAS_SEMANTIC_LOAD_REAL=1 ATLAS_SEMANTIC_LOAD_OUTPUT=docs/qa/semantic-load-real.json \
  .venv/bin/python -m pytest tests/classroom_semantic_load.py -q -s
# Add ATLAS_SEMANTIC_LOAD_MAXIMUM=1 and change output to semantic-load-maximum.json
# to exercise 20,000 code points per participant.
```

The first maximum-load attempt timed out its WebSocket keepalive because the harness repeatedly polled private status and stopped draining clients once their own result arrived. The harness now subscribes once and keeps all clients receiving, as the browser does. The successful measurements use that corrected workload. GPU evaluation also failed because the installed PyTorch rejected driver 12060; measurements use CPU and do not claim GPU readiness.

Artifacts: [representative load](semantic-load-real.json), [maximum load](semantic-load-maximum.json), [browser scenarios](classroom-ui-report.md), [operations progression](operations-report.md).

## Semantic release blocker

EmbeddingGemma provides vectors; selecting meaningful topic boundaries is a separate algorithm. The bounded adjacent-context adapter is implemented, with task-specific prompts, one shared executor and exact span validation. Stock Chonkie 1.7.0 was evaluated but not adopted: its unbounded prepared inputs and size-driven visible splits did not satisfy the contract. No Chonkie runtime dependency was added.

The frozen policy is STS similarity, adjacent context width 1, threshold 0.60. Its `classroom_enabled` flag is false. The 80-case corpus is synthetic (40 calibration, 40 held out); it has no independent human boundary judgments.

| Held-out metric | Observed | Planned human-reviewed threshold |
| --- | --- | --- |
| Required boundary recall | 32/35 = 91.43% | ≥90% |
| Forbidden boundary cuts | 21/420 = 5.00% | ≤5% |
| Single-topic cases left intact | **3/8 = 37.5%** | **≥95%** |
| Exact original partition | 40/40 | All cases |

Long repetitive cases dilute the pooled forbidden-cut rate; excluding them gives 21/63 = 33.33%. All five shorter single-topic held-out examples were split. Human agreement cannot be inferred from these synthetic labels. The current quality result therefore fails release even though storage, token bounds and latency work.

A separate real-tokenizer stress case exceeded the 2,048-token model context: 2,440 prepared tokens became two internal windows, at most 1,800 prepared tokens each, while remaining one visible unit. Its pooled vector matched an independent normalized token-weighted reference. This proves bounded long-input handling, not semantic quality.

Before classroom enablement, improve the boundary algorithm using a new calibration set, have two readers label realistic inputs and record disagreements, and evaluate the frozen revision on an untouched held-out set. Do not retune on these held-out results or lower thresholds to claim acceptance. See [full semantic evidence](segmentation-evaluation.md).

## Human interaction evidence and remaining release work

A Mac embedded-browser walkthrough connected through an owned localhost SSH tunnel to a temporary synthetic preview. It covered joining, explicit source/target choice, a full multi-topic paste, exact original inspection, reload/receipt recovery, unit-source correction, instructor login, search and visible/empty-selection export previews. The fake model was used; its chosen boundaries do not establish quality.

Chromium tests add phone-sized/touch and keyboard scenarios. **Two actual people, a physical phone, the real classroom QR URL and a live operator walkthrough were not tested.** Full advanced-viewer WebGPU rendering was unavailable. Those are separate release checks; no claim of classroom readiness is made.

A local synthetic preview can be recreated without opening any class database:

```bash
# On june, from the isolated application directory:
.venv/bin/python scripts/qa_preview.py --port 18126
# On the Mac, in a separate terminal:
ssh -N -L 127.0.0.1:18126:127.0.0.1:18126 june
# Open http://127.0.0.1:18126 ; synthetic participant writer1 or writer2.
# Instructor route /#admin uses qa-preview. Stop both commands when finished.
```

Real-model development additionally requires `ATLAS_ALLOW_UNREVIEWED_SEMANTICS=1`; it is an explicit development override, not classroom approval. No real classroom endpoint was opened during implementation. The two temporary Mac preview tabs, owned preview server and localhost SSH tunnel were closed after QA.

## Migration and rollback procedure

After semantic and actual-device gates pass, schedule a change between sessions. Preserve unsent v1 browser text and resolve old in-flight submissions before refreshing clients. Close intake, drain or preserve pending/failed originals, export instructor recovery data, and take a private full SQLite backup. Validate migration on a copy first.

```bash
.venv/bin/python scripts/backup_database.py --db atlas.db --out recovery/atlas-before-v2.db
```

For a v1 database, migration takes a fresh mode-0600 consistent backup before schema changes. The default name is `<database>.pre-v2-backup`; an existing name causes a timestamp suffix. Successful migration records the exact name in the SQLite `meta` row `migration_backup_path`. A failed schema transaction rolls back schema/data changes. Tests verified rollback and backup freshness; no live database was migrated.

On rollback, close intake and retain the current v2 database, a private full backup and recovery exports of every post-upgrade accepted original, including failed/pending work. Account for those submissions before starting matching older code with a copied older database:

```bash
.venv/bin/python scripts/backup_database.py --db recovery/atlas-before-v2.db --out restored-v1.db
# Set ATLAS_DB to the unused restored-v1.db with the matching old code.
```

Never overwrite newly accepted feedback with an older backup. Legacy records without an owner capability remain instructor-recoverable; knowledge of a submission ID alone does not grant ownership. Deidentified archives remove recovery credentials and are unsuitable as classroom recovery databases.
