# V2 operations implementation — 2026-09-09

Implemented and verified in the isolated June checkout. **43 focused tests passed**: 37 import/archive/backup/launcher checks plus six existing parent-store checks. No real class database, server process, tunnel or deployment was operated by this work.

## Import

`scripts/seed_ai_opinions.py` now accepts exact raw parents through `Store.accept_submission`; it does not call legacy direct opinion insertion or normalize the original. CSV quoted newlines, CRLF, NFD and emoji are retained. The raw limit is 20,000 code points. Source defaults to AI, with validated per-row source/week/target overrides.

The existing default is explicitly already-unitized: one original span is staged, then the normal worker computes vectors, projection and publication. `--segment` leaves the parent queued for semantic splitting. Neither path loads a model or publishes placeholder points.

The coordinating task authorized a narrow extension to `SubmissionStoreMixin.accept_submission`: optional `initial_split` is included in the acceptance fingerprint and staged inside the same transaction. A staging failure rolls back the parent. Existing callers without an initial split retain their request-fingerprint convention.

Import replay identity hashes exact file bytes, logical row position, canonical reviewer/target, effective week/source and mode. Separate identical rows remain separate parents. Replays use the immutable request fingerprint, so a subsequently corrected parent does not need to be duplicated. Capabilities are generated randomly and only hashes are stored; raw capabilities, hashes, nonces and original text are not printed. Imported records are managed through instructor authorization.

The pending limit is 300, optionally lowered by `--max-pending`. A capacity failure retains already accepted rows, reports skipped rows and returns nonzero. Drain the queue and rerun identical file/options to resume. `--dry-run` never opens/creates a database and explicitly does not check current replay/capacity state.

This remains an **offline administrative importer**: stop the server before importing, then start the normal worker. No online bulk-intake API was invented or added. The script does not automatically detect all running server instances; the offline precondition is operationally documented.

## Archive and recovery backup

`backup_database.py` uses a read-only source connection and SQLite's backup API. Tests hold uncheckpointed committed WAL data open and verify it is present in the snapshot. Destinations are private mode 0600, checked for SQLite integrity, and published without overwriting an existing path. These full recovery backups intentionally retain credentials and pending originals; they are not pseudonymous artifacts.

`deidentify.py` reads a consistent snapshot and migrates only the private temporary copy. The source database remains byte-identical in tests. Outputs are:

- `archive.db`: all original parents, revision manifests, unit offsets/opinions and available coordinates, including pending, failed and withdrawn records.
- `archive.csv`: currently published units, with opaque parent/unit linkage and span metadata.
- `originals.csv`: every original and its state/history, including unpublished work.
- `mapping.csv`: separate private randomized roster-code mapping.

Reviewer IDs and optional target IDs are replaced consistently. Opaque parent/unit IDs and dataset ID are remapped to sever direct joins to the live dataset. Nonces, capability/request hashes, legacy receipts, action records and private metadata are removed. Unknown actor labels are redacted; failure diagnostics are reduced to a generic code. Compaction removes deleted identifiers/handles from free database pages. Every revision partition and unit quotation is checked, followed by foreign-key and integrity checks.

A review regression found that source target IDs resembling generated codes could cascade through bulk replacements. Revision updates now address individual primary keys, preserving the relation even for swapped `P01`/`P02` codes.

The copied SQLite database preserves exact raw text. CSV outputs apply spreadsheet-safety escaping. **Pseudonymous does not mean anonymous:** free text, timing and retained project information can identify people. The script does not automatically redact prose or establish research consent. Sanitized archives remove recovery credentials and must not be treated as classroom recovery databases.

## Launcher and operator instructions

`class.sh start` reports roster count, classroom context, input bounds, queue/failure counts, model readiness, viewer state and semantic-review status. It checks readiness before opening a tunnel. Unreviewed semantics or fake embeddings require the explicit `ATLAS_ALLOW_UNREVIEWED_SEMANTICS=1` development override, which is labeled as an exception rather than quality approval.

The tested `/healthz` fields are `ok`, `roster_count`, `class_context`, `limits.max_raw_codepoints`, `limits.max_pending`, `limits.automatic_units`, `model_warmed`, `segmentation_reviewed`, `processing`, `viewer`, and `cache_key`. The coordinating task confirmed integration of this exact server-side schema, plus an explicit fake-embedder flag and a real-model startup quality gate.

`class.sh backup [unused-path]` creates a full private snapshot. `stop` requires closed intake and resolved pending/failed work, creates a backup, then stops processes. `stop --force` permits unresolved work for restart recovery but still refuses to proceed if backup fails. Launcher tests stub process signals; they do not kill host processes or open real tunnels.

The development override is documented off by default in `.env.example`, and `CLASSROOM_QA.md` records operations evidence plus semantic NO-GO. English and Korean README operations sections now describe one database per class, stable-origin draft continuity, offline imports, queue replay, close/drain/recovery-export/backup/stop order, v1 migration backups, preservation of unsent old-client drafts, and rollback to a fresh filename with matching code after accounting for all post-upgrade originals.

## Verification record

1. Import/archive regressions: **8 failed, 18 passed** before implementation (duplicate import, raw newline loss, missing `--segment`, missing atomic initial split, omitted unpublished archives).
2. Import implementation: **16 passed**; backup helper regression failed because the helper did not exist.
3. Import/archive/backup implementation: **27 passed**.
4. Launcher regressions: **4 failed, 3 passed**, followed by a separate failing backup-before-stop check.
5. Combined operations: **35 passed**.
6. Archive target-code collision: observed failure, corrected per-revision updates.
7. Final command: `.venv/bin/python -m pytest tests/test_seed_ai_opinions.py tests/test_deidentify.py tests/test_backup_database.py tests/test_class_launcher.py tests/test_submission_store.py -q` → **42 passed in 2.17s**.
8. A documented explicit environment override was reproduced failing when `.env` supplied its default 0. The launcher now preserves the explicit value across `.env` loading. The same final command then passed **43 tests in 2.15s**.

Shell syntax verification also passed. Tests use temporary synthetic databases and mocked launcher processes, not live classroom data.

## Release status

Operations correctness does not clear the semantic release gate. The frozen real-model held-out synthetic evaluation measured 91.43% required-boundary recall, 5.00% forbidden cuts, but only **37.5% single-topic preservation**; independent human judgments are absent. The corrected long-input stress probe verified two bounded internal windows and normalized token-weighted pooling. See `segmentation-evaluation.md` for the measured details. **Automatic classroom segmentation remains NO-GO.**

Real classroom throughput, actual-device interaction, live operator walkthrough and deployment remain outside this operations increment. No commits were made.
