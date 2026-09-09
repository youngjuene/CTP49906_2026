# Semantic implementation report — 2026-09-09

Status: runtime/splitter implemented; 37 focused tests passed before the final cache-dimension fix. That regression failed as expected and its correction awaits the next combined test run. CPU real-model calibration is complete; selected STS/width 1/threshold 0.60. Held-out evaluation and the corrected real long-input probe completed. Single-topic preservation was only 3/8 (37.5%), so classroom release is NO-GO. Classroom human-quality approval remains **not passed**.

## Interfaces

- `ModelRuntime(embedder, content_token_limit=1792, total_token_limit=2048, batch_size=32)` owns one `ThreadPoolExecutor(max_workers=1)` around the already loaded embedder.
- `await runtime.run(fn, *args)` enters the model worker. `runtime.run_sync(fn, *args)` supplies the blocking startup path and safely supports nested worker calls. Use it from blocking startup work, not the event loop.
- `encode_similarity(texts)`, `encode_units(texts)`, and compatibility `encode(texts)` are synchronous and refuse calls outside that worker. `spec`, `dim`, and `model_id` preserve the existing embedder interface.
- `runtime.unit_cache_key` and `runtime.similarity_key` separate the tasks and fingerprint model revision, actual prompt, installed model libraries, dimension, normalization and long-unit pooling/window policy. AtlasState must use the runtime key when receiving this adapter.
- `runtime.close()` joins the executor. Call from shutdown work off the event loop if inference may still be running.
- `load_policy(path=None)` returns an immutable `SegmentationPolicy`. `split_feedback(raw, runtime, policy)` runs synchronously on the same worker and returns `SplitResult` with exact `FeedbackSpan` offsets plus algorithm fingerprint. `validate_spans` delegates to the shared `validate_partition` invariant.

## Behavior

Candidate positions come from original code-point offsets: punctuation, Korean final forms, paragraph and bullet structure. Standalone headings attach to following prose. Adjacent context embeddings select output boundaries; repeated text is never looked up from offset zero, reordered, or merged across an intervening unit. The algorithm has no output overlap or forced two-sentence minimum.

Long units retain one visible span where semantic scores support that grouping. Internal windows obey both 1,792 content tokens and 2,048 total prepared tokens, further limited by the actual model context. Windows are original substrings rather than decoded token sequences. Normalized content-token-weighted pooling produces the long clustering vector; similarity vectors are not reused as clustering vectors. Short normalized inputs call the existing clustering encoding path.

The 64-unit limit raises a recoverable error rather than silently truncating or merging topics. Parent storage is responsible for preserving the accepted original and reporting processing failure.

## Recorded test progression

1. New contract tests: **24 failed** because the new span/runtime modules were absent. `src/submissions.py` was subsequently supplied by the coordinating task.
2. First implementation: **24 passed**, with three newly introduced prompt/context tests failing because `encode_task` did not exist.
3. Added prompt adapter: **17 passed, 1 failed** in runtime + existing embedder tests; the remaining expected failure was the newly added `run_sync` contract.
4. Added `run_sync`: **37 passed** across `tests/test_segmentation.py`, `tests/test_model_runtime.py`, and `tests/test_embedder_contract.py` on isolated Python 3.12.13.
5. Review added a fake-vector dimension cache regression. It failed as expected because dimensions 8 and 16 had the same key; dimension is now included universally (final verification pending).
6. Local syntax compilation succeeded with a temporary Python cache. The Mac system Python has no NumPy; authoritative execution is the isolated June environment.

Fake vectors prove structural and execution invariants only. They are not evidence of semantic quality.

## Evaluation and limitations

The first CUDA evaluation could not run: the installed PyTorch runtime rejected the NVIDIA driver (reported driver 12060 too old). No driver or dependency upgrades were attempted; the CPU calibration completed successfully.

See `segmentation-evaluation.md` and the generated calibration/held-out JSON for actual model results. The fixed 80-case corpus has 50 Korean, 20 English and 10 mixed-language cases, split 40/40. Every expectation is explicitly synthetic; declared source alternates hypothetical AI/person scenarios. There are no human-authored submissions or human boundary judgments in this fixture.

Two-reader held-out judgments, classroom semantic acceptance, real long-submission load testing and physical-device validation remain outstanding. The immutable policy retains `classroom_enabled: false`; this is a release gate, not a claim that deterministic development integration cannot execute the algorithm.

No commits, deployment, live database changes, original-checkout edits, or historical resegmentation were performed by this work.

## Handoff state

Production files are stable for integration. The frozen calibration policy is now recorded in `config/segmentation.json`; the coordinating task owns final sync, the held-out run and combined verification. Calibration measured 32/35 required boundaries, 27/420 forbidden cuts, 8/8 preserved single topics and 40/40 valid partitions. The 6.43% forbidden-cut rate exceeds the proposed 5% target, and short/non-long cases are much worse (42.86%). The original long probe stayed within one model window; an explicit over-context supplemental probe is implemented after its confirmed failing regression; the real held-out run verified two bounded windows for 2,440 prepared input tokens, one visible unit and agreement with independent normalized token-weighted pooling. Held-out semantic scores were 91.43% required-boundary recall, 5.00% forbidden cuts and 37.5% single-topic preservation; the preservation gate failed. The evaluation script must not be represented as having completed on the failed CUDA attempt. No human-quality gate is satisfied by this handoff.
