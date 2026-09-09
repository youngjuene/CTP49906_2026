# Segmentation evaluation — 2026-09-09

This report is an implementation experiment, not classroom acceptance. **No two-reader human boundary judgments have been supplied.** Automatic segmentation remains disabled in the classroom release policy while development integration can exercise it.

## Reproduction and corpus

Runtime environment: Python 3.12.13, NumPy 2.5.3, sentence-transformers 6.0.1, transformers 5.16.1. Existing dependency pins remain unchanged. The candidate experiment installed only into the isolated virtual environment: Chonkie 1.7.0, chonkie-core 0.10.2, tokie 0.1.4, tenacity 9.1.4.

`tests/fixtures/segmentation_cases.jsonl` contains 80 synthetic fixtures: 50 Korean, 20 English, 10 mixed-language; 40 calibration and 40 held out. Cases cover multi-sentence single topics, paragraphs, bullets, headings, omitted punctuation, contrast, pronouns, recurring topics, independent short observations, long single topics, NFD and emoji. Each includes required/optional/forbidden code-point positions and an explicit synthetic expectation label. The AI/person source fields describe hypothetical source scenarios, not actual human authorship. `human_judgments` is empty throughout.

The corpus uses a small set of reusable scene-feedback topics in different combinations. This limits diversity and means even held-out synthetic performance can overestimate generalization. In particular, repeated long cases contribute many forbidden boundaries; inspect per-case failures rather than relying on their denominator in a pooled rate. Empty optional-boundary lists reflect unambiguous construction expectations, not genuine inter-reader agreement.

The evaluator compares the structure-only candidate baseline against adjacent context widths 1 and 2 and thresholds 0.60, 0.70, 0.80, 0.90, under both STS and Clustering prompts. It computes each case/prompt/context score set once and reuses those scores across thresholds. The selection score, fixed before held-out access, is required-boundary recall minus forbidden-cut rate plus half the single-topic preservation rate minus processing failures. This is only a synthetic screening score.

The first CUDA attempt failed before evaluating cases because PyTorch rejected NVIDIA driver 12060 as too old. No drivers or runtime dependencies were changed. The measured run uses CPU; model-load time is separate from per-case warm inference latency.

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/evaluate_segmentation.py \
  --cases tests/fixtures/segmentation_cases.jsonl --split calibration \
  --policy config/segmentation.json --output docs/qa/segmentation-calibration.json \
  --calibrate --chonkie-probe --device cpu

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/evaluate_segmentation.py \
  --cases tests/fixtures/segmentation_cases.jsonl --split heldout \
  --policy config/segmentation.json --output docs/qa/segmentation-heldout.json --device cpu
```

## Candidate inspection

The installed Chonkie `SentenceTransformerEmbeddings` constructor accepts the already loaded SentenceTransformer and preserves object identity, so a second model is unnecessary. Its stock `embed_batch` calls `model.encode(texts, convert_to_numpy=True)` without an explicit task prompt. Changing the shared default prompt would be unsafe, and is not performed.

The stock semantic chunker concatenates sentences for similarity windows and calls the embedding adapter before its output-size grouping. Its output `chunk_size` therefore does not constrain the fully prepared model inputs. Its later `_split_groups` also creates additional visible chunks when a semantically coherent group exceeds that size. The measurement probe wraps the stock adapter in a guard that records actual content/prepared token lengths and refuses known oversized input before silent model truncation. A second probe uses the safe runtime encoder to isolate the visible-unit length behavior.

Internal `Sentence` offsets are ordinal values in Chonkie 1.7.0, but `_create_chunks` reconstructs output offsets cumulatively. The initial suspicion about those internal offsets is **not** used as a rejection reason. Exact output partition behavior is checked separately on a repeated-text example. No skip-and-merge or overlapping refinement is enabled.

Sources checked against the installed package: [Chonkie semantic chunker documentation](https://docs.chonkie.ai/oss/chunkers/semantic-chunker), [Chonkie embeddings documentation](https://docs.chonkie.ai/oss/embeddings/overview), [Chonkie 1.7.0 package](https://pypi.org/project/chonkie/1.7.0/).

## Measured results

The CPU calibration completed on all 40 calibration cases with the real model. Model load took 5.77 seconds. The resolved EmbeddingGemma revision is `57c266a740f537b4dc058e1b0cda161fd15afa75`. The selected frozen policy is **STS similarity, adjacent context width 1, threshold 0.60**. Its release flag remains false.

| Calibration configuration | Required recall | Forbidden cuts | Single-topic preservation | Valid partitions | Warm p95 |
| --- | --- | --- | --- | --- | --- |
| Structure-only candidates | 33/35 (94.29%) | 417/420 (99.29%) | 0/8 | 37/40 | 71.39 ms |
| **STS, width 1, 0.60** | **32/35 (91.43%)** | **27/420 (6.43%)** | **8/8 (100%)** | **40/40** | **190.05 ms** |
| STS, width 2, 0.60 | 26/35 (74.29%) | 25/420 (5.95%) | 8/8 (100%) | 40/40 | 296.49 ms |
| Clustering, width 1, 0.80 | 20/35 (57.14%) | 6/420 (1.43%) | 7/8 (87.5%) | 40/40 | 190.00 ms |

The selected configuration's median warm case time was 132.94 ms. These are isolated CPU segmentation timings, not end-to-end classroom throughput or publication latency. Three structure-only cases exceeded the 64-unit limit; the selected semantic configuration had zero processing errors. All 16 searched configurations and individual offsets/scores remain in `segmentation-calibration.json`.

**Failure analysis:** three required boundaries were missed: two punctuation-free English/mixed passages had no suitable structural candidate; one Korean pair concerning a narrow passage and a hard-to-find entrance sign was merged. Forbidden cuts occur especially between an observation and its recommendation in bullets/headings, and around pronouns or recurring topics. Long repeated cases dominate the denominator: after excluding the three long cases, the forbidden-cut rate is **27/63 (42.86%)**. The mean per-case forbidden-cut rate among cases with forbidden boundaries is **36.49%**. The pooled 6.43% must not be interpreted as near classroom readiness.

The synthetic calibration already misses the proposed 5% forbidden-cut target. Moreover, its expectations were authored synthetically, not independently judged. No classroom human-quality gate has passed, regardless of recall or single-topic preservation.

The frozen policy file SHA-256 is `7292becebd9967f9ea9f34d9064ffe907c57071186eaf0dc937d53f9105640f0`. Corpus SHA-256 is `d8a928264ab834ad9c5ded9a7d1c429f0817e74c72ff7082bce10a76963dae76`. The selected algorithm key under the measured model/runtime is `adjacent-context-v1:6be26e0bbee538c5eff9e12e5ef310cea076edbd03870bc999574306ea0c3ff5`.

### Measured Chonkie decision

Chonkie 1.7.0 reused the same model object and its short repeated-text output partition was exact. The model's default prompt was `null`; the stock wrapper would therefore apply no STS prompt. The shared default remained unchanged throughout the probe.

The stock adapter attempted a window with **5,602 content tokens and 5,611 prepared tokens**, above the 2,048-token model context. The evaluation guard blocked that call before silent truncation; this guard is ours, not protection provided by Chonkie. With safe runtime embeddings substituted to isolate output behavior, a repeated coherent case still became **two visible chunks** solely under the 1,792-token chunk-size policy.

The stock adapter/chunker was therefore not adopted. Satisfying both requirements would require replacing its input-window handling and its visible-length grouping behavior. The implementation instead uses the bounded adjacent-context adapter with independent original-span validation. This is a scoped decision about the tested stock integration, not a claim that no custom Chonkie implementation could satisfy the requirements. No Chonkie production dependency was added.

### Long-vector and held-out checks

The original calibration long-vector probe was insufficient: although its input had 6,539 code points, the loaded tokenizer counted only 1,200 content / 1,208 prepared tokens and it used one window. Its finite normalized output is a short-path result; **it does not verify real long-input pooling**.

The evaluator now constructs a separate, explicitly synthetic stress input that grows until the unwindowed prepared input exceeds 2,048 tokens while remaining within 20,000 code points. It asserts at least two bounded internal windows, one visible unit, exact internal substring coverage, and agreement with an independently computed normalized token-weighted mean. The new regression was observed failing for the absent helper before implementation. The real held-out run exercised the new helper successfully: 4,095 code points / 2,432 content tokens / 2,440 prepared tokens became two internal windows (maximum 1,792 content / 1,800 prepared tokens), one visible unit, exact internal coverage, a finite unit-norm vector and agreement with the independent weighted-pool reference. This supplemental probe does not alter the corpus, selected policy, or calibration scores.

The frozen policy was evaluated once on the 40 held-out synthetic cases. Results in `segmentation-heldout.json`: **32/35 required boundaries (91.43%)**, **21/420 forbidden cuts (5.00%)**, **3/8 single-topic cases preserved (37.5%)**, **40/40 valid partitions**, zero processing failures. Warm CPU median/p95 were **139.37/177.89 ms**.

The single-topic preservation target was missed by a wide margin. All three repetitive long cases survived, but the five shorter single-topic cases split; long repetitions also dilute the forbidden-cut denominator. Excluding long cases, forbidden cuts are **21/63 (33.33%)**. The real stress probe verifies context safety and pooling, not semantic coherence across ordinary observation/recommendation pairs.

**Release decision: NO-GO for automatic classroom segmentation.** Do not enable `classroom_enabled` or adjust the frozen policy after seeing held-out results to manufacture a pass. Further algorithm/corpus work and independent human evaluation are required.

## Release gate

Structural tests and real-model synthetic evaluation are prerequisites, but neither supplies the required independent human judgments. Classroom enablement still needs two readers, agreement/disagreement records, at least 90% recall on agreed required boundaries, at most 5% cuts on agreed forbidden boundaries, and at least 95% preservation of agreed single-topic cases. Real classroom load, recovery integration and actual-device testing are separate gates owned by the coordinating task.
