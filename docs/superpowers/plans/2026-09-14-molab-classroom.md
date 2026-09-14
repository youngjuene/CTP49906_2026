# Molab classroom improvement implementation plan

**Goal:** Apply the approved QA recommendations to make the Korean AVLLM lab clearer, explorable, and suitable for evidence-based student submissions.

**Architecture:** Keep the existing marimo notebook and shared model. Put tested, low-dependency safety/display/ledger logic in small src helpers. Keep forms independent of the run ledger so recording observations cannot repeat GPU work.

**Specification:** The approved 2026-09-14 classroom QA report and its 37-item source audit, preserved in the parent workspace outputs/molab-qa-2026-09-14/.

- [x] Record integrity: match settings and named clip pairs, retain all unique runs, report persistence errors, export full JSON and readable Markdown evidence.
- [x] Display: reconstruct readable Korean captions without inventing token probabilities, show complete answer text and truncation, use a bounded negative-drop percentage and common color scale, bundle a licensed Korean font.
- [x] Inputs: constrain frame count, reject blank prompts and missing metadata, bound decoding and encoded sequence length on every route, separate form token limits from global reruns.
- [x] Teaching flow: start/run instructions, short exploration missions, observation versus inference, matched control explanation, no categorical listening claims; optional advanced inputs remain available.
- [x] Reproducibility: preserve edited clone contents, pin helper/model identity, record resolved versions and runtime provenance, validate replay clip/frame identity.
- [x] Verification: reproduce the QA bugs in focused tests before implementation; run helper tests, both notebook dataflow graphs, and Korean/English CPU replay; inspect actual UI and GPU student actions on a reviewable branch.

Validation commands: `.venv/bin/python -m pytest avllm_interpretability/tests -q`; `.venv/bin/python -m marimo check avllm_interpretability/CTP49906_avllm_molab_kr.py`; Molab mirror/fork of the branch notebook with RTX Pro 6000, Run all, Korean/English control pairs, frame/token changes, invalid inputs, verdict and export.

No silent fallback to main, unmeasured GPU capacity claims, or destructive checkout reset. Existing English notebook imports remain compatible. Integration is tracked in [PR #1](https://github.com/youngjuene/CTP49906_2026/pull/1), targeting `main`; the user authorized this merge on 2026-09-14. Preserve the immutable GPU-tested notebook and helper references when updating the deployment documentation.

Completed against notebook commit `61d540d01d6ed9ca46e98933ad961238dbfbeba8`: 293 CPU tests, Molab RTX Pro 6000 Run all, seven unique experiments, input/verdict guards, cache-cap recovery, browser reconnect and kernel restart recovery. Full evidence exports were verified from their visible copy previews; actual browser download receipt, server recreation and classroom GPU concurrency remain unverified. See [the QA record](../../../avllm_interpretability/QA_IMPROVEMENTS.md).
