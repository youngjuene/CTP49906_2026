# Molab classroom improvement implementation plan

**Goal:** Apply the approved QA recommendations to make the Korean AVLLM lab clearer, explorable, and suitable for evidence-based student submissions.

**Architecture:** Keep the existing marimo notebook and shared model. Put tested, low-dependency safety/display/ledger logic in small src helpers. Keep forms independent of the run ledger so recording observations cannot repeat GPU work.

**Specification:** The approved 2026-09-14 classroom QA report and its 37-item source audit, preserved in the parent workspace outputs/molab-qa-2026-09-14/.

- [ ] Record integrity: match settings and named clip pairs, retain all unique runs, report persistence errors, export full JSON and readable Markdown evidence.
- [ ] Display: reconstruct readable Korean captions without inventing token probabilities, show complete answer text and truncation, use a bounded negative-drop percentage and common color scale, bundle a licensed Korean font.
- [ ] Inputs: constrain frame count, reject blank prompts and missing metadata, bound decoding and encoded sequence length on every route, separate form token limits from global reruns.
- [ ] Teaching flow: start/run instructions, short exploration missions, observation versus inference, matched control explanation, no categorical listening claims; optional advanced inputs remain available.
- [ ] Reproducibility: preserve edited clone contents, pin helper/model identity, record resolved versions and runtime provenance, validate replay clip/frame identity.
- [ ] Verification: reproduce the QA bugs in focused tests before implementation; run helper tests, both notebook dataflow graphs, and Korean/English CPU replay; inspect actual UI and GPU student actions on a reviewable branch.

Validation commands: `.venv/bin/python -m pytest avllm_interpretability/tests -q`; `.venv/bin/python -m marimo check avllm_interpretability/CTP49906_avllm_molab_kr.py`; Molab mirror/fork of the branch notebook with RTX Pro 6000, Run all, Korean/English control pairs, frame/token changes, invalid inputs, verdict and export.

No silent fallback to main, unmeasured GPU capacity claims, or destructive checkout reset. Existing English notebook imports remain compatible. Publishing to main requires review; the review branch supplies the concrete proposed result.
