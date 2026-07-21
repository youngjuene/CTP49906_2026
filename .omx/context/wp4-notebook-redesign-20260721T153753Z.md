# WP-4 primary notebook redesign context

## Task
Execute WP-4 only by redesigning `avllm_interpretability/CTP49906_avllm_molab.py` against the approved PRD/test specification and the completed WP-1–WP-3 foundations. Stop before WP-5.

## Desired outcome
A strict-valid Marimo treatment notebook with exactly four student-facing sections and eight ordered stages:

1. **Prepare your project** — Orient, Compose, Register V1
2. **Guided demonstration** — Observe
3. **Exploratory playground** — Experiment, Compare, Revise V2
4. **Synthesis and architecture challenge** — Synthesize

The notebook must present a short deterministic guided replay before open exploration; use polished classroom language; and implement Required / Choice / Advanced progressive disclosure.

## Canonical sources
- `.omx/plans/prd-avllm-arts-integrated-classroom.md`
- `.omx/plans/test-spec-avllm-arts-integrated-classroom.md`
- `.omx/logs/wp1-wp3-team-evidence-20260721.json`
- `study_materials/wp0/` frozen contracts and release/governance gates

## Known facts and evidence
- WP-1–WP-3 source/test tree is integrated and previously passed 95 combined tests.
- WP-1–WP-3 evidence SHA-256: `e2831488af4d6ef2c04589172fde259f64a72e9b6ee1ee912af7fdf3a966d9b1`.
- Existing notebook is 1,833 lines and already has coarse Prepare / Guided demo / Research playground / Synthesis headings, but not the approved eight-stage classroom composition.
- Notebook pre-WP-4 SHA-256: `4302b5933fed8d7e54b39c2e8e2989d7e4a786630391c636a082bbd8a873c064`.
- Installed course environment: Python 3.10.16, Marimo 0.23.14, Ruff 0.8.6, Mypy 1.14.1.
- The current main worktree is intentionally dirty and contains the approved WP-0–WP-3 state plus older unrelated work; preserve all non-WP-4 content.

## Single-owner boundary
- Exactly one executor owns `avllm_interpretability/CTP49906_avllm_molab.py` and any WP-4-specific notebook tests it adds.
- No other worker or leader edits the notebook.
- The worker must not edit `.omx/ultragoal`, planning artifacts, comparison-arm files, WP-5+ artifacts, or unrelated implementation files unless a minimal notebook-test helper is indispensable and reported before widening scope.

## Required behavior
- Default to Teaching mode; missing/invalid Research configuration must not expose collection actions or destinations.
- No automatic or unapproved student-data egress.
- Capture and commit immutable pre-run prediction, initial explanation, evidence, challenge/rival explanation, revised explanation, and V1→V2 creative decisions.
- Use command nonces and the completed pure reducer so reactive reruns/reloads remain exactly once.
- Support controlled swap, signed offset, silence/neutral signal controls, true audio/video modality omission, and separately labeled direct attention-edge knockout.
- Never substitute silence/neutral input for modality omission.
- Present entropy, probability margin, and aligned trajectories as uncalibrated probe/teacher-forced summaries; do not claim calibration, confidence, free-generation uncertainty, or causal localization.
- Keep audience/model readings hidden until valid audience import/reveal sequencing permits them.
- Preserve V1 when V2 is registered; render from committed reducer state rather than mutable forms.
- Every expensive operation is explicit-button gated and snapshots displayed inputs at submission.
- Cached replay/live provenance is visible and cannot be confused.

## Student-facing language constraints
Student-rendered cells must contain no PRD/FR/AC/WP IDs, reviewer/research-review comments, developer annotations, implementation history, roadmap/release narration, or internal contract prose. Use concise classroom instructions, evidence limits beside the relevant output, accessible labels, and visible checkpoint/next-action cards.

## Validation requirements
- `marimo check --strict` on the primary notebook.
- Python compile plus Ruff on the notebook and changed WP-4 tests.
- Existing AVLLM/common/WP-0 regression suites.
- Cached replay clean-process smoke test without GPU/model download.
- Tests for the eight-stage hierarchy, forbidden student-facing text, progressive disclosure, button gating, Teaching default/no egress, exactly-once reducer behavior under reactive rerun/reload, V1/V2 preservation, intervention-label separation, distribution caveats, and provenance.
- Resource-bound observation: cached CPU path peak RSS and no GPU allocation; preserve the existing bounded live/capture contracts and report live-GPU verification as unresolved if not available.
- `git diff --check` and generated-artifact scan.

## Stop gates / unresolved external work
- Do not claim research readiness. Institutional governance, licensed release stimuli, live Molab GPU/VRAM rehearsal, and human/browser accessibility review remain external gates unless concrete evidence already exists.
- Do not begin WP-5 or rewrite the comparison/audience/study protocol surfaces.
- Stop after WP-4 implementation, evidence, leader integration, and verification.
