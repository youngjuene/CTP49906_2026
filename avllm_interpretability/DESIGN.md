# Design

## Source of truth

- Status: Implemented; real Studio replay and browser checks pass. Direct hosted Molab/GPU qualification remains unverified.
- Last refreshed: 2026-10-02.
- Primary product surface: the Korean notebook on the left of the Molab editor, with one HTML/CSS/JavaScript Studio workbench beside it on the right. The companion has three compact panels inside one Studio view.
- Scope: this Korean lab. The English notebook and unrelated course applications retain their own behavior.
- Evidence reviewed: notebook (58 cells, 2,960 lines), `README_kr.md`, `WORKSHEET.md`, `src/probe_grid.{py,js,css}`, `src/run_ledger.py`, `src/teacher_forcing.py`, `precomputed_kr/`, existing graph/replay tests, and [Studio documentation](https://marimo-team.github.io/marimo-studio/).
- Implementation and acceptance details: [verification guide](docs/studio-verification.md) and `tests/test_studio_*.py`.

## Brand

- Personality: a precise, approachable scientific instrument for learning through controlled experiments.
- Trust signals: visible submitted configuration, model revision, measurement units, replay/live status, baseline and control identity, and limitations beside results.
- Avoid: decorative dashboard metrics, implied causal conclusions from attention mass, and color implying that larger diversity is inherently better.

## Product goals

- Gather existing input interfaces into a persistent side workbench where learners can configure, run, inspect, and record without scrolling the notebook to locate cells.
- Preserve the Korean lesson and the distinction between observation, intervention, and interpretation.
- Make cheap exploration immediate while keeping inference behind explicit submission.
- Success signals: in the desktop split workspace, the selected method, basic inputs, Run/status and a primary result summary are accessible in the side pane without scrolling the notebook or outer page. Advanced controls/details are reached by compact tabs or disclosure; long data may scroll within its own bounded region. Display-only interactions cause zero additional model calls.
- Non-goals: replacing the model backend, adding a general experiment service, browser-based CUDA inference, automatically exploring a combinatorial parameter grid, or public multi-user GPU hosting.

## Personas and jobs

- Students: choose a question, state a prediction, compare against a control, inspect layers/tokens, record a defensible conclusion.
- Instructors: teach the guided examples, explain Korean null effects without implying measurement failure, and recover with saved results when GPU access is unavailable.
- Researchers: enter custom prompts and valid multi-rule interventions, compare metrics with their provenance, and export an evidence log.
- Confirmed delivery: upload `CTP49906_avllm_molab_kr.py` to hosted Molab and use its managed runtime/GPU. Local machine package versions and GPU access are not deployment blockers. No separately hosted application server is in scope.

## Information architecture

The first release opens **one** Studio view, `explore`, beside the notebook in the Molab editor. This is the primary delivery surface; a separate app/page is not required. Studio's authoring layout supports Notebook and Preview side by side; the hosted Molab integration must expose that actual surface. [Studio workspace](https://marimo-team.github.io/marimo-studio/guide/work-in-studio)

The side view contains the following panels. Panel selection is browser-local and does not change the URL path or public query. Separate named views are deferred; do not apply run-mode session claims to the edit-mode companion without testing the actual host. [Session behavior](https://marimo-team.github.io/marimo-studio/guide/navigation-and-sessions)

| Panel | Purpose | Main contents |
| --- | --- | --- |
| 실험실 — Explore (default panel) | Run and inspect an experiment | Persistent status, experiment selector, one visible native input form, active results, recent-run summary |
| 가이드 — Guide | Read and teach | Existing diagrams, method caveats, fixed Korean examples, captions, probe grid, attention panels |
| 실습 기록 — Evidence | Compare and conclude | Ledger, existing verdict form and worksheet download; selection-assisted verdicts in the later enhancement milestone |

Within Explore, select **프로브 다양성**, **생성 경로 차단**, or **답변 근거 점검**. These correspond to the existing logit playground, generated-token band sweep, and teacher-forcing playground. The selected mode is browser display state. Each native form owns its own draft and submitted values.

Desktop workspace sketch (the view must fit the allocated right-pane width):

```text
┌─ Molab notebook ──────────────────┬─ Studio HTML workbench ──────────┐
│ Korean lesson + Python cells     │ [실험실 | 가이드 | 실습 기록]      │
│                                  │ [experiment selector] [status]   │
│ Existing notebook remains        │ [compact basic controls]         │
│ readable/editable independently  │ [advanced/upload disclosure]     │
│                                  │ [Run] [submitted configuration]  │
│                                  │ [primary result / plot]          │
│                                  │ [tokens | layers | comparison]   │
└──────────────────────────────────┴──────────────────────────────────┘
```

For the first release, generation-band inputs explicitly show the fixed guided clip, prompt, and frames as read-only context. They must not look like the upload-capable inputs of the other modes.

## Design principles

1. Preserve experimental meaning before simplifying presentation.
2. Separate draft configuration, submitted computation, and display selection.
3. Reuse native controls and widgets before authoring custom bridges.
4. Identify the source and unit of every displayed result.
5. Make controls discoverable through progressive disclosure, with the active experiment visible at all times.

Tradeoff: retain independent native form state/submission, while compacting their layout templates as needed. Projecting the current long Markdown form cells unchanged would reproduce the scrolling problem. Reuse the underlying controls, validation and compute graph; do not interpret reuse as an obligation to keep verbose form markup. A truly shared clip/prompt state still requires a later, tested dispatcher redesign.

## Visual language

- Color: reuse existing blue/orange series and signed delta conventions, with text labels and signs. Never rely on red/green alone. Keep probe content/junk/undecodable categories distinct from confidence or correctness (`src/probe_grid.py:20`).
- Typography: system sans-serif with Korean fallbacks; monospace for rules, token text, IDs, and numeric configuration. No new font dependency.
- Spacing/layout: 8 px rhythm, compact control rows, 16–24 px result spacing. Fit controls above the primary result within the Studio side pane; do not assume another 320–380 px rail can fit inside half of the Molab window. A wider optional view can use columns.
- Shape: restrained borders and small radii; no ornamental elevation.
- Motion: only status/progress feedback; honor reduced motion.
- Imagery: reuse the instructional SVG diagrams and real clip preview.

## Components

- Reuse: `ko_controls`, `band_controls`, `tf_controls`, ProbeGrid, TextCompare, TangleSlider, matplotlib figures, native tables, ledger renderer, and download control.
- Add: view navigation, experiment switcher, launch-mode badge, submitted-configuration summary, method-aware help, and compact recent-run display.
- Initial delivery preserves native form fields and submission semantics, but compacts the templates and puts lengthy help, upload areas and advanced rules behind accessible disclosure. A later enhancement milestone adds run selection for verdicts and earlier validation. Prediction fields were deliberately removed to favor exploration (`tests/test_notebook_kr_replay.py:278`); do not restore them incidentally.
- States: unsubmitted, validating, running, completed, previous result/stale, failed, and unavailable in replay.
- Ownership: notebook owns controls, validation, computations and run records. View HTML/CSS/JS owns layout and display navigation. Keep initial CSS/JS local to each Vanilla view; do not add a design-system framework.

## Accessibility

- Target: WCAG 2.2 AA for authored shell and new controls; explicitly audit inherited widgets rather than claiming blanket conformance.
- Keyboard: full tab access, visible focus, labeled experiment tabs, range entry alternatives, no hover-only explanations; preserve ProbeGrid's existing arrow/Home/End support (`src/probe_grid.js:580`).
- Readability: Korean labels with precise English scientific terms where useful; units adjacent to values; 200% zoom without lost Run controls.
- Semantics: heading hierarchy, `aria-live` status, disabled reasons, table equivalents for key chart values.
- Sensory: keep audio user-initiated; preserve muted-control meaning; avoid autoplay.

## Responsive behavior

- Size against the **actual Studio pane**, not only the full browser window. Use a bounded-height layout, persistent method navigation and visible Run/status, with compact controls and a primary result summary.
- Desktop qualification: 1440×900 full Molab window with roughly equal notebook/Studio split; repeat at 1280×800. Record actual inner pane dimensions because host chrome consumes space. Basic configure→Run→inspect summary must not require notebook/outer-page scrolling.
- Wide pane: optional control/result columns. Narrow pane: compact vertical regions and result tabs, with independent scrolling only for extended data/help. At 390×844, use a usable mobile fallback; simultaneous notebook/Studio visibility is a desktop requirement.
- Do not duplicate projected controls or reconstruct forms on resize, disclosure or result-tab changes. Scrolling the notebook must not move the workbench out of view.
- Touch: usable range inputs and explicit selection, with a textual alternative to hover inspection.

## Interaction states

- Loading: distinguish environment/model startup from experiment execution; report the submitted configuration while running.
- Empty: show the next action and what the selected method measures.
- Error: render a local Korean explanation; keep the last valid result clearly marked as previous, where supported. Never attribute that result to the failed draft.
- Success: show exact effective inputs, control flag, metric and unit, and the corresponding run ID.
- Disabled: replay offers saved guided exploration; new inference/upload controls are unavailable with an explanation. Backend guards remain authoritative.
- Offline/slow network: saved replay still requires installed Python/native/widget dependencies in the first release. Static/offline delivery is not implied.

## Content voice

- Tone: direct, educational Korean.
- Terminology: `source` is the querying token, `target` the attended key; layer intervals are end-exclusive. Distinguish `generated` from teacher-forced `answer`.
- Microcopy: label diversity and attention mass as descriptive; show teacher-forced knockout minus baseline in nats/token and total nats separately.
- A near-zero Korean effect is a finding with alternative explanations, not proof that the instrument failed or that the model never uses audio.
- Ledger coverage means a control exists in that experiment family; it does not prove an individually matched causal comparison (`src/run_ledger.py:198`).

## Implementation constraints

- Use one Vanilla Studio view backed by Molab's Python runtime. Reuse the notebook and single eager model; qualify the actual Molab editor/Studio session behavior. Do not replace Molab's managed GPU torch build or assume changing a kernel package updates the hosting server.
- Keep native form submission boundaries; no speculative JavaScript setter/RPC layer. `mo-value` is a read projection, not a generic control mutation API.
- Mount each cell/output target once per presentation; use CSS layout changes for responsiveness. Prefer named cells/aliases over numeric index selectors in source.
- Use tracked `studio/ctp49906-kr/` as `view_root`, since `.gitignore` excludes `__marimo__/`. Keep generated artifacts ignored and Studio owner records tracked.
- Single-file upload restores a generated, integrity-checked view bundle beside the uploaded notebook. Editable HTML/CSS/JS remain tracked separately; `scripts/build_studio_bundle.py --check` prevents drift. Scientific helpers/clips still come from the pinned repository checkout. This avoids depending on new view files being published before an uploaded script works.
- One PEP 723 configuration source in the Korean notebook; no new project packaging solely for Studio.
- Verified package pair: Studio 0.2.3 with marimo 0.25.0. Direct Molab Studio/server activation still needs hosted evidence; local marimo 0.23.14 was only the original baseline. A cell-level pip install does not establish that the managed host activated its extension.
- Screenshot acceptance: actual Molab notebook-plus-Studio split at 1440×900 and 1280×800, plus 390×844 fallback; verify inner-pane width, Korean wrapping, keyboard paths and persistent Run/status. Runtime call counts prove that navigation, thresholds and ledger edits do not trigger inference.
- Preserve existing backend numerical definitions and replay metadata validation. Do not auto-execute a Cartesian sweep of prompts, clips, frames, and layers.
- Include the verdict mutation/status cell as well as its form in the projection map. Define visible empty/unavailable states for stopped teacher-forcing cells; no indefinitely loading hidden projections.
- Retain Molab's dependency bootstrap and media compatibility shims. Fetch scientific code/assets from the pinned commit, restore the embedded view without overwriting edits, and open the view after initial setup. Running regular setup also runs view restoration through an explicit notebook dependency.
- Before live qualification, make the caption-cache identity include the requested token budget and verify one experiment at a time on the shared model. A busy indicator must not create new reactive dependencies into GPU cells.
- Show submitted results as immutable input/result snapshots. Browser history and a disk ledger are different lifetimes; do not promise that full results, uploads, or widgets survive kernel death.

## Open questions

- [ ] Verify Molab Studio activation, routes, iframe/WebSocket behavior and companion-file discovery from a fresh single-script upload. If unsupported, report the hosting limitation; grouped native notebook controls are an explicit alternative, not completed Studio integration.
- [ ] Separate Guide/Evidence URLs: qualify run-mode session identity and model count first. First release uses in-view panels regardless of edit-mode navigation behavior.
- [ ] Broader shared-control dispatch and a structured multi-rule editor: defer until the projected-native-control release is verified.
- [ ] Prepared static Guide: defer until output/widget portability is demonstrated. Current Korean artifacts do not include teacher-forced scoring results.
- [ ] Validated frame budget: reconcile current <=16 guidance with 2–32 form limits using device evidence; do not silently change recorded experiments.
