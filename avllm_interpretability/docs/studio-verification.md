# Korean Molab Studio verification

The supported product flow is a Korean notebook with an HTML Studio view beside
it. Native controls submit to the same Python kernel; display tabs never submit
an experiment. The main notebook carries a generated, integrity-checked bundle
of the view so a single-file upload does not depend on unpublished view files.

## Updating the interface

Edit `studio/ctp49906-kr/explore/`, then run:

```bash
python scripts/build_studio_bundle.py
python scripts/build_studio_bundle.py --check
python -m pytest -q
marimo-studio validate explore --target CTP49906_avllm_molab_kr.py
marimo-studio view build explore --target CTP49906_avllm_molab_kr.py
```

Use the declared Studio/marimo versions for the last two checks. The build
generator copies only marked delivery code; the frontend remains authored in
separate files. Do not edit its encoded payload by hand. Restoring the bundle
updates owned, unedited files and refuses to replace local edits.

## Runtime checks

Use a fresh Molab session with the uploaded `.py`, attach a GPU for live work,
and run setup. Confirm the Studio toolbar/view opens beside the notebook and
all three modes are reachable. For GPU-free checking, set `USE_PRECOMPUTED`
to `True` in the configuration cell; the developer-only environment variable
`CTP49906_REPLAY=1` provides the same setting for automated verification.

Check the actual split workspace, not only a full-window view:

- Main inputs and Run are reachable at 1440×900 and 1280×800.
- Panel switching and resizing preserve draft form state.
- Replay shows saved Korean examples and disables inference submission.
- Idle teacher-forcing panels settle without ancestor-stopped errors.
- The native verdict form updates its ledger handler and worksheet output.
- Local display interactions do not reload the model or rerun inference.
- Single-file bootstrap is repeatable and preserves edited view files.
- Inspect browser errors, failed requests and Studio projection diagnostics.

The checked-in browser runner can verify a disposable replay editor session:

```bash
python scripts/verify_studio_browser.py --url http://127.0.0.1:2786 \
  --notebook /path/to/disposable/CTP49906_avllm_molab_kr.py
```

Launch that development server with `CTP49906_REPLAY=1`. `--notebook` uses
marimo's documented pairing API to execute initial cells in the connected
fixture; it refuses a non-replay server. Without it, initialize the notebook
first. `--browser-executable` selects an already installed Chromium binary.
The runner checks real native controls, preserved draft values, disabled replay
submissions, and fully visible Run buttons at both desktop sizes.

Worksheet downloads deliberately use marimo's lazy download callback. An eager
relative virtual-file link returned HTTP 404 inside Studio during browser
verification; the callback delivers the captured worksheet through the native
widget protocol without that route dependency.

Live qualification also requires one valid run for each method and the silent
control, followed by failure/retry checks on the target GPU. A cached caption
must be regenerated when the token budget changes. A new Molab session is not
a durable disk backup; download the worksheet before it expires.

## Evidence boundary

Unit/graph/replay tests, real Studio build validation, and local browser checks
are distinct from a direct authenticated Molab/GPU run. Record which were
actually performed. A native-notebook fallback is not a completed Studio
integration, and successful Python imports alone do not prove that the hosting
server activated Studio's extension.

## Implementation verification, 2026-10-02

- Tested Studio 0.2.3 and marimo 0.25.0 with Python 3.11.11.
- Full test suite: **166 passed**, including English/Korean replay and actual
  teacher-forcing cell call-boundary/cache tests with a CPU fake backend.
- Official Studio static and isolated replay-runtime validation: passed, no
  projection errors. The view mounts 24 named notebook cells.
- Real Chromium split workspace: passed at 1440×900 and 1280×800; each mode's
  Run button is fully visible, drafts survive navigation, replay submissions
  are disabled, and browser/Studio error diagnostics are empty.
- Narrow standalone preview: 390×844, ready with no document overflow.
- Isolated browser ledger test: a newly submitted annotation persisted, and
  the actual downloaded worksheet contained that exact annotation.
- Embedded view restore, edited-file preservation, safe bootstrap paths and
  pinned public commit availability were checked.

The loop found and fixed idle ancestor-stop errors, an empty projected verdict
handler, cramped controls, a missing caption-cache token budget, an eager
download route returning 404, and directory-name inference based on suffixes.
It does not constitute an authenticated Molab session test or live GPU benchmark.

## Main merge verification, 2026-10-02

- Integrated with main at `5e9cb63`, retaining its bounded media decoding,
  input validation, token budgets, matched-control ledger, and evidence exports.
- Full combined test suite: **331 passed**, including the local-source provenance
  regression. Sources without Git metadata record a content-based helper digest.
- Strict marimo checks and official Studio replay-runtime validation passed;
  runtime validation reported no issues.
- Chromium split-workspace checks passed at 1440×900 and 1280×800, including
  the submitted caption-token slider, draft preservation, disabled replay Run
  buttons, and empty browser/Studio diagnostics.
- Both actual downloads (`lab_log.md` and `lab_evidence.json`) succeeded from
  the Studio evidence panel. The JSON included provenance, runs, and schema version.
- Embedded bundle consistency and focused Ruff checks passed.

These merge checks used local CPU replay. Authenticated Molab and live GPU
execution remain separate qualification steps.


## Classroom interface verification, 2026-10-06

Helper revision: `409db993bcc041dc63f6929625916e7bfd430328`.
Python 3.12, marimo 0.25.0, marimo-studio 0.2.3. The earlier dated
checks above describe their own source/runtime; they are not fresh release evidence.

- CPU suite: 341 passed. Includes real notebook execution-boundary checks,
  last-result ID/config consistency, threshold value round-trip, host theme
  resolution and exact UTF-8 download callback payloads.
- Official marimo check and Studio static validation/build passed (24 projections).
- Chrome computer-use tested the actual split Studio and standalone view.
  At 1280×800, the split view was 617px wide and the result frame was 300px
  high including borders. At 390px, page width equalled viewport width;
  42px tabs fit the 55px header.
- An isolated display fixture replayed the saved 17 GPU runs from October 4.
  Its forms could not start inference. TF run b49a9ede kept its recorded
  -0.454 nats/token and -9.08 nats total. Threshold 1.52 selected 2/17 units
  (-8.40 nats, 87% of negative mass); 0 selected 9/17 (-9.68, 100%);
  7.14 selected none. Re-entering 1.52 preserved it and ArrowUp gave 1.53.
  Final review also caught an empty number-input exception. Both the explorer
  and guide now show an input prompt on clear and recover 2/17, -8.40 nats,
  87% after entering 1.52 again; confirmed in Chrome with the actual display cells.
- Expanded TF details followed the full result height with an 8px gap.
  Long token strips wrapped without horizontal overflow.
- A separate exact-source CPU replay used a copy of the 17-run ledger.
  Draft values survived navigation. Invalid-ID feedback was fully visible
  (112px frame, 100px cell, no clipped content); a valid annotation updated
  the ledger and reported a saved session file. The original evidence was untouched.
- Probe foreground became #1f2328 on the light Studio surface, and Home
  changed the visible layer summary. Both theme resolution directions are unit tested. An actual-CSS Chrome
  fixture with conflicting ancestor themes returned #1f2328 for a light widget
  and #e7eaee for a dark widget; the widget theme now wins the CSS cascade.
- The bundled 10-second MP4 loaded with readyState 4, played and sought in
  Chrome after conversion to an inline video URI.

### Follow-up: authenticated Molab GPU, 2026-10-06

An immutable Mirror of `a09713f` was forked into an owned notebook. Run all
completed on an RTX Pro 6000 Blackwell (95 GiB reported), torch 2.11.0+cu130,
marimo 0.25.1, helper revision `409db99`. All 15 planned interactive conditions
completed. A keyboard-focus correction produced one additional recorded run;
the 16 records were retained. English-original audio knockout measured
-0.453897 nats/token, while its matched silent control measured +0.003223.
These are observed values for this runtime, not acceptance targets.

The same TF inputs reproduced the same ID without adding a ledger row.
Threshold 0 selected 9/17 displayed units; the maximum selected 0/17. Clearing
either threshold showed the input prompt, and restoring 1.52 kept that exact
interactive value. Invalid prompts, missing uploads, empty layer ranges and
malformed/inert advanced rules were refused; valid inputs recovered. An empty
claim was refused and a complete QA claim/verdict/rival was saved.

The official copy previews preserved the full JSON and Markdown. Reopened
local copies passed 444 content/coverage checks, including matching Markdown
and JSON, all 15 conditions, arithmetic, token alignment and annotation. They
are **preview copies**, not proof of a native browser download. Download-click
and `downloadMedia` completion checks timed out; this gate remains open.

Opening the owned notebook in another tab kept the same sandbox and all 16
records. A compute change created a different sandbox and exposed a startup
failure: Molab restored the project files without `.git`. Setup now compares
those files against an isolated checkout of the pinned revision, restores
missing tracked files and Git metadata, and preserves additional files. It
refuses differing files and symlinked paths. The live failed sandbox then
reported `checkout-restored`. Its ledger started with 0 records, so this is
startup recovery, **not durable experiment storage**. Keep both exports before
changing compute or ending a session.

The final recovery copy was run again on a fresh RTX Pro 6000 sandbox. Setup
reported `checkout-restored`; the GPU banner confirmed the device and all four
guided completion signals returned without cell exceptions. The final local
suite passed 345 tests. During compute changes, the RTX selection label could
remain visible while `torch.cuda.is_available()` was false. Explicitly changing
the selection from None to RTX before saving allocated a new GPU sandbox; use
the runtime GPU banner as evidence, rather than the selection label alone.

### Remaining release gates (updated 2026-10-06)

The owned Molab GPU notebook reported `marimo-studio=MISSING`; its Present as
menu offered app view, Vertical, Grid and Slides. Fresh native GPU QA passed,
but no Molab Studio/GPU pass is claimed. A hosting server must activate Studio;
installing/importing a package or restoring view files alone does not do that.

Native download callbacks return distinct, complete Markdown/JSON payloads.
Chrome download-completion events did not arrive in this check, so actual
file download/reopen remains unverified. The fresh JSON/Markdown copy previews
contained all 16 runs and passed content validation. Native downloads are retained; no alternate custom download
protocol or browser security setting was introduced.

Both Mirror refresh and the owned notebook's new sandbox started an empty
ledger. The interface states the current-session scope and asks students to
export before leaving; an owned/forked notebook does not guarantee saved runs.
These gates prevent calling this an end-to-end Molab classroom release.
