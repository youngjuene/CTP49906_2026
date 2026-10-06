# Korean Molab Studio verification

The intended product flow is a Korean notebook with an HTML Studio view beside
it. The local Studio implementation is verified; hosted personal Molab/GPU
integration remains an open release gate. Native controls submit to the same Python kernel; display tabs never submit
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

Prepare the view files before starting a compatible hosting server. A notebook
cell restoring files after the first request is insufficient for initial view
discovery. In a provisioned environment, use `python scripts/launch_studio.py`;
use `--prepare-only` for a host's pre-start hook. The launcher retains the
current interpreter/torch via `--no-sandbox`, requires marimo 0.25.0 and Studio
0.2.3, validates/builds the view, and starts an authenticated loopback server.
It does not configure Molab's managed server or its public proxy.

For personal Molab qualification, attach a GPU to an owned notebook and confirm
that the host exposes the actual Studio view with all three modes. Do not infer
support from successful package imports. For GPU-free checking, pass `--replay`
to the launcher, or set `USE_PRECOMPUTED`
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

## Upstream source audit and startup repair, 2026-10-06

Audited the official repository at
[`851a98b9fe61e18103d691a45ba8c8c6d2e8d325`](https://github.com/marimo-team/marimo-studio/tree/851a98b9fe61e18103d691a45ba8c8c6d2e8d325),
including middleware, extension entry points, compatibility metadata, authoring
APIs, view projections and hosting documentation. Runtime verification used
the installed Studio 0.2.3 release, not the unreleased main checkout.

### Findings and changes

1. **The HTML implementation follows Studio's state model.** It mounts 24
   named complete-cell hosts once inside `#app-shell`. Native marimo forms own
   input submission and Python dependencies own results. View JavaScript only
   changes visibility/navigation; it does not implement a second RPC channel.
2. **Single-file first launch had a preparation-order gap.** The view was
   restored by an ordinary notebook cell. Upstream
   [middleware](https://github.com/marimo-team/marimo-studio/blob/851a98b9fe61e18103d691a45ba8c8c6d2e8d325/packages/marimo-studio/src/marimo_studio/_server/middleware.py)
   sends a viewless run-mode notebook to the native app before that cell runs.
   `scripts/launch_studio.py` now reads only literal bundle data through AST,
   restores files, validates and builds, then launches the server. It preserves
   edited view files and never executes the notebook to extract the bundle.
3. **Package installation is not server activation.** Upstream
   [package metadata](https://github.com/marimo-team/marimo-studio/blob/851a98b9fe61e18103d691a45ba8c8c6d2e8d325/packages/marimo-studio/pyproject.toml)
   pins marimo 0.25.0 and registers ASGI middleware and a kernel lifespan hook.
   The launcher fails on incompatible versions, retains authentication, and
   reuses the provisioned GPU stack. It does not run pip or replace torch.
   The notebook's PEP 723 torch pins and legacy `requirements.txt` describe
   different installation paths; they must not replace Molab's working GPU
   stack merely to run a view. This launcher deliberately uses `--no-sandbox`.
4. **A missing toolbar alone does not establish missing Studio.** The official
   [configuration](https://github.com/marimo-team/marimo-studio/blob/851a98b9fe61e18103d691a45ba8c8c6d2e8d325/docs/reference/configuration.md)
   supports a native editor at `/` with Studio at `/studio/`. The host must
   route and authenticate those paths. The earlier menu-only inference is
   corrected in the Korean guide, README and generated notebook notice.
5. **The hosting guide does not establish hosted Molab support.** The official
   [marimohub guide](https://github.com/marimo-team/marimo-studio/blob/851a98b9fe61e18103d691a45ba8c8c6d2e8d325/docs/guide/marimohub.md)
   describes a configured marimohub deployment. Its persistent `studio/`
   location is appropriate here, but its server setup cannot be assumed for
   a student's existing molab.marimo.io session.

### Fresh verification of this repair

- Python 3.12 CPU suite: **351 passed**, including six launcher regressions.
  A lone `.py` changes from official `needs-view` status to a discovered
  `explore` view without importing or executing that notebook.
- Embedded bundle check, strict marimo check, official Studio static validation,
  production build and dependency doctor passed. Static validation found
  62 cells and 24 projections, with no issues.
- Official saved-workspace API runtime validation passed with no issues in
  CPU replay. CLI runtime validation initially tried an isolated uv environment;
  the successful check used the provisioned interpreter through the public API.
- Chrome opened a fresh local fixture that initially had no Studio directory.
  After the launcher prepared it, the first authenticated browser connection
  opened the split Studio view, reached `ready`, and mounted all 24 hosts.
  An input draft survived all three top-level tabs; replay inference was disabled.
  Submitting a nonexistent run ID changed the Python-produced feedback to the
  expected rejection. This verifies a native form/result round trip, not GPU work.
- A second fresh fixture passed the same initial discovery in `run` mode:
  authentication first, then the HTML view with 24 hosts and `ready` status.
  The launcher explicitly passes `--token` in both modes because marimo's
  `run` command otherwise defaults to unauthenticated access.
- Observed browser console errors came from installed Chrome extensions
  (share modal and MetaMask); no application-origin error appeared in that sample.

### Hosted Molab evidence remains separate

After installing Studio in the owned GPU session, its terminal reported Studio
0.2.3/marimo 0.25.0 while the managed editor still reported marimo 0.25.1.
A separate authenticated Studio process answered inside the sandbox, but a
usable route from the owned Molab page was not established. The official app
link opened a native app in a different sandbox. The direct documented
`/studio/` route attempt was blocked by the browser client, so it is **not**
evidence of a server 404 or proof that Studio is absent. No bypass was used.

The personal Molab gate is still: actual Studio HTML input → fresh GPU compute
in that student's session → changed HTML result, followed by download/reopen.
The current repair makes the startup hook reviewable and tested; it does not
complete or replace that hosted integration test.
