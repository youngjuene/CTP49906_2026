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
