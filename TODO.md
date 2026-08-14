# Open items

Written after the Week-0 explorability pass. Everything here is either
*unverifiable outside molab* or *deliberately deferred pending classroom
evidence* — nothing in this list is known-broken.

## What is already verified, so you do not re-litigate it

Run locally on an RTX 3090 against the real Qwen2.5-Omni-3B weights:

- A full guided run completes in **~30 s, peak 13.70 GiB**.
- Live attention numbers reproduce the committed artifacts exactly
  (layer 0 `video` mass: baseline `0.0094`, knockout `0.0000`).
- The probe grid's live CSV reproduces the committed anchors
  (Layer_34: 6 unique, 245 junk, 92 matching final).
- **F1's acceptance criterion holds:** silent control `+0.0005` nats/token vs
  default clip `−0.4637` nats/token.
- The run ledger survives a process restart through `lab_log.jsonl`.

Both notebooks also execute headlessly in CI (`app.run(defs=…)`), and the
dataflow graph, widget reactivity invariants, and notebook lint are asserted by
tests. See the README's testing section.

---

## 1. The molab verification run — the only real blocker

Nothing below can be checked outside molab, because molab differs from a local
box in three ways that matter: a Blackwell GPU, a locked-down iframe with a CSP,
and a kernel that pip-installs at runtime.

Work through these **in order** — each one invalidates more of what follows.

- [ ] **Does `ProbeGrid` render at all?** This is the one genuinely open
      question. The evidence that anywidget works in molab is *empirical*
      (`TextCompare` renders) rather than mechanistic — and `mo.iframe` uses the
      same virtual-file transport yet comes up blank. If the grid does not
      render, fall back to the ~12-line matplotlib `imshow` of the junk mask,
      which carries most of the lesson; do not spend time debugging the ESM
      first.
- [ ] **Does any GPU cell re-run when a widget is touched?** Drag the grid, drag
      both Δ thresholds, and watch the cell run-indicators. The static
      invariants are asserted in `tests/test_notebook_graph.py`, but only a live
      kernel proves the runtime behaviour. Expect: no spinner, no re-run.
- [ ] **Does the runtime installer succeed?** `_ensure_packages` must install
      `wigglystuff==0.5.21` and `anywidget` into molab's kernel. It runs
      `subprocess.run(..., check=True)`, so a Python < 3.11 kernel raises here
      rather than degrading.
- [ ] **Does the PyAV `read_video` shim still fire?** It only matters if molab's
      torchvision is ≥ 0.23. Locally torchvision 0.21 has `read_video`, so this
      path has never executed.
- [ ] **Re-check VRAM on molab's GPU.** 13.70 GiB is the local peak with one
      shared model; confirm it holds there.

## 2. Plan after that run

- [ ] **Teach a session before building another widget.** `DeltaLensStrip`,
      `KnockoutRuleRibbon` and `ProbeGrid` phase 2 (the entropy/margin overlay)
      are specified and deliberately unbuilt. If students do not miss the pinned
      comparison once the ledger exists, most of them never need building.
      Notebook A already has six measurement surfaces; instruments were never
      the constraint.
- [ ] **Run the jlens notebook on a real GPU.** It has only ever run against
      stubs (`tests/test_jlens_notebook_smoke.py`). The weights are cached
      locally, but jlens needs `transformers>=5.13` while avllm pins `4.52.0`,
      so it needs its own environment. Unverified live: the model load, the
      reference-lens download, the guided demo's forward passes, and
      `compute_slice`.
- [ ] **Decide whether avllm should prefer a local checkout.** Today its setup
      cell always clones from GitHub and imports `src/` from that clone, so local
      edits are invisible until pushed — while `jacobian-lens/CTP49906_jlens_molab.py:135-139`
      prefers the checked-out source. Adopting the same preference removes a
      GitHub round-trip per iteration, but the unconditional hard-sync to
      `REPO_REF` is exactly what makes a classroom run deterministic. This is a
      course-design call, not a cleanup.
- [ ] **Pin the semester.** `REPO_REF = "main"` in both notebooks. Before the
      course starts, cut a tag and point `REPO_REF` at it so later pushes cannot
      change what students execute mid-course (PRD risk R7).
- [ ] **Tick the PRD checkboxes**, including `§6 Plan in effect: ☐ A ☐ B` —
      that one retires risk R4 with a single character, but which plan is in
      effect is yours to decide.

## 3. Known-fragile, worth watching

- **marimo name mangling.** A `lambda` default over an `_`-prefixed name can
  resolve against a different cell's prefix and raise `NameError` at runtime;
  the file still parses and the graph still builds. It shipped once, reachable
  only after a student had fitted a lens. `tests/test_notebook_lint.py` bans the
  construct — keep that test.
- **`--write_probe_distributions` is now opt-in** in
  `scripts/generate_precompute.py`; it writes ~15 MB of JSON that no cell reads.
  Turn it on only when building the phase-2 sidecar.
- **Regenerating precomputed artifacts** must keep
  `summarize_attention(..., decode_only=True)`. The live notebook path passes it
  too; if the two ever disagree, the replayed and live heatmaps silently
  describe different quantities.
