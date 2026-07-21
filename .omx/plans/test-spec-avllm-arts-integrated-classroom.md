# Test specification: Counterpoint Lens arts-integrated classroom redesign

**Companion plan:** `.omx/plans/prd-avllm-arts-integrated-classroom.md`  
**Purpose:** define evidence required before the redesigned treatment notebook and
its research companions can be called classroom-ready or research-ready.

## 1. Verification strategy

Use four independent gates:

1. **Computation correctness:** deterministic identity/transforms, compact metrics,
   record linkage, and export validation.
2. **Notebook behavior:** Marimo reactivity, button-gated expensive work,
   live/cached parity, failure recovery, and clean student hierarchy.
3. **Research-design integrity:** condition fidelity, comparison-arm isolation,
   outcome/instrument traceability, human-audience blindness, and honest claim
   labels.
4. **Classroom fitness:** timing, accessibility, bilingual parity, privacy, GPU
   envelope, and portfolio completion by representative users.

Automated tests can prove contracts and detect leakage; they cannot prove construct
validity, ethics approval, artistic quality, or causal identification. Those require
the human review/pilot gates listed below.

## 2. Test environments and fixtures

### Environments

- **CPU CI:** Python version pinned by the project; no CUDA/model download/network.
- **Cached notebook:** saved course replay from committed artifacts, no GPU.
- **Live GPU:** Molab-equivalent CUDA environment with a 24 GB GPU and immutable
  model revision.
- **Rendered accessibility:** exported/served Marimo app in a current Chromium-class
  browser, with keyboard-only and accessibility-tree inspection.
- **Study dry run:** synthetic pseudonymous participants and non-student media only.

### Required fixtures

- Tiny deterministic video with known frame count/color sequence and audio impulse.
- Same-duration donor audio with known impulses/tones.
- Silent audio, neutral visual stream, corrupt file, oversize media metadata, and
  unsupported stream/container cases.
- Feasibility fixtures for video-with-audio, video-without-audio, audio-only, and
  model-input requests that truly omit audio or video.
- Congruence/pretest codes independent of technical operation, including congruent
  and incongruent donor swaps within pairing blocks and failed manipulation checks.
- Synthetic logits with analytically known softmax, entropy, margin, top-k, and
  target rank; include ties, extreme values, NaN/Inf rejection, and large vocab.
- Matching and mismatching token-layout fingerprints.
- Valid/invalid Teaching and Research configuration fixtures.
- Independent permission-scope combinations (analysis, classroom sharing,
  reproduction/quotation, future reuse) and expired/revoked authorization.
- V1/V2 lineage; complete/incomplete/duplicate/orphan/withdrawn records; audience
  records with blindness true/false; malicious CSV/HTML-like text.
- Separate rich private portfolios and expected minimized research projections;
  audience presentation packets with included/excluded creator/model fields and
  duplicate respondent/session identities.
- Instrumented browser/kernel transport and fake Git/model/package/instructor
  endpoints that record destinations, methods, and whether payloads contain fixture
  artifact/process data.
- Legacy `precomputed/` artifacts and new versioned artifacts.
- Valid/invalid immutable course release profiles shared across both arms.
- English/Korean content-key inventory and representative long-string cases.

No fixture contains actual student work or direct identifiers.

## 3. Unit test matrix

### 3.1 Media identity and variants

Target: `src/playground_clips.py`, planned `src/stimulus_variants.py`, and tests.

| ID | Test | Expected evidence |
|---|---|---|
| U-M01 | Same bytes under different filenames | Same content SHA and artifact identity input; filename is not identity |
| U-M02 | One-byte content change | Different SHA/artifact ID |
| U-M03 | Normalize recipes with reordered mapping keys | Same canonical recipe and stimulus ID |
| U-M04 | Change donor/offset/condition/schema version | Different stimulus ID |
| U-M05 | Duration-match shorter/longer donor audio | Exact protocol-length output by deterministic trim/zero-pad; operation recorded |
| U-M06 | Positive and negative temporal offset | Audio impulse moves by signed offset with zero-fill and no circular wrap; video unchanged |
| U-M07 | Audio silence control | Duration-matched silence, video stream/hash preserved under declared container tolerance; never typed as modality omission |
| U-M08 | Neutral video control | Documented neutral/black visual stream, audio samples preserved under declared tolerance; never typed as modality omission |
| U-M09 | Audio swap | Donor provenance present; target video and duration preserved; donor normalized to processor contract |
| U-M10 | Repeated transform | Byte-identical or decoded-signal-identical output per selected codec policy and identical manifest checksum |
| U-M11 | Unsafe/corrupt input | Rejected before allocation/model call with actionable message; prior state unchanged |
| U-M12 | Manifest round trip | Serialize/validate/load without field loss; unknown future fields handled per version policy |
| U-M13 | Operation versus congruence | Technical operation never assigns congruence implicitly; pair/block/coder/pretest provenance and check are required |
| U-M14 | Edit-decision manifest | Common editor/export preset, sources, trims/order, mix/levels, accessibility, and assistance round-trip identically across arms |

The feasibility spike must choose whether equality is byte-level or decoded
audio/frame-level for each codec/container; tolerances must be written before the
test is accepted.

### 3.2 Distribution metrics

Target: planned `src/probe_metrics.py`, `src/logitlens_experiment.py`,
`src/teacher_forcing.py`.

| ID | Test | Expected evidence |
|---|---|---|
| U-P01 | Known three-class logits | Entropy in nats, optional `H/log(V)`, and top-1 minus top-2 log-probability margin match a high-precision reference |
| U-P02 | Constant shift / stable softmax | Entropy, ranks, and probabilities invariant within tolerance |
| U-P03 | Extreme logits in lower precision | Stable log-softmax; no accidental overflow/NaN |
| U-P04 | Tied top logits | Deterministic documented ordering and zero margin |
| U-P05 | Target set absent/present/tied | Correct minimum rank/log probability and explicit missing state |
| U-P06 | Streaming summary vs full reference on small vocab | Top-k, entropy, margin, and rank agree within declared numeric tolerance |
| U-P07 | Large-vocab/multi-layer path | Only configured layers/positions are sliced, projection is chunked, full logits/outputs are released and absent from serialization, and peak host/GPU memory stays under the release-profile limit |
| U-P08 | Modality index mapping | Audio/video/answer/unknown positions follow processor metadata; unknown is never guessed |
| U-P09 | Layout mismatch | Position-wise comparator refuses; aggregate/bin comparator emits warning and mode |
| U-P10 | Layout match | Layer/position/modality trajectory aligns stable identities in deterministic order |
| U-P11 | Probe vs final result schema | Distinct measurement-kind enum/caveat: raw probe dispersion versus teacher-forced answer-distribution dispersion; labels cannot be interchanged |
| U-P12 | Metric version contract | Units, normalization, margin/tie/rank rules, dtype, vocab/support, probe transform, and target-set version are required; unsupported JS/calibration/free-generation claim paths are rejected |

### 3.3 Session, reflection, audience, and export records

Target: planned treatment-neutral `curriculum_common/production_manifest.py`,
`session_records.py`, `audience_packets.py`, and `portfolio_export.py`.

| ID | Test | Expected evidence |
|---|---|---|
| U-R01 | Commit run | UUID, command nonce, local UTC/event/elapsed time, prediction, initial explanation, exact parameters, and model/stimulus/artifact/schema IDs present |
| U-R02 | Reflect on known run | Append-only reflection links to run and snapshots initial explanation |
| U-R03 | Edit after submit | New revision record with parent link; original serialized bytes unchanged |
| U-R04 | Orphan/duplicate/missing prediction or initial explanation | Validator reports exact record/field and blocks research-ready export |
| U-R05 | Withdraw record | Withdrawal state retained; export follows configured omission/tombstone policy without rewriting history |
| U-R06 | Teaching private portfolio | User-visible private projection may retain hashes/local history but has no research participant/destination claim and discloses the Molab boundary |
| U-R07 | Invalid/missing research config | Mode falls back to Teaching; collection-only fields/actions unavailable |
| U-R08 | Valid approved config | Protocol/schema versions and only allowlisted fields included |
| U-R08a | Permission combinations | Each use is independently allowed/blocked; no scope implies another; revoked scope is enforced |
| U-R09 | Audience blindness | Response cannot be committed after creator/model reveal without an explicit protocol deviation flag |
| U-R10 | Audience count | Triadic completion remains incomplete below two valid independent responses |
| U-R11 | Formula/markup injection text | CSV and report export neutralizes executable spreadsheet formula/unsafe rendering while preserving content |
| U-R12 | Bundle reproducibility | Same logical records produce same canonical files/checksums except declared UUID/time fields |
| U-R13 | Export/re-import | Supported records, links, tags, omissions, and version data survive round trip |
| U-R14 | Delete/reset | Molab-session state and temp derived media removed; checked-in assets and unrelated files untouched |
| U-R15 | Pseudonym/timing export | Export says pseudonymous, replaces raw content hashes with study IDs, strips direct/file/device identifiers, and omits or coarsens local wall-clock time unless protocol-allowlisted |
| U-R16 | Exactly-once reducer | Reapplying one command nonce, attaching the same result, or replaying event history is idempotent; conflicting payload reuse is rejected |
| U-R17 | Private-to-research projection | Pure versioned projection replaces raw hashes with study IDs, strips unapproved fields, emits a redaction report, and leaves the private portfolio byte-unchanged |
| U-R18 | Audience packet allowlist | Presentation packet excludes creator/model fields, validates sharing scope/checksum/reveal state, and rejects duplicate respondent-session identity or unsafe import |

## 4. Integration test matrix

### 4.1 Media → processor → model-input invariants

- **I-M01:** Create every course operation from a pairing block, run existing upload
  preflight, and assert duration/stream/recipe/provenance plus independent
  congruence/manipulation-check facts.
- **I-M02:** Process original/swap/offset/silence/neutral variants and true audio-
  omitted/video-omitted requests through their exact processor paths; store layout
  fingerprints and permit position comparisons only for identical layouts. If
  either omission path is unsupported, assert that modality-ablation claims and
  research-ready status are disabled.
- **I-M03:** Demonstrate that direct-edge knockout changes an attention mask/hook but
  not the media manifest, signal controls change presented media, and modality
  omission changes processor modality presence. Result schemas must carry different
  intervention types.
- **I-M04:** Cache keys include content hash, normalized recipe, processor/model
  revision, prompt, frame settings, and relevant analysis parameters. Changing any
  semantic input invalidates the result; label-only changes do not.
- **I-M05:** A transform/preflight exception leaves V1/V2 manifests, forms, and prior
  run records intact; retry after correction succeeds.
- **I-M06:** Build the research design matrix from the stimulus manifest and declared
  contrasts; fail research-ready status when congruence, operation, donor/source,
  block, or order are perfectly aliased or a manipulation cell is absent.

### 4.2 Live and precomputed result contract

- **I-P01:** For the immutable reference run, live and generated precompute outputs
  have the same schema, condition IDs, layer/position domains, and result labels.
- **I-P02:** Numeric live/replay summaries match within recorded hardware/dtype
  tolerances; decoded token comparisons use documented normalization.
- **I-P03:** Legacy artifacts load with explicit `legacy_argmax_only` status and
  `not measured` uncertainty fields; no values are fabricated.
- **I-P04:** Teacher-forced baseline/knockout Δ log-likelihood remains unchanged by
  adding final distribution summaries on frozen fixtures.
- **I-P05:** Attention summaries preserve existing generated-query/key-region
  semantics and never populate causal-effect fields.

### 4.3 Notebook reactive workflow

- **I-N01:** Changing a form before `Run` does not execute the model; committing
  snapshots all displayed values and produces exactly one run ID.
- **I-N02:** Subsequent form edits do not mutate a displayed/serialized prior run.
- **I-N03:** `Reflect` is disabled until a successful or explicitly failed run is
  committed, then links to the selected run rather than the latest mutable form.
- **I-N04:** Registering V2 updates version selectors and comparisons but preserves
  V1 outputs and lineage.
- **I-N05:** Audience/model outputs remain hidden until audience response/import;
  revealing them sets a state that cannot be reversed to claim blindness.
- **I-N06:** Advanced cells stay collapsed and are not dependencies of required
  checkpoints.
- **I-N07:** Switching live/replay mode cannot show stale live results as replay or
  vice versa; every result has a visible provenance badge.
- **I-N08:** Kernel restart plus portfolio re-import restores supported artifacts,
  run/reflection links, and completion state without rerunning the model.
- **I-N09:** After committing a run/reflection, invalidate an unrelated upstream
  dependency and trigger repeated downstream execution; the command nonce/event
  count/result linkage remain exactly once. Repeat after event-log reload.
- **I-N10:** Cells render exclusively from committed reducer state; editing a form,
  retrying a failed side effect, or switching a display filter cannot attach a
  result to the wrong command/run.

### 4.4 Cross-arm and instrument integrity

- **I-C00:** Before treatment UI implementation begins, an executable/static route
  manifest proves mirrored activity/time, neutral production shell/editor/export,
  help policy, audience handoff, instruments, cognitive-demand rationale, and
  contamination boundaries have versioned owners and approval status.
- **I-C01:** A shared fixture submitted through treatment and comparison surfaces
  yields identical production, audience, and common instrument schemas.
- **I-C02:** Static import and rendered-text scan of the comparison surface finds no
  treatment-only modules or terms (`logit lens`, layer probe, attention knockout,
  internal ablation) before the configured post-measure boundary.
- **I-C03:** Time/task/source/output/facilitator-contact configuration diff is empty
  for matched time, hardware, asset budget, accessibility support, output, and
  contact fields; deviations require a protocol-deviation record.
- **I-C04:** Condition assignment is absent from student portfolio and treatment
  notebook UI unless the protocol explicitly authorizes later debriefing.
- **I-C05:** Rubric scoring export records instrument/rubric version, blinded rater
  pseudonym, item scores, missingness, and adjudication without altering artifacts.
- **I-C06:** EN and KO administration map to the same content/instrument identifiers;
  translation status and permission are recorded.
- **I-C07:** `curriculum_common` has no treatment imports; the treatment and
  comparison surfaces depend inward on the neutral package, never on each other.

## 5. End-to-end scenarios

### E2E-1 — Required treatment path, live GPU

1. Start from a clean checkout in Teaching mode.
2. Register intention and accessible V1; verify Molab-session preview, common edit
   manifest/export preset, and private-versus-research projection disclosure.
3. Complete the guided demo with one shared reference model instance.
4. Predict and run one duration-matched paired manipulation.
5. Inspect probe trajectory and teacher-forced answer-distribution metric with
   distinct caveats.
6. Import two valid responses from the separate blind audience surface, then reveal
   creator/model readings.
7. Commit revised explanation and rival account.
8. Register V2, rerun the fixed comparison, and export/re-import portfolio.

Pass: all classroom-column criteria hold; no kernel restart, duplicate/stale event,
lost record, unbounded activation/logit retention, release-profile host-RAM breach,
or >24 GB VRAM peak.

### E2E-2 — Complete guided learning without GPU

Start with `USE_PRECOMPUTED`/saved replay and no CUDA/network. Complete orientation,
guided observations, example process-data critique, and all reflection checkpoints.
Unsupported personalized model runs are disabled with an explanation; no 8 GB model
download occurs. The exported practice portfolio clearly says `saved course replay`
and `teaching-only example`, not student research evidence.

### E2E-3 — Failure and recovery

After V1 and one valid run, attempt corrupt media, an unsafe decoded-memory file,
invalid transform, CUDA OOM simulation, and interrupted export. Each failure creates
at most one explicit failed-run event, preserves prior state, cleans temp files, and
succeeds on corrected retry. No partial portfolio is labeled valid.

### E2E-4 — Triadic blindness and disagreement

Use an artifact whose creator tags, two audience interpretations, and model tags
partly disagree. Generate a permission-valid presentation packet that excludes
creator/model fields; collect distinct blinded respondents externally; safely
import; assert reveal/deviation state, open-text retention, neutral disagreement
matrix, contextual prompts, and absence of automated human-answer substitution.

### E2E-5 — Production-matched arm rehearsal

First pass I-C00 before treatment UI freeze. Then two facilitators run treatment and
comparison scripts with synthetic students using the same default editor/export
shell. Capture timings, hardware/asset/accessibility support, prompts/help,
V1/V2/audience steps, common measures, assistance, and disruptions. Pass only if
matching fields/exposure are within protocol tolerances and comparison participants
see no interpretability material before post-measures.

### E2E-6 — Research-mode governance dry run

With a synthetic approved configuration, consented pseudonym, and mock approved
instructor destination, create/export/withdraw a session. Repeat with missing,
expired, wrong-
version, and over-permissive configs. Only the valid case exposes collection fields;
an in-notebook acknowledgement is never labeled consent; separate analysis,
sharing, quotation, and reuse scopes are obeyed; all exports match the allowlist and
audit report. This tests software enforcement, not ethics approval.

### E2E-7 — Accessibility and bilingual parity

Keyboard-only users complete the required route and operate every form/dialog.
Screen-reader/accessibility-tree review finds labels, heading order, result status,
and text alternatives. Compare EN/KO content-key inventories and have qualified
reviewers inspect meaning, terminology, line expansion, accessibility instructions,
and instrument parity.

### E2E-8 — Modality omission and alignment limits

Run original, silence, neutral-video, audio-omitted, video-omitted, and edge-knockout
conditions through the frozen processor/model revision. Verify actual modality
presence/absence, record each distinct operation, and allow only compatible
aggregate/final-output comparisons when layouts differ. If either omission path is
unsupported, verify the release cannot call the condition “modality ablation” or
claim research-ready RQ3 coverage.

### E2E-9 — Molab deployment/data-egress boundary

Instrument browser uploads, kernel HTTP/storage clients, Git checkout, dependency
resolution, model acquisition, private download, permission-authorized audience
exchange, and mock instructor export. Verify/display the browser → Molab boundary;
permit expected code/weight reads, user-initiated device download, and only the
allowlisted `exchange_artifact_id` presentation packet; ensure Teaching mode cannot
send fixture data to Git/model/package/instructor endpoints; and ensure Research
mode can send only the minimized study-ID projection to its approved destination.
Reconcile behavior with the platform-retention statement in instructor/student docs.

## 6. Observability and classroom rehearsal evidence

### Per-run session diagnostics

- run/stimulus/artifact/condition/model/schema identity;
- state transition (`prepared`, `running`, `succeeded`, `failed`, `reflected`,
  `withdrawn`) and validation errors;
- elapsed time, peak allocated/reserved VRAM, peak host RSS, configured activation/
  chunk sizes, cache hit/miss, result provenance;
- token-layout comparison status and reason;
- record counts: predictions, runs, reflections, orphan/missing items, audience
  readings, V1/V2 lineage;
- operating/consent configuration status without direct identifiers.

Diagnostics are visible in the Molab session/private download and excluded from the
research projection unless allowlisted. Instrument HTTP/storage clients and assert
that default Teaching/cached modes make no automatic or unapproved outbound request
containing student artifacts/process/audience records. Explicitly distinguish the
three authorized egress classes (private device download, permission-approved
audience packet, approved Research-mode projection) from expected code/dependency/
model reads and all other student-data egress.

### Rehearsal report

For every release candidate, record:

- clean-start, model-load, guided-demo, each condition, and total required-route
  duration distributions;
- peak VRAM, peak host RAM, activation/chunk configuration, and OOM/retry behavior;
- number/type of student help interventions and reactive-state failures;
- completion/missingness for every required evidence field;
- accessibility defects, translation discrepancies, audience-blindness deviations,
  and cross-arm fidelity deviations;
- exact immutable git/course tag, dependency lock, model and processor revisions,
  prompt/template hash, asset/precompute checksums, and protocol/schema/instrument
  versions.

Release thresholds are frozen after feasibility/pilot evidence, not invented in
this plan.

## 7. Manual research and pedagogy gates

The following require named human sign-off and artifacts:

| Gate | Evidence | Accountable reviewer |
|---|---|---|
| Ethics/data governance | Approval or formal teaching-only determination; consent/assent, withdrawal, storage, Molab/platform retention, cross-border processing, access, incident process | Institutional PI/ethics authority |
| Construct/instrument quality | Source/provenance, adaptation permissions, content validity rationale, cognitive interviews/pilot, scoring and missingness plan | Research-methods lead |
| Comparison fidelity | Matched activity/time/help/source/output matrix, contamination plan, allocation and estimand | Study designer/statistician |
| Artistic authenticity | Production workflow affords meaningful composition/revision and rubric does not reduce creativity to model agreement | Arts educator + student advisory reviewers |
| Accessibility | WCAG-aligned interaction/media review, accommodations, EN/KO parity | Accessibility/localization reviewers |
| Audience validity | Independent human recruitment/response, blindness, minimum valid readings, contextual coding | Qualitative-methods lead |
| Claim audit | Every report sentence maps to metric/manipulation and uses approved epistemic label | Interpretability expert + critic |

No automated pass can waive a failed manual gate.

## 8. Static and command-level quality gates

Run from the repository root with the project environment:

```bash
python -m py_compile avllm_interpretability/CTP49906_avllm_molab.py \
  avllm_interpretability/src/*.py \
  curriculum_common/*.py \
  avllm_interpretability/scripts/generate_precompute.py

avllm_interpretability/.venv/bin/marimo check --strict \
  avllm_interpretability/CTP49906_avllm_molab.py

avllm_interpretability/.venv/bin/pytest -q avllm_interpretability/tests

avllm_interpretability/.venv/bin/ruff check \
  avllm_interpretability/CTP49906_avllm_molab.py \
  avllm_interpretability/src \
  curriculum_common \
  avllm_interpretability/scripts \
  avllm_interpretability/tests

git diff --check
```

Add repository scripts/tests that fail if:

- rendered student markdown contains `PRD`, feature/release codes, acceptance or
  reviewer language, implementation-history annotations, or broken heading levels;
- the comparison arm imports treatment-only modules or pre-measure terminology;
- required schema fields/condition codes/caveat labels are missing;
- Research mode can start without valid approved configuration;
- Teaching mode can transmit student artifacts/process/audience data beyond the
  approved Molab boundary or the research projection exports a forbidden field;
- full-vocabulary logits occur in serialized data;
- checked-in replay/model/assets use mutable references or mismatched checksums;
- release configuration still points at `main` or another mutable notebook/model/
  processor/course reference;
- EN/KO content keys diverge.

If an official notebook execution/export command is available in the pinned Marimo
version, add a clean-process cached export smoke test. Do not rely only on importing
the notebook because reactive execution order and browser presentation matter.

## 9. Acceptance traceability

| Acceptance criteria | Primary verification |
|---|---|
| AC-01–AC-04 | I-N01–I-N08, E2E-1, E2E-2, rendered hierarchy scan |
| AC-05 | U-M01–U-M14, I-M01/I-C01, E2E-1 |
| AC-06–AC-07 | U-R01–U-R05/U-R16, I-N01–I-N04/I-N09–I-N10, E2E-1/E2E-3 |
| AC-08–AC-10 | U-M03–U-M14, I-M01–I-M03, E2E-8, condition/manipulation audit |
| AC-11–AC-13 | U-P01–U-P12, I-P01–I-P05, claim audit |
| AC-14–AC-16 | I-C00–I-C07, E2E-5, manual instrument/fidelity gates |
| AC-17 | U-R09–U-R10/U-R18, I-N05, E2E-4, audience-validity gate |
| AC-18–AC-19 and AC-23 | U-R06–U-R08a/U-R14–U-R18, E2E-3/E2E-4/E2E-6/E2E-9, data-egress, projection, and permission-scope assertions |
| AC-20 | I-C06, E2E-7, accessibility/localization sign-off |
| AC-21 | command gates, E2E-1/E2E-2/E2E-8, host/GPU rehearsal report |
| AC-22 | U-M11/U-R12–U-R14, I-M05, E2E-3 |

## 10. Definition of done

The implementation is **classroom-ready** when all **classroom-column** automated
gates, live/cached rehearsals, artistic-authenticity, accessibility, localization,
and claim audits pass on an immutable release; Teaching mode, authorized audience
exchange, and private portfolio export are complete. Research-only failures remain
visibly disabled and do not masquerade as classroom failures.

It is **research-ready** only after classroom readiness plus every research-column
automated gate and the institutional ethics, instrument quality, comparison
fidelity, audience validity, allocation/analysis, retention, and rater-training
human gates are approved and version-locked. A failed or unknown research gate must
be reported as such; it must never be hidden behind a successful notebook run.
