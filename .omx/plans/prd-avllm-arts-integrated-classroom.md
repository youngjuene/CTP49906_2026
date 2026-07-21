# PRD: Counterpoint Lens arts-integrated classroom redesign

**Status:** Approved Ralplan consensus  
**Primary surface:** `avllm_interpretability/CTP49906_avllm_molab.py`  
**Planning mode:** deliberate  
**Context:** `.omx/context/avllm-arts-integrated-classroom-redesign-20260721T063942Z.md`

## 1. Executive decision

Evolve the AVLLM notebook into the **Counterpoint Lens treatment studio**: a
progressively disclosed path in which students state an artistic intention,
register a first audiovisual cut, inspect a common guided demonstration, run
controlled sound/image experiments, compare creator–audience–model readings,
revise both their explanation and artifact, and export an analyzable portfolio.

Use a **modular hybrid production design**:

- Both arms use a treatment-neutral **common production kernel**: one default
  no-cost editor route, a common export preset, source/trim/order/mix/accessibility/
  assistance decision manifest, and identical V1/V2 registration. This controls the
  core production experience without building a fragile timeline editor in Marimo.
- Students may use broader approved external tools only after the common outcome
  artifact, or when editor choice/allocation and assistance are matched and recorded
  across arms. The notebook records intent, versions, provenance, accessibility
  choices, and experiment results.
- The notebook and focused Python modules create deterministic research variants
  (audio swaps, temporal offsets, signal controls, and distinct modality-omission
  processor paths) from registered media.
- A separate production-matched digital-storytelling surface uses the same brief,
  time budget, source pool, V1/V2 workflow, audience exchange, and outcome
  instruments but does not expose interpretability tools. It is required for RQ1
  and must not be embedded in the treatment notebook.
- A companion study protocol owns consent, allocation, instruments, scoring,
  retention, and analysis. The notebook defaults to teaching-only session storage
  and permits no automatic or unapproved student-data egress. User-initiated
  private download, permission-authorized audience exchange, and approved minimized
  Research-mode export are the only explicit egress classes.

This boundary is essential: one treatment notebook can be excellent classroom
material, but cannot alone supply a defensible comparison condition, validated
measurement, or research governance.

## 2. RALPLAN-DR decision summary

### Principles

1. **Construct validity before feature breadth.** Every research claim must map to
   an observable artifact, process record, manipulation, or shared measure.
2. **Comparability and provenance by construction.** Paired variants preserve
   everything possible except the named manipulation and carry stable identities.
3. **Epistemic honesty.** Probe-score dispersion, teacher-forced answer-distribution
   dispersion, attention visualization, signal controls, modality omission, and
   edge knockout remain distinct.
4. **Privacy, accessibility, and cultural situatedness by default.** Research
   collection is opt-in and governed; human interpretation is not replaced by an
   automated judge; media has accessible alternatives.
5. **Progressive disclosure under classroom constraints.** A short required route
   works in one sitting and on cached replay; optional architecture work remains
   available without crowding the learning path.

### Top decision drivers

1. Operationalize RQ1–RQ4 with analyzable, non-confounded evidence.
2. Keep the student journey coherent and reliable within Molab and a 24 GB GPU.
3. Make artifacts reproducible, ethically collectable, and reusable across both
   study arms without leaking the interpretability treatment.

### Options considered

| Option | Strength | Cost / failure mode | Decision |
|---|---|---|---|
| A. Constrained notebook-native composition recipe | Standardized decisions, automatic provenance, little tool switching | Restricts artistic vocabulary; media rendering/reactivity risk; may teach the tool rather than composition | Adopt only the small deterministic recipe controls used for experimental variants |
| **B. Modular hybrid studio + common production kernel + matched companion surfaces** | Clean classroom route, shared default editor/export/manifest, testable modules, valid study boundaries | More release artifacts and instructor coordination; external-tool policy must be enforced | **Adopt** |
| C. One notebook-native nonlinear editor, experiment lab, survey, and research database | Single URL and apparent integration | Bloated reactive graph, unreliable encoding, comparison contamination, privacy complexity | Reject |
| D. Add only logs, swaps, and metrics to the existing playground | Smallest code diff | No production cycle; RQ1 and RQ4 remain unanswerable | Reject unless the aim/RQs are narrowed |
| E. Move all production and research activity to an external platform | Low notebook engineering load | Fragments learning and weakens the making–model inquiry link | Reserve as operational fallback |

## 3. Goals and non-goals

### Goals

- Make audiovisual creation and revision—not merely clip upload—the spine of the
  treatment learning experience.
- Preserve the current guided-demo-to-playground strength while connecting each
  model measurement to a student prediction, result, challenge, and revision.
- Implement controlled, coded RQ3 conditions with stable stimulus identity.
- Retain compact distribution summaries needed for layer-wise trajectories while
  accurately qualifying what they measure.
- Capture linked explanation revisions and artifact versions as exportable process
  data in research mode.
- Collect comparable creator, blind audience, and model readings of the same work.
- Supply a production-matched comparison arm and common measurement protocol for
  RQ1 without contaminating the treatment experience.
- Present a streamlined bilingual-ready, keyboard-operable notebook hierarchy with
  no visible PRD or release-management annotations.

### Non-goals

- Build a full non-linear audio/video editor inside Marimo.
- Automatically upload student work, names, or research records.
- Claim that attention is explanation, knockout is modality ablation, raw
  logit-lens entropy is calibrated uncertainty, or calibration is measured without
  a labeled evaluation corpus.
- Invent validated scales, scoring thresholds, required sample sizes, lag levels,
  or a retention period. Those require evidence review, pilot data, and ethics or
  protocol approval.
- Use `evaluation/caption_fidelity_eval.py` or another LLM judge as a substitute
  for independent human audience interpretation.
- Put research allocation, consent adjudication, or inferential analysis inside
  the treatment notebook.

## 4. Learning and evidence architecture

### 4.1 Required student route

Rendered notebook headings should retain only four top-level sections, with the
eight learning stages as numbered subsections. Implementation feature IDs, PRD
terminology, risks, and acceptance language remain in developer documentation and
tests—not notebook cells.

1. **Prepare your project**
   - **1.1 Orient — Make, test, revise:** learning outcomes, time map,
     evidence-versus-inference legend, teaching/research availability, Molab data
     boundary, and live-versus-saved-replay status.
   - **1.2 Compose — Plan your first cut:** audience and artistic intention,
     storyboard/sound-image relationship, cultural references, source/license
     provenance, accessibility plan, common editor/export preset, and planning card.
   - **1.3 Register — First cut (V1):** upload/preflight, session preview, content
     identity and media facts, edit-decision manifest, concept tags, and an explicit
     explanation of browser-to-Molab transfer and permitted export destinations.
2. **Guided demonstration**
   - **2.1 Observe — Shared reference:** retain saved replay/live reference clip,
     layer probe, attention view, edge knockout, and teacher-forced delta, with one
     interpretation checkpoint after each measure instead of long front-loaded prose.
3. **Exploratory playground**
   - **3.1 Experiment — Change one thing at a time:** choose a coded paired
     manipulation, predict before running, inspect trajectories/answer effects, and
     record result, limitation, rival explanation, and next control.
   - **3.2 Compare — Three readings of one work:** import at least two blinded
     independent audience readings, then reveal creator/model readings; examine
     culture, accessibility, prompt design, training unknowns, and label choice.
   - **3.3 Revise — Explanation and second cut (V2):** name the evidence trigger,
     revise the explanation, edit externally, register linked V2, and rerun the
     preselected comparison.
4. **Synthesis and architecture challenge**
   - **4.1 Synthesize — Portfolio and architecture challenge:** export human- and
     machine-readable records, then make one bounded proposal grounded in findings,
     distribution/measurement limitations, alternatives, and a next test.

### 4.2 Progressive disclosure

- **Required path:** orientation, intention card, V1, guided results, one paired
  manipulation, two audience responses, explanation revision, V2, export.
- **Choice path:** students choose one of congruence, swap, timing, or ablation
  questions and one model measure to pursue more deeply.
- **Advanced path:** custom layer bands, attention-head rules, feature/token
  intervention spike, and deeper architecture redesign; visually collapsed and
  never a prerequisite. The required Stage 8 proposal is a short evidence-synthesis
  exercise, not this optional engineering/design extension.
- Each section ends with one visible `Checkpoint` card containing completion state
  and the next action. Long method caveats live beside the relevant plot in a
  concise `What this can / cannot show` callout.

## 5. RQ-to-evidence matrix

| RQ | Required treatment evidence | Cross-arm / external evidence | Claim boundary |
|---|---|---|---|
| RQ1 | V1/V2 production manifests, common pre/post reasoning task, common artifact submissions | Production-matched comparison surface; common blinded rubrics; allocation, fidelity, rater reliability, and analysis protocol | Notebook completion alone cannot establish comparative influence |
| RQ2 | Initial and revised explanations linked to run IDs; artistic intention; model views; audience readings; intervention results; evidence trigger | Time-ordered process export and qualitative coding plan | Revision sequence can be described; mechanism/causality requires the approved design |
| RQ3 | Independently coded congruence crossed with swap/offset; silence/neutral controls; feasibility-tested audio/video omission; compact layer/position/modality distributions; final answer-token effects | Course stimulus manifest, manipulation checks, repeated paired stimuli, predeclared estimand and analysis | Signal controls, modality omission, and edge knockout are distinct; raw probes show score trajectories, not calibrated belief |
| RQ4 | Creator labels/free text; blind audience labels/free text; fixed neutral and student-prompt model outputs; contextual reflections | Audience collection/import protocol and shared coding vocabulary | Disagreement is situated evidence, not an automatic error label |

## 6. Functional requirements

### FR-1 — Safe operating mode and governance gate

- Publish a deployment threat model with four boundaries: (1) student browser/
  device, (2) Molab-hosted session/container and documented platform retention,
  (3) Git/model/package endpoints used to acquire code and weights, and (4) any
  instructor research destination. Never describe a hosted Molab kernel as simply
  “local.”
- Default to **Teaching mode**: session-scoped state inside the Molab processing
  boundary, manual private-portfolio download, no research identifier/destination,
  and a visible delete/reset action. Repository/dependency/model reads may use the
  network. Student data has no automatic/unapproved egress; allowlisted flows are
  user-initiated private download to the device, permission-authorized presentation
  export to the audience surface, and—only in Research mode—approved minimized
  export to the configured instructor destination.
- Expose **Research mode** only when an instructor-supplied approved configuration
  declares protocol version, consent/assent wording, permitted fields, retention
  destination/period, withdrawal path, and contact. A missing or invalid config
  must fall back to Teaching mode.
- Treat consent as an institutionally approved process outside the notebook. An
  in-notebook checkbox may acknowledge the active mode but must never be represented
  as informed consent. Keep grading independent of research participation and offer
  an equivalent non-research route.
- Represent separate permissions for process-data analysis, classroom/audience
  sharing, reproduction or quotation, and future reuse; do not infer one from
  another. Audience respondents are participants when their responses are analyzed.
- Use a student-generated pseudonym/session code; never request a legal name,
  email, demographic free text, or raw-media transmission to an unapproved service.
- Describe records as **pseudonymous**, never anonymous: hashes, faces, voices,
  filenames, free text, and exact timing can still identify people. Keep any
  identity-code linkage outside the notebook and research bundle.
- Keep two projections: a rich user-controlled private learning portfolio may carry
  content hashes and exact local history; a minimized research export maps a
  Teaching-mode `exchange_artifact_id` to an investigator-issued
  `study_artifact_id` and omits raw hashes, filenames, raw media, exact times,
  and device/platform metadata unless individually justified and allowlisted.
- Stamp both projections with schema and protocol/release versions.
- Keep exact wall-clock time in local teaching history only when useful. Research
  exports default to event order and session-relative elapsed time; exact/coarsened
  timestamps require a protocol justification and allowlist.

### FR-2 — Audiovisual production and version cycle

- Provide a structured intention/production card with creator intent, intended
  audience, intended sound-image relation, aesthetic/cultural references,
  intended concept tags, source/license provenance, and accessibility plan.
- Register V1 and V2 with immutable content hash, media facts, local filename alias,
  creation/import time, parent version, and creator-authored change rationale.
- Provide preview, replace, remove, and manifest-download controls. Never overwrite
  V1 when V2 is registered.
- Compare V1/V2 at the level of duration/streams, selected intent fields, model
  outcomes, audience coding, and creator explanation; do not pretend to infer the
  artistic reason from the files.
- Human-facing captions, transcripts, and descriptions are accessibility overlays.
  They must not silently enter the model prompt/input or change a formal stimulus;
  any such use is a separately coded condition.
- Use one default no-cost editor route and common export preset in both arms. Record
  editor/version, source assets, ordering/trims, mix/level decisions, accessibility
  work, and human/AI assistance in an edit-decision manifest. Broader external-tool
  use is allowed only after the shared outcome artifact or when matched/recorded
  under the study protocol; no proprietary product is required.

### FR-3 — Controlled RQ3 stimulus variants

- Add a checked-in `StimulusManifest` and deterministic transformation recipes.
- Every variant ID must derive from source hash + normalized recipe + transform
  schema version. Store `pair_id`, parent/donor IDs, pairing/block, technical
  operation, independently coded `congruence_level`, congruence coding/pretest
  provenance, manipulation-check result, duration/streams, parameters, and license.
- Congruence is not inferred from the operation: an audio swap may be congruent or
  incongruent. Freeze coding instructions, independent coders/pretest, and
  disagreement handling in the study protocol.
- Minimum technical-operation taxonomy:
  - `original_reference` with separately coded congruence;
  - `audio_swap_duration_matched` with identified donor and pairing block;
  - `temporal_offset` with signed milliseconds and zero-fill rather than wrapping;
  - `audio_silence_control` using duration-matched silence;
  - `video_neutral_control` using a documented neutral/black visual signal;
  - `audio_omitted_model_input` with no audio modality supplied to the processor;
  - `video_omitted_model_input` with no video modality supplied to the processor;
  - `direct_attention_edge_knockout`, recorded as a model intervention rather than
    a media transform.
- Name intervention levels precisely:
  - **stimulus-signal control** presents silence or neutral visual input;
  - **modality omission** supplies no audio or no video modality to a distinct
    processor path;
  - **processor/token or internal-representation ablation** is a further distinct
    engineering/research intervention;
  - **direct attention-edge knockout** masks selected information flow.
- Preserve source duration and the non-target stream; normalize audio to the
  processor contract (currently 16 kHz), trim/pad deterministically, and record
  every normalization.
- Before committing to live rendering, run a PyAV encoding feasibility spike on
  Molab. If exact playback/export cannot be made reliable without a new dependency,
  ship pre-rendered licensed course variants and keep upload transforms limited to
  safe in-memory/model-input operations. Never present a separately injected chat
  audio stream as if it were the same token layout.
- Feasibility-test actual audio-omitted and video-omitted processor paths; the
  current clip validator/audio-in-video route is not evidence that omission works.
  Because omission changes token layout, restrict its comparison to compatible
  aggregate or final-output estimands. If reliable omission is impossible, remove or
  rename “modality ablation” in the research question/materials rather than using
  silence/neutral signal as a substitute.
- Formal research must use a coded multi-stimulus set and paired/repeated design;
  the number of stimuli and offset levels are protocol/pilot decisions.

### FR-4 — Distribution summaries and aligned trajectories

- Replace argmax-only retention with a streaming compact summary per selected
  layer, position or normalized position bin, modality, stimulus, prompt, and
  condition:
  - top-k token IDs/text/log probabilities;
  - full-vocabulary Shannon entropy in nats and top-1 minus top-2 log-probability
    margin; if normalized entropy is shown, define it as `H / log(vocab_size)` and
    record vocabulary size;
  - rank/log probability for a predeclared target token set when supplied;
  - optional top-k overlap to the final layer. Defer Jensen–Shannon divergence until
    truncated-support semantics are justified as pedagogically necessary.
- Include audio and video positions where processor metadata permits; label unknown
  or ambiguous positions instead of guessing modality.
- Capture only selected layers and selected/binned positions, slice activations
  immediately inside the hook, project through the language head in bounded chunks,
  compute summaries on-device, release tensors after each chunk, and move only
  compact results to CPU. Do not retain every full layer output or persist
  full-vocabulary logits.
- Version every metric definition, including entropy units/normalization, margin,
  tie/rank policy, dtype/accumulation precision, vocabulary/support, probe transform,
  and target-token-set version.
- Align position-wise comparisons only when processor/token-layout fingerprints
  match. Otherwise compare preregistered aggregate summaries or normalized bins and
  show an alignment warning.
- Use the labels **probe score dispersion** or **raw-logit trajectory** for
  intermediate logit-lens entropy because the probe omits the model's final
  normalization and is not calibrated. Label entropy/margin from actual
  teacher-forced answer-token distributions **teacher-forced answer-distribution
  dispersion (uncalibrated proxy)**. It is not free-generation uncertainty or
  calibration. Do not expose calibration/ECE until a labeled evaluation set and
  protocol exist.

### FR-5 — Append-only process records

- Replace the static worksheet as the source of truth with session-scoped append-only
  events and user-controlled JSONL export; keep a human-readable worksheet view.
- Generate a UUID run ID and local UTC timestamp when a model run is committed. A
  run record also receives a monotonic event index and session-relative elapsed time
  and contains session pseudonym, mode/protocol/schema versions, V1/V2 artifact
  ID, stimulus/condition ID, model/revision identity, prompt, exact parameters,
  immutable command nonce, pre-run prediction and initial explanation,
  token-layout fingerprint, result digest, compact metrics, timing, and status.
- Implement reactive side effects as commands: forms create an immutable command
  nonce; a pure transition reducer validates the command; append-once storage is
  keyed by command ID; result/reflection attachment is idempotent; rendered cells
  read only committed state. Upstream invalidation or re-render may not append.
- Require a prediction/initial explanation before execution. After execution,
  collect result interpretation, verdict, evidence trigger, revised explanation,
  rival explanation, next control, and optional artifact-revision intent in a
  reflection record linked by `run_id`.
- Revisions are new records with `parent_run_id` or `revision_of`; no UI action
  silently mutates prior submitted text.
- Support explicit missing/withdrawn states so incomplete data is distinguishable
  from empty text. Export validation must report orphan reflections, missing
  predictions, duplicate IDs, and unapproved fields.

### FR-6 — Creator–audience–model comparison

- Capture the creator statement and codes before model/audience revelation.
- Import at least two independent pseudonymous audience readings collected in the
  separate audience surface for the same artifact, blind to creator intention and
  model output at time of response.
- Audience response fields include open interpretation, perceived sound-image
  relation, shared concept tags plus self-described tags, accessibility barriers,
  and confidence/ambiguity. Allow `prefer not to answer` where appropriate.
- Run one recorded neutral prompt for comparability and optionally a student prompt
  for prompt-sensitivity inquiry. Display model revision and decoding parameters.
- Present a triadic matrix of shared/different tags and free-text readings. Use
  neutral language such as agreement, divergence, or unrepresented reading—not
  truth/error—unless an objectively scored item exists.
- Prompt reflection on cultural convention, accessibility, training provenance
  (known, documented, or unknown), prompt wording, and label-vocabulary effects.
- The existing automated caption evaluation may remain a developer diagnostic but
  cannot populate human-audience fields.
- Collect responses in a separate no-GPU audience surface and import them into the
  treatment notebook; do not make the treatment surface silently double as the
  blinded collector. A Teaching-mode handoff packet uses a random
  `exchange_artifact_id`, includes
  only the permitted presentation package, excludes creator/model fields, verifies
  distinct respondent/session pseudonyms, carries blindness/reveal/deviation state,
  validates on import, and follows an instructor media-sharing/permission runbook.
- Two distinct audience readings are the **minimum classroom comparison activity**,
  not a research sample-size justification. The approved sampling/analysis plan
  determines research sufficiency.

### FR-7 — Production-matched comparison arm and common outcomes

- Create a separate comparison notebook/guide after the common production engine
  and instruments stabilize. It must match:
  - instructional time, hardware access, asset budget, accessibility support, and
    facilitator contact;
  - production brief, source pool, editor options, export constraints, V1/V2 count,
    and audience-exchange exposure;
  - pre/post critical-AI reasoning tasks and artifact submission/rubric procedure.
- It must replace layer probes, attention views, and causal interventions with a
  production-relevant digital-storytelling activity of equivalent time and
  cognitive demand. It must not import treatment-only interpretability modules or
  reveal model-internals terminology before post-measures.
- Outcome instruments:
  - creative flexibility: evidence-reviewed/piloted rubric or task with blinded
    scoring across both arms;
  - audiovisual intentionality: common artifact-and-rationale rubric;
  - critical AI reasoning: common scenario-based pre/post assessment requiring
    claim, evidence, limitation, rival explanation, and next test.
- Store instrument version, administration order, missingness, and rater IDs.
  Formal thresholds, power/sample size, assignment/allocation, covariates,
  inter-rater reliability targets, and analysis belong to the approved study
  protocol. Use at least two trained blinded raters for formal artifact scoring and
  predefine disagreement adjudication before data collection.
- Randomize where feasible. If allocation is clustered, use multiple clusters per
  arm and cluster-aware power/analysis; one class or teacher per condition cannot be
  separated from the treatment effect.
- Record treatment fidelity (time-on-task, completed stages, exposure to prohibited
  material, technical disruptions) identically across arms.

### FR-8 — Accessibility, localization, and classroom resilience

- Provide English and Korean instructional resources from the same content keys or
  a documented parity process; neither language version may expose different study
  guidance.
- Require captions/transcript and an audio-description or equivalent visual-context
  field for submitted work, with an instructor exception path for the artistic
  cases where absence itself is deliberate and documented.
- Target WCAG 2.2 AA across the complete Molab/Marimo workflow without claiming
  conformance until tested. Human-correct automatic captions before submission.
- All controls must be keyboard operable and labeled; plots need text summaries and
  downloadable tables; meaning must not rely on color alone; time-limited tasks need
  an accommodation/adjustment path.
- Never autoplay audio. Give warnings before expensive GPU cells. Preserve saved
  course replay for the complete guided section and provide example process/audience
  records so no-GPU students can complete interpretive exercises.
- A failed optional transform or model run must leave the production manifest and
  prior records intact and provide a recoverable retry path.

### FR-9 — Exportable learning portfolio

- One explicit export action creates a user-controlled private bundle containing:
  - human-readable bilingual-ready report;
  - artifact/version and stimulus manifests;
  - run and reflection JSONL/CSV;
  - audience imports/responses permitted by mode;
  - plot/table data, protocol/schema/model versions, and checksums;
  - immutable notebook/course tag, processor/tokenizer and package versions,
    prompt/template hash, decoding/seed settings, live-versus-cached status, metric
    versions, source ingredients/licenses/releases, edit/manipulation actions, AI
    assistance disclosure, human revisions, and known/unknown provenance;
  - a disclosure of omitted raw media and redacted fields.
- The export must be deterministic apart from declared timestamps/UUIDs, validate
  against its schema, and re-import without loss of supported fields.
- Hashes establish association and change history, not truth or authorship. C2PA
  vocabulary may inform the schema but does not imply compliance or a dependency.
- Teaching-mode export remains a private learning portfolio; it must not be labeled
  research data.

### FR-10 — Runtime and reliability budgets

- Preserve one shared 3B model instance, current upload safety checks, clear CUDA
  failure messages, and the 24 GB GPU envelope.
- Pin the notebook repository reference, course assets, model, processor, prompt
  template, and precompute generator to immutable revisions for release; `main` is
  not a publishable course reference.
- Publish one immutable `course_release_profile.json` consumed by both arms and
  tests. It records course/model/processor/prompt/asset versions, common editor and
  export-preset IDs, permitted condition/metric versions, and pilot-derived time,
  host-RAM, VRAM, and usability thresholds.
- Default the guided route to bounded layers, frames, tokens, top-k, and cached
  artifacts. Advanced controls may widen these only with an estimated-cost warning.
- Add per-run elapsed time, peak allocated/reserved VRAM, and peak host RSS to
  session diagnostics.
  Do not include hardware identifiers in research export unless approved.
- In live-class rehearsal, the required path must avoid kernel restarts and recover
  cleanly after a failed run. Exact time budgets are set after the feasibility
  pilot, then documented as release gates.

## 7. Data contracts

Schemas should be plain dataclasses/typed mappings with explicit validation and
versioning; use standard-library JSON/CSV and existing dependencies unless a
dependency decision is separately approved.

### 7.1 Artifact version

Required fields:

`schema_version`, `artifact_id`, `content_sha256`, `version_label`,
`parent_artifact_id`, `local_registered_at_utc`, `event_index`, `elapsed_ms`,
`media_facts`, `creator_intention`,
`intended_audience`, `sound_image_relation`, `concept_tags`,
`cultural_aesthetic_context`, `source_license_provenance`, `accessibility`,
`editor_name_version`, `source_assets`, `edit_decisions`, `assistance_disclosure`,
`export_preset_version`, `change_rationale`, `processing_boundary`.

### 7.2 Stimulus variant

Required fields:

`schema_version`, `stimulus_id`, `pair_id`, `parent_content_sha256`,
`condition_code`, `technical_operation`, `congruence_level`,
`congruence_coding_provenance`, `pairing_block`, `manipulation_check`, `recipe`,
`donor_stimulus_id`, `offset_ms`, `duration_ms`, `audio_facts`, `video_facts`,
`processor_path`, `token_layout_fingerprint`, `provenance`, `checksum`.

### 7.3 Model run and reflection

Run fields:

`schema_version`, `run_id`, `session_pseudonym`, `local_created_at_utc`,
`event_index`, `elapsed_ms`, `mode`,
`protocol_version`, `artifact_id`, `stimulus_id`, `condition_code`, `model_id`,
`model_revision`, `prompt`, `parameters`, `command_nonce`, `prediction`,
`initial_explanation`, `result_digest`, `metrics`, `metric_versions`,
`token_layout_fingerprint`, `elapsed_seconds`, `peak_vram_bytes`,
`peak_host_rss_bytes`, `status`.

Reflection fields:

`schema_version`, `reflection_id`, `run_id`, `local_created_at_utc`, `event_index`,
`elapsed_ms`,
`initial_explanation_snapshot`, `result_interpretation`, `verdict`,
`evidence_trigger`, `revised_explanation`, `rival_explanation`, `next_control`,
`artifact_revision_intent`, `withdrawn`.

### 7.4 Audience reading

Required fields:

`schema_version`, `response_id`, `exchange_artifact_id`, `audience_pseudonym`,
`local_created_at_utc`, `event_index`, `elapsed_ms`, `blindness_attestation`,
`permission_scope`, `open_interpretation`,
`sound_image_relation`, `shared_tags`, `self_described_tags`,
`accessibility_barriers`, `confidence_or_ambiguity`, `withdrawn`.

The separate handoff envelope contains `packet_schema_version`,
`exchange_artifact_id`, `presentation_asset`, `presentation_checksum`,
`permitted_accessibility_overlays`, `excluded_field_attestation`,
`reveal_state`, `protocol_deviation`, and `sharing_permission_scope`. It contains no
creator intention or model output. Import validates packet/response linkage and
distinct respondent/session pseudonyms.

### 7.5 Private portfolio and research projection

- The private portfolio retains the fields needed by the student to reproduce and
  reflect on the work, including content hashes and local event times.
- The research projection maps `exchange_artifact_id` and private hashes to an
  investigator-issued `study_artifact_id` and exports event order/elapsed time by
  default. It strips raw
  content hashes, filenames, media, exact UTC time, device/platform metadata, and
  free fields not on the approved schema allowlist.
- Projection is a pure, versioned transformation with a preview/redaction report;
  it never mutates the private portfolio.

### 7.6 Backward compatibility

- Existing `precomputed/` captions, attention summary, logit CSV, and meta artifacts
  remain readable during a one-release migration window.
- New readers distinguish legacy argmax-only data and display `not measured` for
  absent distribution metrics; they never synthesize entropy from decoded labels.
- The old `WORKSHEET.md` becomes an example/print fallback or redirects to the new
  export schema; existing content is not silently treated as research data.

## 8. Implementation work packages and file ownership

### WP-0 — Protocol, ethics, and feasibility gates

1. Obtain the KAIST IRB determination and draft the companion study protocol,
   consent/data-management template, separate participation/sharing/reuse scopes,
   RQ estimands, comparison-arm fidelity checklist, evidence review for instruments,
   and human-audience procedure.
2. Before treatment UI composition, freeze a mirrored activity/time map, neutral
   production shell and default editor/export policy, common instrument candidates,
   facilitator help/contact rules, audience handoff, condition-specific cognitive-
   demand rationale, contamination boundary, and fidelity/deviation matrix. These
   may remain pilot-versioned, but matching cannot be reconstructed afterward.
3. Pilot PyAV rendering/preview/export on Molab and document supported codec path.
4. Pilot processor metadata and distinct audio-/video-omitted input paths for
   modality position identification and token-layout
   stability across each planned manipulation.
5. Freeze condition codes, congruence coding/manipulation checks, initial stimulus
   set, lag levels, and scoring procedure
   only after these spikes. No research mode ships before institutional approval.
6. Freeze release paths in the profile before implementation tickets: proposed
   defaults are `curriculum_common/`,
   `digital_storytelling/CTP49906_storytelling_molab.py`,
   `audience/CTP49906_audience_response_molab.py`, and `study_materials/`. Any rename
   updates both arm manifests/tests atomically.
7. Run a stimulus-design estimability audit showing declared RQ3 contrasts are not
   perfectly confounded with donor/source, operation, congruence code, block, or
   presentation order; revise the manifest/design before Research mode if they are.

### WP-1 — Pure media and identity layer

- Add `avllm_interpretability/src/stimulus_variants.py` for normalized recipes,
  deterministic IDs, duration matching, offsets, signal controls, congruence/pairing
  metadata, and manifests. Keep modality-omission processor logic separate from
  media transformations.
- Extend `src/playground_clips.py` for V1/V2 artifact registration and stable
  media-fact extraction while preserving current upload preflight and SHA caching.
- Add a licensed `assets/stimulus_manifest.json` and course variants or generation
  recipes; keep donor provenance explicit.
- Add unit fixtures with tiny synthetic media; do not commit student work.

### WP-2 — Metrics and experiment contracts

- Add `src/probe_metrics.py` for streaming top-k, entropy, margin, target rank,
  modality/position summaries, and alignment checks.
- Refactor `src/logitlens_experiment.py` to return a typed compact result consumed by
  both live and precomputed paths, capturing only selected/sliced activations and
  projecting in bounded chunks; preserve a legacy CSV adapter temporarily.
- Extend `src/teacher_forcing.py` with final answer-token entropy/margin summaries
  without changing its baseline/knockout Δ log-likelihood semantics.
- Extend `src/precompute.py` and `scripts/generate_precompute.py` with schema-versioned
  distributions, condition/manipulation metadata, and migration validation.
- Keep `src/attention_knockout_experiment.py` named and documented as edge knockout;
  add any internal ablation as a distinct module/API only after the feasibility
  gate.

### WP-3 — Process, audience, and export layer

- Add a treatment-neutral top-level `curriculum_common/` package containing focused
  `production_manifest.py`, `session_records.py`, `audience_packets.py`, and
  `portfolio_export.py` modules. It owns schemas, the pure exactly-once reducer,
  append-once linkage, projection/redaction, withdrawal, safe import, and checksums;
  it must not import `avllm_interpretability` internals.
- Add a separate small audience-response surface that consumes only the neutral
  presentation packet, requires no GPU, and runs before model/creator revelation.
- Prefer these four bounded modules over a generic event/research framework and add
  no dependency without a separate decision.

### WP-4 — Primary notebook redesign

- Entry criterion: the WP-0 mirrored route, neutral production contract, audience
  handoff, and contamination boundary have been reviewed alongside the treatment
  route.
- Recompose `CTP49906_avllm_molab.py` using the four-section/eight-stage route in §4.
- Keep computation in modules and cells limited to orchestration, forms, plots,
  concise interpretation scaffolds, and state transitions.
- Implement explicit `Prepare → Run → Reflect` forms keyed by committed run IDs;
  button-trigger all expensive work and snapshot form values at submission.
- Put replay/live badges, measurement caveats, condition IDs, current artifact
  version, and completion checkpoints beside the outputs they qualify.
- Label each activity `Required today`, `Optional investigation`, or `Research-only`;
  hide/disable research-only actions until their gates are satisfied.
- Hide setup boilerplate while keeping meaningful parameter and analysis cells
  inspectable. Scan rendered markdown for PRD codes, internal roadmaps, review
  comments, release labels, and implementation-history narration.

### WP-5 — Matched comparison and instruments

- Add a separate comparison notebook or low-complexity classroom guide powered by
  the common production/process modules but with no treatment imports or language.
- Add versioned common pre/post assessment, artifact rubrics, rater guide, treatment
  fidelity form, and protocol templates under a clearly separated study-materials
  directory.
- Implement export compatibility across arms; instrument/scoring logic remains
  independent of condition label where possible.

### WP-6 — Accessibility, localization, replay, and documentation

- Update English/Korean READMEs, instructor runbook, student quick-start, worksheet
  migration note, privacy/data dictionary, and troubleshooting guide.
- Regenerate course replay from an immutable model revision and coded stimuli;
  publish checksums and generation command.
- Add text alternatives and downloadable data for all required visuals and verify
  keyboard order at the rendered notebook level.

### WP-7 — Pilot and release

1. Technical pilot with non-research synthetic artifacts.
2. Cognitive walkthrough with students/instructors not in the study sample.
3. Small approved pilot for instrument timing, manipulation fidelity, audience
   blindness, rater training, missingness, and export usability.
4. Freeze immutable course tag, model revision, schemas, protocol, assets, and
   translations; rehearse both arms under matching classroom conditions.

## 9. Acceptance criteria

### Classroom journey

- **AC-01:** A first-time student can complete the required route in order from the
  rendered headings/checkpoints without opening source code or the PRD.
- **AC-02:** The rendered notebook contains no `PRD`, feature/release codes,
  acceptance-test language, reviewer annotations, or development-history prose.
- **AC-03:** A student can register V1, run one paired investigation, safely import
  two responses created in the separate blinded audience surface, revise an
  explanation, register V2, and export a
  valid portfolio without losing prior records.
- **AC-04:** Guided replay supplies all outputs required for observation/reflection
  without GPU/model download; unsupported live actions are visibly disabled rather
  than failing late.

### Production and process evidence

- **AC-05:** V1 and V2 have distinct content-derived IDs, immutable lineage, intent,
  source/editor/edit/assistance provenance, accessibility metadata, common export-
  preset identity, and creator-authored change rationale.
- **AC-06:** Every committed model run has a UUID, local UTC timestamp, exact inputs
  and parameters, condition/stimulus/model/schema identity, event order/elapsed
  time, immutable command nonce, result digest, and linked pre-run prediction plus
  initial explanation; research export minimizes wall-clock time per protocol.
- **AC-07:** Every run/reflection is appended exactly once per command ID despite
  unrelated Marimo invalidation/reload; every revision links to its run and export
  validation detects missing/orphan/duplicate/withdrawn records.

### Controlled manipulation and measurement

- **AC-08:** The course manifest independently codes congruence and technical
  operation and includes duration-matched audio swaps, signed offsets, silence and
  neutral-video controls, feasibility-tested audio-/video-omitted processor inputs,
  and edge knockout with pair/block IDs, checks, provenance, and checksums.
- **AC-09:** Automated checks prove the target stream change and non-target stream
  preservation within protocol tolerances; duration/sample/frame differences are
  reported, never hidden.
- **AC-10:** UI, data, and documentation distinguish signal controls, modality
  omission, processor/token/internal ablation, and direct-edge knockout; unsupported
  omission forces the RQ/material to be renamed rather than substituted.
- **AC-11:** Compact summaries include versioned top-k, entropy, margin, and optional
  target rank by supported layer/position/modality; only selected/sliced activations
  are chunk-projected, full logits are not retained, and release-profile peak host
  RAM and VRAM limits pass.
- **AC-12:** Position-wise condition comparisons are blocked or downgraded when
  token-layout fingerprints differ, with an actionable explanation.
- **AC-13:** Intermediate probe dispersion and teacher-forced answer-distribution
  dispersion use distinct labels/caveats; no free-generation uncertainty or
  calibration claim appears without the corresponding evaluation artifact.

### RQ1 and RQ4 validity

- **AC-14:** Before treatment UI freeze, treatment and comparison route specifications
  pass a fidelity audit for matched production time, hardware, asset budget,
  default editor/export shell, brief, V1/V2 outputs, accessibility support, audience
  exchange, instruments, cognitive demand, and facilitator contact.
- **AC-15:** A static dependency/text audit confirms the comparison arm neither
  imports treatment-only interpretability modules nor exposes layer/attention/
  knockout content before post-measure completion.
- **AC-16:** Common instruments have explicit provenance, version, administration,
  scoring, missingness, rater-training, reliability/adjudication, and pilot status;
  no unvalidated instrument is described as validated.
- **AC-17:** The triadic activity imports a schema-valid presentation/response packet
  for the same `exchange_artifact_id` from at least two distinct blinded audience
  sessions, excludes creator/model fields before reveal, records deviations, and
  retains open text/contextual reflection; only a research projection maps the
  exchange ID to an investigator-issued study ID.

### Governance, accessibility, and operations

- **AC-18:** A clean install starts in Teaching mode with no automatic/unapproved
  student-data egress; transport tests allow only user-initiated private download,
  permission-authorized audience presentation export, and approved Research-mode
  minimized export. Private/exchange/research projections are distinct, and
  Research mode rejects absent/invalid approval configuration.
- **AC-23:** An audit confirms that Research mode does not equate an in-notebook
  checkbox with consent, describes records as pseudonymous, enforces distinct
  permission scopes, and supports an equivalent non-research classroom route.
- **AC-19:** A student can inspect session data, download a private portfolio,
  withdraw/mark permitted records, and reset/delete session artifacts; each
  projection reports what it contains, transforms, and omits.
- **AC-20:** All required controls are keyboard usable and labeled; required plots
  have text/table alternatives; EN/KO instructional parity and media accessibility
  fields pass review.
- **AC-21:** Strict Marimo validation, compile/static checks, CPU unit/integration
  tests, cached end-to-end rehearsal, and the bounded 24 GB live-GPU rehearsal pass
  on the immutable course tag.
- **AC-22:** A forced transform/model/export failure preserves earlier versions and
  records and offers a successful retry after correction.

## 10. Deliberate pre-mortem

| Failure scenario | Earliest warning | Mitigation / release gate |
|---|---|---|
| The notebook becomes a tool tutorial and students spend class debugging instead of making and reasoning | Required route exceeds rehearsal budget; many cell-state resets; students cannot explain the question after setup | Hybrid external authoring, pure helper modules, one short required path, cached guided results, cost warnings, timed classroom rehearsal before release |
| Captured process/audience data is unusable or ethically inappropriate | Orphan reflections, names/raw media in exports, inconsistent consent status, audience sees creator/model answer early | Teaching mode default; approval config gate; field allowlist/redaction; append-only schema; blindness sequence; export audit; ethics/pilot sign-off before Research mode |
| Manipulations or “uncertainty” results are confounded and overclaimed | Durations/token layouts differ, decoder outputs collapse, probe entropy is narrated as confidence | Manifest invariants, paired stimuli, token-layout fingerprint gate, collapse diagnostics, separate probe/predictive labels, predeclared estimands, analyst review of pilot |
| The common production route either suppresses artistic choice or editor variation confounds the arms | Students converge on identical recipe artifacts, or editor/help patterns differ by condition | Require one matched default route/shared outcome, record edit decisions and assistance, allow broader tools after that artifact, and gate release on arts-educator plus fidelity review |

## 11. ADR-001: Separate the treatment studio from the comparison arm

### Context

RQ1 asks for comparison with production-matched digital storytelling. Showing
interpretability tools inside the comparison condition would contaminate that
condition; omitting a comparison condition makes RQ1 unanswerable. Building all
study governance and media editing into one notebook would also undermine the
streamlined classroom experience.

### Drivers

- operationalize all six P1 findings and RQ1–RQ4 without treatment contamination;
- preserve an understandable, reliable Molab classroom route within the GPU/memory
  envelope;
- make provenance, accessibility, privacy, and epistemic limits testable by design.

### Alternatives considered

- constrained notebook-native composition recipe;
- modular hybrid studio with common production kernel;
- one all-in-one notebook/editor/database;
- minimal instrumentation of the current playground;
- fully external production/research platform.

Section 2 records the bounded tradeoffs and invalidation reasons. The constrained
recipe is retained only for deterministic experimental variants; the monolith and
minimal patch cannot meet the combined pedagogical/research requirements.

### Decision

Use a shared, treatment-neutral production/record/export core with separate
student-facing surfaces:

- common production kernel: one matched no-cost editor/export route and edit-
  decision manifest, with broader tools governed after the shared outcome;
- Counterpoint Lens treatment notebook: production + interpretability inquiry;
- digital-storytelling comparison surface: production-matched activity without
  interpretability exposure;
- audience response surface: lightweight and blinded;
- study protocol: allocation, consent, measures, raters, retention, and analysis.

The AVLLM notebook remains the primary treatment orchestrator. It links to these
surfaces but does not own research allocation or remote collection.

### Why chosen

This is the only option that simultaneously preserves artistic authorship, gives
both arms a matched production core, prevents interpretability leakage, keeps
Marimo focused on inquiry, and supports independently testable data/metric modules.

### Consequences

- Positive: cleaner notebook, defensible RQ1 boundary, reusable common modules,
  simpler audience workflow, and independently testable data contracts.
- Negative: more artifacts to version, fidelity checks across surfaces, and an
  instructor runbook are required.
- Rejected: a single monolith, because its apparent convenience sacrifices
  reliability, privacy separation, and comparison validity.
- Revisit if: the research aim drops RQ1, or an institutionally approved external
  platform demonstrably supplies matched production, audience, process, and export
  contracts without treatment leakage.

### Follow-ups

Execute WP-0 before UI work, use the readiness matrix to separate classroom from
research release, then follow the Goal/Team handoff in §13. Revisit the ADR only with
new evidence about the aim, platform, or comparison-design boundary.

## 12. Dependencies, sequencing, and stop gates

1. **Co-design/governance first:** freeze the mirrored treatment/comparison route,
   neutral production and audience contracts, instrument candidates, threat model,
   ethics requirements, accessibility obligations, and retention constraints.
2. **Technical spikes next:** media render/token-layout feasibility and compact
   metric correctness. If either fails, choose pre-rendered course variants or
   aggregate-only comparisons before notebook UI work.
3. **Pure modules and tests:** schemas, identity, transforms, metrics, and export.
4. **Treatment notebook:** integrate the required route and cached replay only after
   the mirrored route/fidelity entry criterion passes.
5. **Comparison/audience surfaces:** implement after neutral shared contracts
   stabilize, without changing the already matched route unilaterally.
6. **Pilot/freeze:** revise timing and wording, then freeze protocol/course tag and
   generate release evidence.

### Readiness gate matrix

| Gate | Required to claim classroom-ready | Additional requirement to claim research-ready |
|---|---|---|
| Mode/data | Teaching mode, Molab threat-model disclosure, private portfolio, reset/delete, no unapproved student-data egress | Institutional determination/approval, approved permission scopes, minimized projection, retention/access/withdrawal destination |
| Learning production | Common default editor/export route, V1/V2, intention/edit manifest, accessible alternatives | Frozen matched shell/fidelity tolerances across both arms |
| Experiment | Guided replay/live path, controlled course examples, honest operation/metric labels | Coded multi-stimulus set, congruence/manipulation checks, supported modality-omission estimands, prespecified analysis |
| Audience | Permission-aware teaching exchange or synthetic example packet | Approved human audience procedure, blindness, distinct respondents, research export/coding |
| Outcomes | Classroom reflection/rubric may be explicitly pilot/formative | Evidence-reviewed instruments, allocation/power, raters, reliability, missingness, multiplicity |
| Release | Accessibility/localization review, runtime/memory rehearsal, immutable course/model/assets | Protocol/instrument/schema lock, comparison contamination/fidelity audit, research-methods sign-off |

Teaching-only development and release may proceed when the classroom column passes.
Missing research approval or instruments must keep Research mode/actions absent or
disabled, not block a safe classroom release. Research implementation/claims must
stop rather than improvise when any research-column requirement is missing.

## 13. Agent roster and execution handoff

### Available agent types

Use only installed roles from the repository catalog; inherit the configured model
unless an explicit override is justified.

| Agent type | Suggested reasoning | Use in this program |
|---|---:|---|
| `architect` | xhigh | WP-0 boundaries, processor/estimability decisions, architecture-invariant review |
| `researcher` | high | Current official evidence for instruments, accessibility, provenance, and platform constraints |
| `designer` | high | Arts-production authenticity, student flow, accessibility interaction design |
| `executor` | medium | Media, metrics, neutral-record modules, notebook integration, comparison/audience surfaces |
| `test-engineer` | medium | Fixtures, exactly-once/reactivity, schema/property/integration/E2E/performance tests |
| `writer` | high | EN/KO student/instructor materials, data dictionary, protocol-facing documentation |
| `dependency-expert` | high | PyAV/processor/Molab feasibility only when official upstream behavior is unclear |
| `verifier` | high | Acceptance traceability, replay/live evidence, data-egress and release audit |
| `code-reviewer` | high | Independent final code/Marimo/test review |
| `critic` | high | Research-claim and RQ-coverage challenge after implementation evidence exists |

The research-methods/ethics owner, institutional reviewer, arts educator,
accessibility/localization reviewers, and trained human raters remain accountable
human roles; agents cannot supply their approval.

### Staffing guidance

- **WP-0 (mostly sequential):** one `architect` (xhigh), one `researcher` (high), and
  one `designer` (high) prepare evidence for the human methods/ethics owner.
- **WP-1–WP-3 (four parallel build lanes):** four `executor` workers (medium) own
  media/processor, metrics/precompute, neutral records/audience/export, and fixtures/
  shared release profile respectively; a `test-engineer` reviews contracts as soon
  as each lane lands.
- **WP-4 integration:** exactly one `executor` owns
  `CTP49906_avllm_molab.py`; other workers do not touch it. Integrate only after pure
  module tests and the mirrored-route entry gate pass.
- **WP-5/WP-6:** a second executor can own the comparison/audience surfaces while
  `designer` + `writer` own pedagogy/accessibility/localization without editing model
  computation.
- **WP-7:** independent `verifier`, `code-reviewer`, and `architect` lanes audit the
  release; the leader reconciles all evidence before any durable completion marker.

### Goal-mode follow-up suggestions

- **Default:** `$ultragoal` owns the durable WP-0–WP-7 goal/ledger and stop gates.
- **Parallelizable implementation:** use **Team + Ultragoal** after WP-0. Team
  returns checkpoint-ready implementation/test evidence; only the leader updates
  the Ultragoal ledger from fresh goal state.
- **Research deliverable only:** `$autoresearch-goal` may evaluate the measurement/
  protocol evidence once a human owner defines the research artifact and evaluator;
  it is not a substitute for IRB approval or the implementation lane.
- **Optimization only:** `$performance-goal` is appropriate after correctness for a
  measurable host-RAM/VRAM/runtime target, not before metric validity.
- **Fallback:** use `$ralph` only if the user explicitly requests a persistent
  sequential single-owner verification/fix loop; it is not the default.

### Launch hints (not executed by this plan)

```bash
# Create the leader-owned durable goal plan from the approved artifact.
omx ultragoal create-goals \
  --brief-file .omx/plans/prd-avllm-arts-integrated-classroom.md

# After WP-0 gates are recorded, launch the independent pure-module lanes.
omx team 4:executor \
  "Execute approved WP-1–WP-3 from .omx/plans/prd-avllm-arts-integrated-classroom.md and its test spec; preserve one-owner notebook boundary and return checkpoint evidence."

# Observe rather than shutting down early.
omx team status <team-name>
omx team await <team-name> --timeout-ms 30000 --json
```

### Team verification path

Before Team shutdown, workers provide changed-file ownership, unit/integration
results, schema/manifest fixtures, memory/runtime measurements, and unresolved
risks. The leader then runs sequential integration evidence: strict Marimo check,
cached E2E, live 24 GB rehearsal, comparison contamination/fidelity audit,
accessibility/localization review, and independent code/architecture review. Only
after those results satisfy the applicable classroom/research gate does the leader
checkpoint Ultragoal with fresh goal state; then `omx team shutdown <team-name>` is
safe. Team workers never mutate the Ultragoal ledger themselves.

## 14. Open decisions that implementation must resolve explicitly

- Institution/participant-specific ethics, consent/assent, withdrawal, storage,
  retention, and research-mode configuration.
- Evidence-backed instrument candidates and adaptation/translation permissions.
- Formal estimands, allocation, sample size/power, rater reliability thresholds,
  and missing-data analysis.
- Approved source pool, licenses, number of course stimuli, and final offset levels.
- PyAV render path versus shipped pre-rendered variants.
- Processor-supported modality positions and whether any internal ablation is
  feasible without invalidating the input comparison.
- Synchronous response-surface deployment details, provided the separate blinded
  import contract and Molab/instructor-boundary guarantees remain testable.

These are gates, not reasons to weaken the classroom material or fill gaps with
unsupported defaults.

## 15. External evidence foundation

Implementation and protocol owners should retain dated source notes and recheck
current institutional/law requirements at deployment. The planning constraints
above are grounded in these primary or standards-author sources:

- [KAIST IRB guidance](https://research.kaist.ac.kr/page/en/selectPage.do%3Bjsessionid%3DE53962F754C89CA443E69E6BFE1015B4?menuSeq=3061&pageSeq=3591): the IRB,
  not the researcher, determines exemption and approved-scope changes.
- [HHS informed-consent guidance](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/faq/informed-consent/index.html)
  and the [Belmont Report](https://www.hhs.gov/ohrp/regulations-and-policy/belmont-report/read-the-belmont-report/index.html): voluntary participation and protection from undue influence.
- Korean [PIPA official English translation](https://law.go.kr/LSW/lsInfoP.do?chrClsCd=010203&lsiSeq=248613&urlMode=engLsInfoR&viewCls=engLsInfoR)
  and [current Korean statute](https://law.go.kr/lsInfoP.do?ancYnChk=0&lsId=011357): minimum-necessary collection,
  purpose limitation, rights, and destruction; KAIST/legal reviewers must confirm
  the current applicable requirements.
- [WCAG 2.2](https://www.w3.org/TR/WCAG22/) and
  [W3C audiovisual accessibility guidance](https://www.w3.org/WAI/media/av/): keyboard, timing,
  alternatives, captions, transcripts, and descriptions.
- [Standards for Educational and Psychological Testing](https://www.testingstandards.net/uploads/7/6/6/4/76643089/standards_2014edition.pdf)
  and [WWC Procedures and Standards Handbook 5.0](https://ies.ed.gov/ncee/WWC/Docs/referenceresources/Final_WWC-HandbookVer5_0-0-508.pdf): construct access, fairness,
  reliability, consistent administration, and intervention-overalignment concerns.
- [CONSORT 2025](https://www.bmj.com/content/389/bmj-2024-081123),
  [CONSORT-SPI](https://www.equator-network.org/reporting-guidelines/consort-spi/),
  and [TIDieR](https://www.equator-network.org/reporting-guidelines/tidier/):
  participant flow, prespecification, social/psychological intervention reporting,
  and intervention fidelity/completeness.
- [NIST AI RMF Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf),
  [C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html),
  and [UNESCO guidance on generative AI in education and research](https://www.unesco.org/en/articles/guidance-generative-ai-education-and-research): provenance,
  version/parameter records, privacy-aware disclosure, limitations, inclusion, and
  human responsibility. C2PA is vocabulary guidance here, not a compliance claim.

## 16. Consensus changelog

- **Planner draft:** mapped every P1 finding to a notebook, module, companion
  surface, data contract, acceptance criterion, and verification layer; selected
  the modular hybrid option.
- **External evidence pass:** added teaching-only defaults, separate authorization
  scopes, pseudonymous/minimized research data, accessibility targets, measurement
  safeguards, immutable provenance, and institutional/human approval gates.
- **Architect iteration 1:** added the common production kernel; Molab threat model
  and private/exchange/research projections; independent congruence coding and true
  modality-omission gates; exactly-once Marimo reducer; comparison co-design before
  treatment UI; bounded activation/host memory; separate readiness tiers; blinded
  audience packet.
- **Architect iterations 2–3:** normalized the four-section/eight-stage route,
  authorized egress/identity mapping, classroom-versus-research Definition of Done,
  and required-versus-optional architecture activities. Final verdict: `APPROVE`.
- **Critic pass:** confirmed all six comments and 23 acceptance criteria are traced;
  added exact companion-path freeze, RQ3 estimability audit, classroom-only meaning
  of the two-response minimum, and stronger execution staffing/verification handoff.
  Final verdict: `APPROVE`.
- **Deferred cosmetic cleanup:** late-added AC/U-R identifiers may be renumbered only
  when implementation tickets are generated; their current stable identifiers keep
  this PRD/test-spec cross-reference intact.
