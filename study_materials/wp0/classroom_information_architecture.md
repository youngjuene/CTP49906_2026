# Frozen classroom information architecture

**Contract ID:** `classroom-ia/1.0.0`
**Status:** frozen WP-0 interface; implementation not started

The student-facing treatment surface has exactly four top-level sections and
exactly eight numbered learning stages. The IDs below are immutable within this
contract version. Display text may be translated through a versioned content-key
map, but translation must not change ID, order, hierarchy, or meaning.

| Order | Immutable section ID | Exact top-level heading | Immutable stage ID | Exact stage heading |
|---:|---|---|---|---|
| 1 | `section.1.prepare-project` | Prepare your project | `stage.1.1.orient` | 1.1 Orient — Make, test, revise |
| 2 | `section.1.prepare-project` | Prepare your project | `stage.1.2.compose` | 1.2 Compose — Plan your first cut |
| 3 | `section.1.prepare-project` | Prepare your project | `stage.1.3.register-v1` | 1.3 Register — First cut (V1) |
| 4 | `section.2.guided-demonstration` | Guided demonstration | `stage.2.1.observe` | 2.1 Observe — Shared reference |
| 5 | `section.3.exploratory-playground` | Exploratory playground | `stage.3.1.experiment` | 3.1 Experiment — Change one thing at a time |
| 6 | `section.3.exploratory-playground` | Exploratory playground | `stage.3.2.compare` | 3.2 Compare — Three readings of one work |
| 7 | `section.3.exploratory-playground` | Exploratory playground | `stage.3.3.revise-v2` | 3.3 Revise — Explanation and second cut (V2) |
| 8 | `section.4.synthesis-architecture-challenge` | Synthesis and architecture challenge | `stage.4.1.synthesize` | 4.1 Synthesize — Portfolio and architecture challenge |

No fifth top-level section, ninth stage, unnumbered peer stage, or renumbering is
permitted without a new contract version and a renewed treatment/comparison,
accessibility, and test-spec review.

## Stage obligations

1. **1.1 Orient:** learning outcomes, time map, evidence/inference legend,
   Teaching/Research availability, Molab boundary, and live/replay provenance.
2. **1.2 Compose:** audience and artistic intention, storyboard and sound-image
   relation, cultural references, source/license provenance, accessibility plan,
   common editor/export preset, and planning card.
3. **1.3 Register V1:** upload/preflight, preview, immutable content identity,
   media facts, edit-decision manifest, concept tags, and egress disclosure.
4. **2.1 Observe:** shared saved/live reference, layer probe, attention view,
   direct attention-edge knockout, teacher-forced delta, and a nearby
   interpretation checkpoint for each measure.
5. **3.1 Experiment:** one coded paired manipulation, pre-run prediction,
   trajectory/answer inspection, result, limitation, rival explanation, and next
   control.
6. **3.2 Compare:** at least two distinct blinded audience readings before
   creator/model reveal, followed by situated comparison of culture,
   accessibility, prompt design, training unknowns, and label choice.
7. **3.3 Revise V2:** named evidence trigger, revised explanation, externally
   edited linked V2, and rerun of the preselected comparison.
8. **4.1 Synthesize:** human- and machine-readable portfolio export plus one
   bounded architecture proposal grounded in findings, measurement limits,
   alternatives, and a next test.

## Progressive disclosure contract

### Required

All eight stages remain on the required route. The minimum completion path is:
orientation; intention card; V1; guided results; one paired manipulation; two
valid independent audience responses; explanation revision; linked V2; and
portfolio export. Each of the four top-level sections ends with exactly one visible
`Checkpoint` card containing completion state and the next action.

### Choice

Within Stage 3.1, a student chooses exactly one initial question family—congruence,
swap, timing, or omission/control—and one model measure to pursue more deeply.
Choice does not authorize an unregistered condition or allow congruence to be
inferred from the technical operation.

### Advanced

Custom layer bands, attention-head rules, feature/token intervention spikes, and
deeper architecture redesign are collapsed, visibly optional, and never
prerequisites for any required checkpoint. The short Stage 4.1 proposal is
required; the engineering extension is not.

## Student-surface boundary

Feature IDs, PRD/acceptance codes, release-management status, reviewer language,
implementation history, and research-allocation logic must not render in student
cells. Method caveats appear beside the result they qualify as concise
`What this can / cannot show` callouts.
