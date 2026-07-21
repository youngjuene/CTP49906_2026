# WP-0 RQ3 estimability audit

**Gate verdict:** `STRUCTURAL ESTIMABILITY PASS; RESEARCH RELEASE BLOCKED`
**Manifest:** `study_materials/wp0/rq3_design_manifest.json`
**Manifest SHA-256:** `530eb6ebbf9ce2ad2633b96ce84c35d2c086b1e6b771c8c9e44cba99afc5149c`
**Design schema:** `wp0.rq3-design-manifest/1.0.0`

## Scope boundary

This exact, standard-library audit evaluates structural main-effect rank,
factor overlap, counterbalancing, and single-nuisance perfect confounding.
A structural pass is **not** evidence of adequate power, construct validity,
successful manipulation, licensing, or Molab/model modality-omission
feasibility. Those remain separate mandatory WP-0 gates.

Operation contrasts necessarily use `technical_operation` as their defining
factor. The audit does not nonsensically treat that defining factor as a
nuisance; instead it proves exact full rank for operation alongside the other
main effects and requires every declared non-defining nuisance level on both
sides of each contrast.

## Structural checks

| Check | Status | Evidence |
|---|---:|---|
| `schema.top_level` | **PASS** | Required top-level fields are present  |
| `ids.source_artifact_id` | **PASS** | All 4 source_artifact_id values are unique  |
| `ids.target_artifact_id` | **PASS** | All 4 target_artifact_id values are unique  |
| `ids.stimulus_id` | **PASS** | All 128 stimulus_id values are unique  |
| `ids.assignment_id` | **PASS** | All 256 assignment_id values are unique  |
| `ids.assignment_slot` | **PASS** | All block/sequence/order slots are unique  |
| `schema.factor_columns` | **PASS** | Every assignment contains consistent factor, offset, and check columns  |
| `coverage.operations` | **PASS** | All eight operation families are present  |
| `semantics.operation_kinds` | **PASS** | Silence/neutral controls, true modality omission, and edge knockout are distinct  |
| `coverage.factor_levels` | **PASS** | Required factor levels are covered {"block_levels":["BLOCK-01","BLOCK-02"],"congruence_levels":["congruent","incongruent"],"order_levels":[1,2,3,4,5,6,7,8],"source_levels":4,"target_levels":4,"temporal_offset_levels_ms":[-500,-250,250,500]} |
| `variation.within_block` | **PASS** | Each block varies operation, congruence, and order  |
| `variation.within_pair` | **PASS** | Every matched pair-set crosses congruence with source and target identity  |
| `reuse.source` | **PASS** | Every source is reused across all operations and congruence levels  |
| `balance.order` | **PASS** | Operation and congruence are exactly balanced by order {"congruence_by_order_counts":[16],"expected_congruence_order_cells":16,"expected_operation_order_cells":64,"observed_congruence_order_cells":16,"observed_operation_order_cells":64,"operation_by_order_counts":[4]} |
| `rank.main_effects` | **PASS** | Exact main-effect design matrix has full column rank {"aliased_columns":[],"columns":23,"exact_rank":23,"factors":["congruence_level","technical_operation","source_artifact_id","target_artifact_id","block_id","presentation_order"],"rows":256} |
| `contrasts.single_nuisance_overlap` | **PASS** | No declared contrast is perfectly determined by any declared single nuisance factor ["C01-congruence-within-selected-operations","C02-audio_swap_duration_matched-versus-original","C03-temporal_offset-versus-original","C04-audio_silence_control-versus-original","C05-video_neutral_control-versus-original","C06-audio_omitted_model_input-versus-original","C07-video_omitted_model_input-versus-original","C08-direct_attention_edge_knockout-versus-original","C09-positive-versus-negative-temporal-offset"] |
| `honesty.synthetic_license` | **PASS** | Synthetic fixtures are explicitly non-student and not course-release licensed  |

## Exact rank/alias diagnostic

- Rows: `256`
- Treatment-coded columns: `23`
- Exact rational rank: `23`
- Aliased columns: `[]`
- Factors: `['congruence_level', 'technical_operation', 'source_artifact_id', 'target_artifact_id', 'block_id', 'presentation_order']`

Full column rank supports estimability of the declared additive main-effect
design columns. It does not prove estimability of arbitrary interactions or
a future analysis model not represented here.

## Declared-contrast overlap

| Contrast | Status | LHS n | RHS n | Defining factor(s) |
|---|---:|---:|---:|---|
| `C01-congruence-within-selected-operations` | **PASS** | 64 | 64 | congruence_level |
| `C02-audio_swap_duration_matched-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C03-temporal_offset-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C04-audio_silence_control-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C05-video_neutral_control-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C06-audio_omitted_model_input-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C07-video_omitted_model_input-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C08-direct_attention_edge_knockout-versus-original` | **PASS** | 32 | 32 | technical_operation |
| `C09-positive-versus-negative-temporal-offset` | **PASS** | 16 | 16 | offset_polarity |

## Separate or unauditable gates

| Gate | Status | Why this audit cannot resolve it |
|---|---:|---|
| `statistical_power` | **NOT_AUDITED** | No approved sample size, variance/effect assumptions, or pilot outcomes exist. |
| `construct_validity` | **NOT_AUDITED** | Congruence labels are synthetic scheduling codes pending independent coding/pretest. |
| `manipulation_success` | **NOT_AUDITED** | All empirical manipulation checks are planned_not_run. |
| `target_platform_modality_feasibility` | **NOT_AUDITED** | This audit does not execute Molab processor/model audio- or video-omission paths. |
| `licensed_course_stimuli` | **BLOCKED** | Fixtures are synthetic feasibility records, not a licensed final course set. |

## Minimal redesign / next evidence

No structural redesign is required for the synthetic counterbalancing schedule. Replace synthetic sources/targets with licensed, independently coded assets while preserving the frozen factor crossing and order schedule; then rerun this audit. Do not change a factor level in isolation.

WP-0 must remain non-complete while any mandatory licensing, empirical
manipulation, target-platform omission, resource, governance, or comparison-
arm gate remains unresolved. This RQ3 lane does not waive those gates.
