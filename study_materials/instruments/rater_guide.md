# Blinded rater guide — candidate

**Guide version:** `common-rater-guide/0.1.0-pilot`  
**Status:** training draft; reliability target and adjudication rule unresolved

## Package shown to raters

Raters receive only a pseudonymous record ID, the permission-authorized artifact
presentation, V1/V2 lineage, creator rationale, and the candidate rubric. They do
not receive allocation, surface label, legal name, email, device data, or private
content hash. Accessibility overlays remain visible and human-correctable.

## First-pass scoring

1. Confirm that the package opens and that captions/transcripts and visual
   descriptions are available. Record a technical or accessibility barrier rather
   than guessing at unavailable evidence.
2. Score every item independently using the item-level evidence and the common
   scale. Do not reward agreement with an automated output or with another rater.
3. Quote or point to the smallest observable artifact/rationale evidence supporting
   the score. Record `not_observable_from_package`, `withdrawn`, or
   `technical_failure` instead of converting missing evidence to zero.
4. Submit the first rating before viewing another rater's score.

## Training and reliability

Training uses non-student anchor packages. The methods owner must approve anchor
selection, number of practice cases, acceptable discrepancies, reliability
statistic, target, threshold, and recalibration rule. Every value is currently
unresolved. No software-generated percentage or correlation may be called adequate
reliability before that decision.

## Adjudication

Keep original ratings immutable. If the approved protocol later permits
adjudication, append a separate adjudication record linking both first ratings,
the evidence discussed, the resolution, the adjudicator pseudonym, and the rule
version. This draft does not choose consensus, averaging, third-rater review, or an
exclusion threshold.

