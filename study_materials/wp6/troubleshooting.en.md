# Troubleshooting guide

**Document ID:** `troubleshooting/1.0.0`

## Audience surface

- **Packet rejected / missing field:** request a new packet from the creator.
  Do not add fields manually.
- **Checksum mismatch:** stop. The presentation reference changed or the packet
  is corrupt. Regenerate from the validated creator workflow.
- **Creator/model/private field rejected:** this is the intended privacy guard.
  Remove the source leak by regenerating an allowlisted packet.
- **Response cannot commit:** complete every labelled field, attest blindness,
  use non-identifying pseudonyms, and keep the packet in blinded state.
- **Duplicate audience or session:** collect a genuinely independent response.
  Renaming the same person/session does not make it independent.
- **Response collected after reveal:** retain it only as a documented deviation;
  it cannot count toward the two blinded classroom readings.
- **No download appears:** the draft has not validated. Correct the displayed
  error and submit again; editing draft controls alone creates no record.

## Replay and notebook

- **No GPU:** choose Saved course replay. Live-only controls should remain
  unavailable with an explanation.
- **Replay checksum mismatch:** stop using the replay pack and compare every
  file with [replay_manifest.json](replay_manifest.json). Do not update expected
  hashes unless an authorized regeneration creates a new manifest version.
- **Replay says metadata mismatch:** restore the matching coded stimulus,
  frame count, prompt, model revision, and settings or regenerate a new pack.
- **Uploaded media fails preflight:** correct the source media; do not bypass
  duration, stream, resolution, frame-rate, or decoded-memory checks.
- **Export invalid/interrupted:** preserve prior records, remove any partial
  output, correct the reported link/field issue, and retry with a new command.

## Access and language

- Use keyboard order from the instructor runbook; report missing focus, labels,
  headings, or status announcements as release-blocking defects.
- Use [visual_alternatives.json](visual_alternatives.json) and its downloadable
  data when a visual is unavailable.
- If English and Korean wording differ in meaning, stop the affected language
  route and record a translation discrepancy. Automated key parity is not a
  qualified translation review.

The material is not research-ready. Contact the accountable instructor or human
reviewer for governance, accessibility, localization, audience, or instrument
questions instead of improvising approval.
