# Instructor runbook

**Document ID:** `instructor-runbook/1.0.0`
**Audience:** classroom facilitators
**Status:** teaching rehearsal; not research-ready

## Before class

1. Use a clean checkout and verify the hashes in
   [replay_manifest.json](replay_manifest.json). A mismatch is a stop condition,
   not a prompt to rewrite the manifest.
2. Start the treatment notebook in **Saved course replay**. Do not require a GPU,
   model download, or network connection for the required guided observations.
3. Open the separate
   `audience/CTP49906_audience_response_molab.py` surface and rehearse one
   packet-to-reading download with synthetic data.
4. Confirm that the instructor and audience surfaces have a usable keyboard
   order: language, packet upload, packet validation, response fields in reading
   order, commit, then download. Check visible focus, labels, headings, status
   messages, and the text alternatives in
   [visual_alternatives.json](visual_alternatives.json).
5. Review the Molab/container retention statement available to your institution.
   Record unresolved retention or cross-border questions; do not infer an answer.
6. Prepare an offline transfer method if classroom network access is unreliable.
   The app does not automatically send student data.

## Required classroom sequence

- Orient students to Teaching mode, the browser-to-Molab boundary, private
  downloads, and the evidence/inference distinction.
- Keep human-correctable captions, transcripts, and descriptions as access
  supports. They are not model inputs unless a later activity explicitly and
  visibly selects them.
- Have each creator export only a permission-valid blinded presentation packet.
  Inspect the validation result before distributing it.
- Collect audience responses independently. A respondent must not see the
  creator explanation, condition, machine labels, or an earlier response.
- Import at least two schema-valid readings with distinct audience and session
  pseudonyms for the same `exchange_artifact_id`.
- Reveal creator and machine readings only after the import validator reports a
  complete blinded exchange. Record any deviation; reveal cannot be reversed.
- Use the disagreement reflection: note overlap, divergence, an unrepresented
  reading, and relevant culture, access, prompt, and training-data unknowns.
  Agreement is not correctness and disagreement is not error.
- Register the creator's revised explanation and V2 before private portfolio
  export.

Two distinct readings are a classroom comparison minimum only. They do not
justify population, validity, saturation, or research-sampling claims.

## Failure and recovery

- Preserve the last valid packet/reading files. Never edit a checksum or
  pseudonym merely to make validation pass.
- Correct the source packet or form, generate a new valid file, and retry.
- A duplicate respondent/session must be replaced by an independent reading.
- A response obtained after reveal is a recorded deviation and cannot count as a
  blinded response.
- If saved replay hashes fail, stop using the pack. If live GPU work fails, stay
  on saved replay and keep personalized live actions unavailable.
- Do not copy response files into Git, chat, shared drives, or instructor systems
  without the corresponding permission and approved procedure.

## Evidence to retain for a candidate release

Record the exact repository commit, replay manifest digest, model/stimulus
revision, dependency environment, route time, help events, access barriers,
translation discrepancies, packet/import failures, and deviations. Automated
tests cannot replace human accessibility, localization, audience-validity,
instrument, ethics, or research-methods review.
