# Worksheet migration note

**Document ID:** `worksheet-migration/1.0.0`

The legacy bilingual `avllm_interpretability/WORKSHEET.md` stored one loose row
per run. The current notebook replaces that loose sheet with committed
**Prepare → Run → Reflect** records linked by immutable run IDs.

| Legacy field | Current record | Migration rule |
|---|---|---|
| Setting | prepared run inputs | Copy exact clip/stimulus, prompt, condition, measure, and parameters. |
| Hypothesis before ▶ | prediction | Copy only if it was written before execution; otherwise mark missing. |
| Result | committed result | Link the recorded value and provenance to the run ID; do not retype an estimate. |
| Verdict | reflection | Separate observation, interpretation, limitation, rival explanation, and next control. |
| Control question | next-control field | State what a pass and a failure would each mean. |

Do not backfill timestamps, command IDs, provenance, or predictions that were not
captured. Preserve the old worksheet as a teaching note; do not transform it
into a research record. Start new activity in the notebook's versioned forms.

Audience comments from the worksheet are not valid imports. Use the blinded
audience-reading JSON schema and retain the original downloaded files.
