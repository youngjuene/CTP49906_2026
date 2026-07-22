# Privacy and data dictionary

**Document ID:** `privacy-data-dictionary/1.0.0`  
**Default mode:** Teaching

## Boundaries and permissions

Teaching mode fails safe. It does not expose study allocation, research
destinations, or research export. Process-data analysis, classroom/audience
sharing, quotation/reproduction, and future reuse are separate permissions; one
never implies another.

The notebook does not automatically send student media or records. Authorized
student-data movement is limited to a user-initiated private download, a
permission-valid blinded audience packet, or—only after separate approval—a
minimized Research projection. Package, repository, and model reads are service
inputs, not permission to transmit student data.

## Core audience fields

| Field | Meaning | Privacy rule |
|---|---|---|
| `exchange_artifact_id` | Opaque link between one presentation packet and readings | Must not reuse a content hash, filename, path, or private artifact ID. |
| `presentation_asset` | Allowlisted reference, media type, duration, and optional access-support references | No raw media or creator/model fields. |
| `presentation_checksum` | SHA-256 of the canonical presentation reference | Detects change; does not prove truth, authorship, permission, or ownership. |
| `sharing_permission_scope` | Independent classroom/audience sharing permission | Must be present and valid before packet export. |
| `response_id` | Stable reading record ID | Unique within an exchange. |
| `audience_pseudonym` | Classroom pseudonym | Pseudonymous, not anonymous. |
| `respondent_session_pseudonym` | Distinct session identity used to detect repeats | Must not be a name, email, path, or device identifier. |
| `local_created_at_utc` | Time retained in the private/classroom reading | Not allowed in the minimized Research projection. |
| `open_interpretation` | Respondent's own reading | Preserve verbatim; never substitute an automated answer. |
| `accessibility_barriers` | Barrier report or “prefer not to answer” | Treat as sensitive free text and share only within the authorized route. |
| `blindness_attestation` | Whether prohibited information was unseen | False or post-reveal responses do not count as valid blinded readings. |
| `withdrawn` | Append-only withdrawal state | Withdrawal does not rewrite earlier bytes. |

The blinded packet excludes creator intention, model output, condition
assignment, private IDs/hashes, and unapproved raw media. Reveal is monotonic.
A creator/model comparison may be assembled only after two distinct valid
blinded readings for the same exchange validate.

Institutional retention, deletion, incident response, cross-border processing,
and approved research destinations remain human governance decisions. This
dictionary is not research-ready approval.
