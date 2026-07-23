# Shared technical contracts

**Contract:** `wp0.shared-contracts/1.0.0`
**Status:** frozen technical interface; implementation and human approval pending

`shared_contracts.json` is the machine-readable WP-0 boundary for later shared
modules. It is not WP-1 code, an approved study protocol, or an immutable course
release. The protocol and release identifiers therefore remain `null`.

## Identity and versioning

Filenames are labels, not identities. Content hashes associate exact bytes and
make change history detectable; they do not prove truth, authorship, provenance,
ownership, or permission. Artifact, stimulus, pair, run, reflection, audience,
exchange, and study identifiers are stable and may never be reused, retargeted, or
silently rewritten. Parent/version links are immutable.

Schema version, study-protocol version, and course-release version are distinct.
Freezing a technical schema does not imply that a human approved the study or that
the course dependencies and assets are release-ready.

## Append-only process

Forms prepare commands; only a committed command with a unique `command_nonce`
may append. Exact replay is idempotent, conflicting nonce reuse is rejected, and a
result attaches only to its committed run. A reflection snapshots the initial
explanation. An edit creates a linked revision while prior serialized records
remain byte-unchanged. Withdrawal appends a state/tombstone rather than rewriting
history. A later exactly-once reducer must enforce these invariants.

## Teaching and Research modes

`Teaching` is the initial and fail-safe mode. Missing, invalid, expired,
over-permissive, unapproved, wrong-schema, or wrong-release configuration cannot
enter Research mode. An in-notebook acknowledgement is never informed consent.
Research fields and destinations are unavailable in Teaching mode.

Process analysis, classroom/audience sharing, quotation/reproduction, and future
reuse are independent permission scopes. One never implies another. The only
authorized student-data egress classes are:

1. a user-initiated private download to the user's device;
2. a permission-valid blinded audience presentation packet; and
3. an approved, minimized Research-mode projection to its configured destination.

Repository, dependency, and model reads are expected service inputs, not
student-data egress. Automatic or unapproved student-data transmission is forbidden.

## Three projections

- **Private portfolio:** creator-controlled; may retain private hashes, aliases,
  exact local event history, and selected private media.
- **Audience packet:** opaque `exchange_artifact_id` and the frozen allowlist only.
  It excludes creator intention, model output, condition assignment, private IDs or
  hashes, and unapproved raw media. Reveal is monotonic: once reveal occurs,
  blindness cannot be restored by changing a flag.
- **Research projection:** disabled until approval. It is a pure, versioned
  transformation, emits a redaction report, and does not mutate private bytes. Its
  maximum output is an approved allowlist of structural/coded fields. Raw hashes,
  filenames, raw media, exact UTC, device/platform metadata, direct identifiers,
  uncontrolled demographics, and non-allowlisted free text are always forbidden.

`session_pseudonym` means pseudonymous, not anonymous. Only the approved Research
projection may map private/exchange identities to an investigator-issued
`study_artifact_id`.
