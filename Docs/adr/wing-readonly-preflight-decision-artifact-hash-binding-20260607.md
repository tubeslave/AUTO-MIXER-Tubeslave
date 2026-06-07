# ADR: WING Read-Only Preflight Decision Artifact Hash Binding - 2026-06-07

## Context

The WING read-only preflight flow is file-backed. A real hardware run produces
`summary.json` and `events.jsonl`; the operator reviews those files and records
a decision JSON before supervised fader/gain proof planning can proceed.

Before this change, the decision ledger matched records to candidates by source
id, run id, summary path, and events path. That was enough to identify where the
files live, but not enough to prove the decision still refers to the same file
contents after replacement.

## Options considered

1. Keep path-only binding.
2. Require the operator to copy file hashes into free-form notes.
3. Add machine-validated `summary_sha256` and `events_sha256` fields.

## Decision

Use option 3. File-backed WING read-only preflight evidence rows now expose
SHA256 digests for `summary.json` and `events.jsonl`. Pending decision templates
include the same digests. The review decision ledger rejects decisions whose
hashes are missing or do not match the current candidate.

## Why this won

Byte-level binding makes the operator decision durable and auditable. It also
keeps the gate offline, deterministic, and easy to test.

## Rejected alternatives

Path-only binding is too weak when artifacts can be regenerated at the same
location. Free-form notes are not reliable enough for release gates or proof
planning.

## Implementation plan

1. Add summary/event digest fields to read-only preflight evidence rows.
2. Add the digest fields to artifact review details and pending decision
   templates.
3. Validate digest fields in the decision ledger.
4. Update no-WING template contract expectations, tests, runbook, and reports.

## Test plan

- Focused pytest for WING preflight evidence, review tool, decision ledger,
  decision-review CLI, handoff packet, and no-WING release check.
- Quick no-WING release check.
- Direction guard and full pytest before commit.

## Risks and rollback

Risk: pending review decision templates generated before this change fail closed
with missing digest fields. Regenerate them with the updated review tool.

Rollback is a small revert of the digest fields, decision validation, tests, and
docs. No mixer state is touched.
