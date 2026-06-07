# ADR: WING Preflight Review Decisions Fail Closed On Collisions

## Context

The supervised WING MVP requires a clean read-only preflight artifact and a
manual review decision before any supervised fader/gain proof. The decision root
may contain multiple JSON files. If more than one valid record targets the same
artifact, the ledger should not infer which operator intent is authoritative.

## Options Considered

1. Accept the first valid decision record discovered.
2. Accept the latest UTC timestamped decision record.
3. Treat duplicate decision records for a candidate as invalid and require
   replacement.

## Decision

Use option 3. The ledger marks duplicate records for the same
`artifact_source_id` invalid. If their decisions differ, it also records a
conflicting-decision violation.

## Why This Won

This preserves the safety boundary between read-only proof review and the next
supervised proof. A rejected decision cannot be masked by a separate accepted
record, and stale files must be cleaned up explicitly.

## Rejected Alternatives

- Discovery order is not a safety mechanism.
- Timestamp ordering depends on clock trust and still hides conflicting human
  review intent.

## Implementation Plan

- Add post-validation collision detection to the review-decision ledger.
- Expose duplicate and conflicting source ids in the decision-review report and
  no-WING release detail.
- Add focused regression tests for duplicate accepted/rejected records.
- Update the WING read-only preflight runbook and documentation index.

## Test Plan

- Focused decision ledger tests.
- Focused decision-review CLI tests.
- Focused WING preflight regression tests.
- No-WING quick release check.

## Risks And Rollback

Rollback would allow ambiguous decision roots to produce an accepted-readiness
candidate. It should only happen if operator workflow is changed to store
decisions in a single authoritative append-only ledger with its own collision
policy.
