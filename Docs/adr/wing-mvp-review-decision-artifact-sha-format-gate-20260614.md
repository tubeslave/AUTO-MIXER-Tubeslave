# ADR: WING MVP Review Decision Artifact SHA Format Gate

Date: 2026-06-14
Status: Accepted

## Context

WING read-only review decisions are operator-edited records that pin a captured
summary and event log by SHA-256. These records gate later supervised proof
planning, so malformed artifact digest values must fail before a decision can
be accepted.

## Options considered

1. Keep only stale/mismatch checks against the candidate digests.
2. Reuse `is_sha256_hexdigest()` for review-decision `summary_sha256` and
   `events_sha256`.
3. Defer digest-format checks to downstream supervised proof planning.

## Decision

Use option 2. Review-decision records now fail closed on malformed summary and
event-log artifact digests.

## Why this won

The decision ledger is the manual acceptance boundary. It should reject
placeholder or malformed digest strings directly and keep the operator feedback
specific.

## Rejected alternatives

- Mismatch-only checks: they catch stale values but do not name malformed
  inputs.
- Downstream-only checks: allow a bad manual decision to look accepted earlier
  in the workflow.

## Implementation plan

- Import the shared SHA-256 helper into the review-decision ledger.
- Emit `review_decision_summary_sha256_invalid` and
  `review_decision_events_sha256_invalid` for malformed manual digests.
- Add focused regression coverage.
- Keep readiness/live authority flags denied.

## Test plan

- `PYTHONPATH=backend python -m pytest -q
  tests/test_wing_readonly_preflight_review_decision.py`
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Risks and rollback

Rollback removes the digest-format checks, test, and report. The change is
offline ledger validation only; it does not contact hardware, mutate console
state, grant live authority, or close readiness.
