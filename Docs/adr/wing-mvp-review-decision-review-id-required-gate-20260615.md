# ADR: WING MVP Review Decision Review-ID Required Gate

## Context

The WING read-only accepted-decision template requires `review_id`. The ledger
used an internal fallback identifier when a record omitted the field, so a
manual decision could be valid without carrying its operator-facing decision ID.

## Options Considered

1. Leave fallback IDs as sufficient for valid records.
2. Reject records that omit an explicit `review_id`.
3. Reject records unless `review_id` exactly matches the generated template
   pattern.

## Decision

Use option 2. The ledger must fail closed when an operator decision omits
`review_id`.

## Why This Won

The field is part of the required decision contract. A fallback can still label
an invalid row internally, but it must not satisfy the manual accepted-decision
schema.

## Rejected Alternatives

Fallback-only validation was rejected because it lets incomplete manual records
advance. Exact pattern matching was rejected for this step because the current
manual flow does not require generated-only review IDs.

## Implementation Plan

- Add an explicit missing-`review_id` violation in the ledger validator.
- Add a regression test for missing `review_id`.
- Update the read-only runbook and rails test.

## Test Plan

- Run targeted ledger and runbook rails tests.
- Run focused WING/no-WING tests.
- Run full standard pytest.
- Run no-WING quick, memory lint, Project Director, and `git diff --check`.

## Risks And Rollback

Risk is limited to invalidating incomplete manual records. Rollback is removing
the missing-ID violation, but that would restore the incomplete decision path.
