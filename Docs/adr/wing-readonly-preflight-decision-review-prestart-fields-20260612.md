# ADR: WING Read-Only Preflight Decision Review Prestart Fields

## Context

The artifact review report displays the prestart backend-port contract for valid
WING read-only preflight candidates. The decision review report still only
listed accepted/rejected source ids and did not show the candidate prestart
boundary in the final offline review surface.

## Options considered

1. Keep prestart details only in artifact review.
2. Add candidate prestart rows to the decision ledger and decision review report.
3. Add the prestart fields to each operator decision record.

## Decision

Use option 2. The ledger summary now carries `candidate_reviews`, and the
decision review markdown renders the candidate prestart contract.

## Why this won

It makes the final decision review self-contained without expanding the operator
decision record schema or granting any authority.

## Rejected alternatives

Option 1 leaves the final review surface incomplete. Option 3 would burden
operator-authored records with data already present in validated evidence.

## Implementation plan

- Add candidate prestart rows to `backend/wing_readonly_preflight_review_decision.py`.
- Render those rows in `tools/review_wing_readonly_preflight_decisions.py`.
- Update decision ledger and decision review tests.

## Test plan

- Focused decision review tests.
- Adjacent WING/no-WING tests.
- Full backend suite.

## Risks and rollback

Risk: decision markdown is longer. Rollback: remove `candidate_reviews` from the
ledger summary and markdown. No hardware access or live write behavior is
changed.
