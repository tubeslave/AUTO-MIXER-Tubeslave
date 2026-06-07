# ADR: Supervised Proof Plans Carry Review Decision Collision Diagnostics

## Context

The next WING supervised fader/gain proof is generated only after a clean
read-only preflight candidate and accepted operator decision. If the decision
ledger finds duplicate or conflicting decisions, the proof plan blocks correctly
but should also expose the exact records and sources that need operator cleanup.

## Options Considered

1. Keep only aggregate invalid counts in the proof plan.
2. Re-run decision collision analysis inside the proof-plan builder.
3. Propagate ledger diagnostics into the proof-plan summary.

## Decision

Use option 3. The proof plan copies invalid decision record ids, duplicate
source ids, and conflicting source ids from the review-decision ledger summary.

## Why This Won

The ledger remains the single authority for decision validation while the proof
plan becomes actionable for the operator. No execution authority is added.

## Rejected Alternatives

- Counts alone are too vague during a real hardware handoff.
- Duplicating ledger analysis risks inconsistent blocking behavior.

## Implementation Plan

- Add decision collision diagnostic fields to `supervised_wing_proof_plan`
  summary.
- Teach no-WING validation to compare those fields against the ledger overlay.
- Render the fields in no-WING summaries.
- Add focused tests for blocked proof plans with conflicting decisions.

## Test Plan

- Focused supervised proof plan tests.
- Focused no-WING summary/validator tests.
- Focused WING review decision tests.
- No-WING quick release check.

## Risks And Rollback

Rollback would hide the exact decision records that block supervised proof
planning. It would not grant authority, but it would make the operator workflow
less predictable.
