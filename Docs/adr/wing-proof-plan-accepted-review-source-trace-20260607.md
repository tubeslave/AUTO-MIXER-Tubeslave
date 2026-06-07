# ADR: Supervised Proof Plans Preserve Accepted Review Source IDs

## Context

The supervised WING MVP requires a clean read-only preflight artifact and an
accepted operator review decision before planning a fader/gain proof. Counts
alone prove that some decision exists, but the proof plan should identify the
accepted read-only artifact source.

## Options Considered

1. Keep only decision counts in the proof plan.
2. Copy full decision records into the proof plan.
3. Copy accepted/rejected source ids from the decision ledger summary.

## Decision

Use option 3. The proof plan carries accepted and rejected source ids, and
no-WING validation ensures they match the review-decision ledger overlay.

## Why This Won

It preserves traceability without duplicating ledger validation or granting
execution authority.

## Rejected Alternatives

- Counts alone are weak during operator handoff.
- Full record copies would make the proof plan noisy and risk drift from the
  ledger.

## Implementation Plan

- Add accepted/rejected source id fields to `supervised_wing_proof_plan`
  summary.
- Extend no-WING validation and summary rendering.
- Add focused tests for propagation and stale-source rejection.
- Update the supervised proof runbook and documentation index.

## Test Plan

- Focused supervised proof plan tests.
- Focused no-WING tests.
- WING review-decision tests.
- No-WING quick release check.

## Risks And Rollback

Rollback would keep proof planning safe but less traceable. It should only be
used if the decision ledger source-id contract changes.
