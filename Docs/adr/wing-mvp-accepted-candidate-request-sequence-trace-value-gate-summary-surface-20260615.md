# ADR: WING MVP Accepted-Candidate Request-Sequence Trace Value-Gate Summary Surface

## Context

The supervised WING MVP cannot advance to supervised fader/gain proof until a
real read-only WING preflight candidate has been captured and manually accepted.
The no-WING proof plan must still make the preconditions visible and fail closed
when the accepted-candidate request sequence does not match the expected plan.

## Options Considered

1. Reuse the existing no-WING proof-plan value-gate summary surface.
2. Add a separate request-sequence checker.
3. Defer all request-sequence validation to manual review.

## Decision

Use the existing no-WING proof-plan value-gate summary surface for the accepted
candidate request-sequence trace. The surface records required fields,
violations, pass/fail state, self-check mismatches, and explicit no-authority
flags.

## Why This Won

The existing no-WING proof-plan checker already has the full packet context and
can compare the rendered summary surface to the canonical next proof-plan
fields. Keeping this there avoids a second source of truth.

## Rejected Alternatives

The separate checker was rejected as duplicate proof logic. Manual-only review
was rejected because it would allow a malformed request-sequence trace to appear
as operator prose without a fail-closed machine gate.

## Implementation Plan

1. Keep `tools/no_wing_release_check.py` as the implementation authority.
2. Validate the regression that fails when request-sequence value-gate summary
   fields drift from the next proof plan.
3. Record no-WING quick and artifact-manifest verification evidence.

## Test Plan

- Compile the no-WING checker and focused tests.
- Run the focused request-sequence value-gate summary-surface regression.
- Run no-WING quick into `/tmp`.
- Verify the generated artifact manifest.
- Run focused and full pytest before commit.

## Risks And Rollback

This does not contact WING or grant live authority. If the surface blocks valid
no-WING packets, revert this documentation package and adjust the existing
summary-surface checker in a separate implementation change with focused tests.
