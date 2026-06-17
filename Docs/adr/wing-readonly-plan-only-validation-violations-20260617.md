# ADR: WING read-only plan-only output preserves validation violations

## Context

The read-only WING runner supports `--plan-only` so an operator can review the
expected request sequence before a real backend/WING WebSocket run. When
plan-only inputs are invalid, the output must be as machine-readable as the
real-run validation-blocked summary.

## Options considered

1. Test only real-run validation summaries.
2. Add a regression that plan-only validation-blocked output includes
   `validation_violations`.

## Decision

Use option 2. The plan-only invalid channel-confirmation regression now asserts
the stable validation marker.

## Why this won

Plan-only is part of the preflight evidence chain. If its blocked output loses
the validation marker, automated review can miss why the operator never got a
valid expected plan.

## Rejected alternatives

Relying on the real-run validation test was too indirect because the plan-only
branch constructs a separate output object.

## Implementation plan

Extend the existing plan-only invalid channel-confirmation test to assert
`validation_violations`.

## Test plan

Run the targeted plan-only regression, the focused WING/no-WING suite,
no-WING quick, `git diff --check`, and the standard pytest gate.

## Risks and rollback

Risk is minimal because this is test coverage for already-present output.
Rollback is to remove the assertion and ADR.
