# ADR: WING read-only runner reports validation violations

## Context

The WING read-only preflight runner validates CLI inputs before it can open a
backend/WING WebSocket. These validation failures are part of the same
fail-closed preflight chain as prestart and expected-plan stops, but they only
reported a human-readable `reason`.

## Options considered

1. Keep validation failures as reason-only blocked summaries.
2. Add a stable `validation_violations` list derived from the validation
   reason.

## Decision

Use option 2. Validation-blocked summaries now include
`validation_violations` with normalized machine-readable markers such as
`validation_value_types_unsupported_eq`.

## Why this won

Review tooling needs a stable field to assert that invalid inputs stopped the
preflight before any WebSocket opened. A human-readable reason is useful for the
operator, but it is weaker than a dedicated violation list for automated gates.

## Rejected alternatives

Leaving the summary as reason-only would keep the UI readable but force review
tools to parse prose.

## Implementation plan

Add a small normalization helper and include its marker in plan-only and real
run validation-blocked outputs. Add regressions for the helper and a real-run
blocked summary.

## Test plan

Run targeted validation regressions, the focused WING/no-WING suite, no-WING
quick, `git diff --check`, and the standard pytest gate.

## Risks and rollback

Risk is low because this only adds fields on already-blocked paths. Rollback is
to remove the helper, summary field, and tests.
