# ADR: WING read-only runner reports invalid expected-plan violations

## Context

The WING read-only preflight runner requires a plan-only expected-plan artifact
before a real run can open a backend/WING WebSocket. Missing and mismatched
expected plans already report `expected_plan_violations`, but malformed or
unreadable expected-plan files only reported `expected_plan_error`.

## Options considered

1. Keep malformed expected-plan failures as error-only summaries.
2. Add a machine-readable `expected_plan_violations` marker for malformed
   expected-plan failures.

## Decision

Use option 2. Invalid expected-plan artifacts now include
`expected_plan_violations: ["expected_plan_missing_or_invalid"]`.

## Why this won

Review tooling and operator checklists can treat every expected-plan stop the
same way: inspect `expected_plan_match`, `expected_plan_violations`, and then
optional error detail. The runner still blocks before any WebSocket.

## Rejected alternatives

Keeping only `expected_plan_error` was readable for humans, but weaker for
machine checks that need a stable violation list before hardware preflight.

## Implementation plan

Add the violation marker in the expected-plan load exception path and extend the
existing invalid expected-plan regression.

## Test plan

Run the targeted invalid expected-plan regression, the focused WING/no-WING
suite, no-WING quick, `git diff --check`, and the standard pytest gate.

## Risks and rollback

Risk is low because this only adds a summary field on an already blocked path.
Rollback is to remove the extra violation field and its assertion.
