# ADR: WING Read-Only Pre-WebSocket Test Execution Denial

## Context

The first WING MVP blocker remains real read-only connection proof. The
no-WING and artifact-manifest contracts require `executes_tests: false`, but
runner-produced offline or pre-WebSocket surfaces did not all expose that
top-level denial.

## Options considered

1. Rely on the fact that the runner command is not a test command.
2. Add explicit `executes_tests: false` to runner pre-WebSocket surfaces.
3. Add the field only to no-WING handoff packet summaries.

## Decision

Use option 2. Offline plans, plan-only blocked outputs, and real-run
pre-WebSocket summaries now expose `executes_tests: false`.

## Why this won

The operator proof chain should not infer side-effect boundaries from command
names. A direct field lets downstream gates verify that plan-only,
validation-only, and early blocked preflight artifacts are not test execution
artifacts and do not close readiness.

## Rejected alternatives

- Option 1 leaves an important safety property implicit.
- Option 3 keeps the runner artifact surface weaker than the handoff/no-WING
  contract it is supposed to support.

## Implementation plan

- Add `executes_tests: false` to the offline preflight plan.
- Add the same field to plan-only blocked validation and prestart outputs.
- Add the same field to the base real-run pre-WebSocket summary.
- Assert representative plan-only, expected-plan, validation-only, and blocked
  validation surfaces.

## Test plan

- Targeted runner tests for the changed surfaces.
- no-WING quick release check.
- Focused WING preflight, no-WING, handoff, and Project Director tests.
- `git diff --check`.
- Full pytest.

## Risks and rollback

The risk is low because this only makes an existing command boundary explicit.
Rollback is to remove the field and assertions.
