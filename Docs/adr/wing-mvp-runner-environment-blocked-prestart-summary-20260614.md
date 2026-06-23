# ADR: WING MVP runner environment-blocked prestart summary

## Context

The WING read-only preflight runner requires a passed no-WING prestart
`summary.json` before it opens the backend/WING WebSocket. The no-WING gate can
now fail closed with `environment_blocked_checks` when the local frontend e2e
dev server cannot bind `127.0.0.1:3000`.

## Options considered

1. Keep only the generic invalid-prestart runner block.
2. Let an environment-blocked no-WING summary satisfy the prestart requirement.
3. Keep the runner blocked, but preserve the environment-blocked details and
   operator next step.

## Decision

Use option 3.

## Why this won

It keeps the physical-session boundary strict while giving the operator and
future Goal-mode agents enough context to rerun the correct gate in the correct
environment.

## Rejected alternatives

Generic invalid-prestart output is safe but less actionable. Accepting the
environment-blocked summary would overstate readiness and weaken the prestart
gate.

## Implementation plan

- Copy `environment_blocked_checks` and `environment_blocked_check_details`
  from the prestart summary into the runner's blocked summary.
- Add `prestart_summary_environment_blocked` as a violation before the generic
  status violation.
- Add a specific `operator_next_step`.
- Add a regression test that fails if the WebSocket opens for this input.

## Test plan

- `tests/test_wing_readonly_preflight_runner.py`
- Focused WING MVP rail tests.
- Full standard suite before commit.

## Risks and rollback

Risk: the added detail could be treated as proof by downstream readers. The
runner still records `status: blocked`, `reason:
prestart_summary_missing_or_invalid`, and the existing
`prestart_summary_status_not_passed` violation. Rollback is to remove the
environment-blocked detail copy and specialized next-step mapping.
