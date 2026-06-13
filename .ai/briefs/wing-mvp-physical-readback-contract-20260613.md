# Brief: WING MVP Physical Readback Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. The runner and
evidence intake both decide whether a readback is physical enough to count as
real proof. Those status/source identities must not drift.

## Task

Move the WING read-only preflight physical readback statuses and sources into
the shared contract, use them in runner and evidence intake, and make no-WING
fail closed if either surface diverges.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize OSC/WING writes.
- Preserve the current physical readback identity values.
- Do not close readiness from command, runner, evidence, handoff, or no-WING
  artifacts.

## Validation

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest before commit.
