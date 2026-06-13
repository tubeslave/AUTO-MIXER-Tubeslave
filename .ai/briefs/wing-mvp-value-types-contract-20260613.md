# Brief: WING MVP Value Types Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. Recent work
centralized read-only target defaults, but the supported read-only value-type
allow-list still existed as local sets in the runner and handoff builder.

## Task

Move the WING read-only preflight supported value types into the shared contract
and make no-WING fail closed if the runner allow-list drifts from that contract.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize OSC/WING writes.
- Do not add new read targets beyond the current `fader`, `gain`, and `pan`
  proof scope.
- Do not close readiness from command, runner, handoff, or no-WING evidence.

## Validation

- Compile touched modules.
- Run focused runner, handoff, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest before commit.
