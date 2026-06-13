# Brief: WING MVP Allowed Request Types Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. The read-only
preflight proof is only allowed to ask the already-running backend for operator
mode, state, and explicit mixer-parameter readbacks. That allowed request list
was duplicated across runner output, evidence validation, no-WING validation,
and tests.

## Task

Move the WING read-only preflight allowed request types into the shared contract
and make runner, evidence, no-WING, and tests use that contract.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize OSC/WING writes.
- Do not add new request types or read targets.
- Do not close readiness from command, runner, evidence, or no-WING output.

## Validation

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest before commit.
