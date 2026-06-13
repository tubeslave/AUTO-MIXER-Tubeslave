# Brief: WING MVP Request ID Policy Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. The read-only
preflight runner and evidence intake already require request/response identity
correlation for `query_mixer_parameter` responses, but the policy string is
duplicated across runner, evidence, no-WING, and tests.

## Task

Move the WING read-only preflight request-id policy into the shared contract and
make all local validators and fixtures use that contract.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize OSC/WING writes.
- Do not weaken request-id matching for physical readbacks.
- Do not close readiness from command, runner, evidence, or no-WING output.

## Validation

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest before commit.
