# Brief: WING MVP Request Response Map Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. Recent work
centralized allowed request types and expected response types, but the mapping
from each request type to its accepted response types still lived in runner and
evidence code.

## Task

Move the WING read-only preflight request-to-response mapping into the shared
contract and make runner, evidence, and tests use that mapping.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize OSC/WING writes.
- Do not add new request types, response types, or read targets.
- Do not close readiness from command, runner, evidence, or no-WING output.

## Validation

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest before commit.
