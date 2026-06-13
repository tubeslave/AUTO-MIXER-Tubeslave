# Brief: WING MVP Target Defaults Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. The offline
expected-plan, capture, and handoff surfaces must point at the same bounded
read-only target before an operator attempts the real WING preflight.

The current target defaults are:

- `url`: `ws://localhost:8765`
- `channels`: `3`
- `value_types`: `fader,gain,pan`
- `artifact_root`: `artifacts/wing_readonly_preflight`

## Task

Move those target defaults into the shared WING read-only preflight contract and
make no-WING validation fail closed if plan, capture, or handoff commands drift
from the contract.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize live OSC/WING writes.
- Do not close readiness from packet, handoff, or no-WING evidence.
- Preserve existing public defaults.

## Validation

- Compile touched modules.
- Run focused handoff, no-WING, Project Director, direction guard, and WING MVP
  rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and the full backend pytest suite before
  commit.
