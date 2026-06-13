# Brief: WING MVP Live Authority Disallowed By Packet Next Action

## Context

The first non-closed WING MVP blocker remains `WING read-only connection proof`.
The offline no-WING handoff packet already emits `live_authority_granted: false`,
but the current blocker action and autoloop docs do not yet require that field
before supervised proof planning.

## Task

Promote `live_authority_granted: false` into the WING MVP rails for the current
read-only handoff boundary.

## Constraints

- Do not contact WING hardware.
- Do not add live OSC/WING writes.
- Do not close readiness from offline packet evidence.
- Keep the change limited to docs, rails, council artifacts, and regression
  tests.

## Validation

- Focused Project Director and WING MVP rail tests.
- Project Director selected action still targets `WING read-only connection
  proof`.
- Direction guard remains `on_target`.
- Standard backend pytest before commit.
